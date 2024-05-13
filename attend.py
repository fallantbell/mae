import torch
import torch.nn as nn
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

class Epipolar_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, do_epipolar,do_bidirectional_epipolar, qkv_bias = True, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.do_epipolar = do_epipolar
        self.do_bidirectional_epipolar = do_bidirectional_epipolar

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)
    
    def get_epipolar(self,b,h,w,k,src_c2w,target_c2w):
        H = h
        W = H*16/9  #* 原始圖像為 16:9

        k = k.to(dtype=torch.float32)
        src_c2w=src_c2w.to(dtype=torch.float32)
        target_c2w=target_c2w.to(dtype=torch.float32)

        #* unormalize intrinsic 

        k[:,0] = k[:,0]*W
        k[:,1] = k[:,1]*H

        k[:,0,2] = h/2
        k[:,1,2] = h/2

        device = k.device

        #* 創建 h*w 的 uv map
        x_coords, y_coords = torch.meshgrid(torch.arange(h), torch.arange(w))
        coords_tensor = torch.stack((x_coords.flatten(), y_coords.flatten(), torch.ones_like(x_coords).flatten()), dim=1)
        coords_tensor[:,[0,1]] = coords_tensor[:,[1,0]]
        coords_tensor = coords_tensor.to(dtype=torch.float32)
        coords_tensor = repeat(coords_tensor, 'HW p -> b HW p', b=b)

        x_coords = x_coords.to(device)
        y_coords = y_coords.to(device)
        coords_tensor = coords_tensor.to(device)

        k_3x3 = k[:,0:3,0:3]
        src_c2w_r = src_c2w[:,0:3,0:3]
        src_c2w_t = src_c2w[:,0:3,3]
        target_c2w_r = target_c2w[:,0:3,0:3]
        target_c2w_t = target_c2w[:,0:3,3]
        target_w2c_r = torch.linalg.inv(target_c2w_r)
        target_w2c_t = -target_c2w_t

        cx = k_3x3[:,0,2].view(b, 1)
        cy = k_3x3[:,1,2].view(b, 1)
        fx = k_3x3[:,0,0].view(b, 1)
        fy = k_3x3[:,1,1].view(b, 1)
        coords_tensor[...,0] = (coords_tensor[...,0]-cx)/fx
        coords_tensor[...,1] = (coords_tensor[...,1]-cy)/fy

        #* 做 H*W 個點的運算
        coords_tensor = rearrange(coords_tensor, 'b hw p -> b p hw') 
        point_3d_world = torch.matmul(src_c2w_r,coords_tensor)              #* 相機坐標系 -> 世界座標
        point_3d_world = point_3d_world + src_c2w_t.unsqueeze(-1)           #* 相機坐標系 -> 世界座標
        point_2d = torch.matmul(target_w2c_r,point_3d_world)                #* 世界座標 -> 相機座標
        point_2d = point_2d + target_w2c_t.unsqueeze(-1)                    #* 世界座標 -> 相機座標
        pi_to_j = torch.matmul(k_3x3,point_2d)                              #* 相機座標 -> 平面座標

        #* 原點的計算
        oi = torch.zeros(3).to(dtype=torch.float32)
        oi = repeat(oi, 'p -> b p', b=b)
        oi = oi.unsqueeze(-1)
        oi = oi.to(device)
        point_3d_world = torch.matmul(src_c2w_r,oi)
        point_3d_world = point_3d_world + src_c2w_t.unsqueeze(-1)  
        point_2d = torch.matmul(target_w2c_r,point_3d_world)
        point_2d = point_2d + target_w2c_t.unsqueeze(-1)  
        oi_to_j = torch.matmul(k_3x3,point_2d)
        oi_to_j = rearrange(oi_to_j, 'b c p -> b p c') #* (b,3,1) -> (b,1,3)

        #* 除以深度
        pi_to_j_unnormalize = rearrange(pi_to_j, 'b p hw -> b hw p') 
        pi_to_j = pi_to_j_unnormalize / (pi_to_j_unnormalize[..., -1:] + 1e-6)   #* (b,hw,3)
        # pi_to_j = pi_to_j_unnormalize / pi_to_j_unnormalize[..., -1:]
        oi_to_j = oi_to_j / oi_to_j[..., -1:]   #* (b,1,3)

        # print(f"pi_to_j: {pi_to_j[0,9]}")
        # print(f"oi_to_j: {oi_to_j[0,0]}")

        #* 計算feature map 每個點到每個 epipolar line 的距離
        coords_tensor = torch.stack((x_coords.flatten(), y_coords.flatten(), torch.ones_like(x_coords).flatten()), dim=1)
        coords_tensor[:,[0,1]] = coords_tensor[:,[1,0]]
        coords_tensor = coords_tensor.to(dtype=torch.float32) # (4096,3)
        coords_tensor = repeat(coords_tensor, 'HW p -> b HW p', b=b)
        coords_tensor = coords_tensor.to(device)

        oi_to_pi = pi_to_j - oi_to_j            #* h*w 個 epipolar line (b,hw,3)
        oi_to_coord = coords_tensor - oi_to_j   #* h*w 個點   (b,hw,3)

        ''''
            #* 這裡做擴展
            oi_to_pi    [0,0,0]     ->      oi_to_pi_repeat     [0,0,0]
                        [1,1,1]                                 [0,0,0]
                        [2,2,2]                                 [1,1,1]
                                                                [1,1,1]
                                                                .
                                                                .
                                                                .

            oi_to_coord     [0,0,0]     ->      oi_to_coord_repeat      [0,0,0]
                            [1,1,1]                                     [1,1,1]
                            [2,2,2]                                     [2,2,2]
                                                                        [0,0,0]
                                                                        .
                                                                        .
                                                                        .
        '''
        oi_to_pi_repeat = repeat(oi_to_pi, 'b i j -> b i (repeat j)',repeat = h*w)
        oi_to_pi_repeat = rearrange(oi_to_pi_repeat,"b i (repeat j) -> b (i repeat) j", repeat = h*w)
        oi_to_coord_repeat = repeat(oi_to_coord, 'b i j -> b (repeat i) j',repeat = h*w)


        area = torch.cross(oi_to_pi_repeat,oi_to_coord_repeat,dim=-1)     #* (b,hw*hw,3)
        area = torch.norm(area,dim=-1 ,p=2)
        vector_len = torch.norm(oi_to_pi_repeat, dim=-1, p=2)
        distance = area/vector_len

        distance_weight = 1 - torch.sigmoid(50*(distance-0.5))

        epipolar_map = rearrange(distance_weight,"b (hw hw2) -> b hw hw2",hw = h*w)

        #* 如果 max(1-sigmoid) < 0.5 
        #* => min(distance) > 0.05 
        #* => 每個點離epipolar line 太遠
        #* => epipolar line 不在圖中
        #* weight map 全設為 1 
        max_values, _ = torch.max(epipolar_map, dim=-1)
        mask = max_values < 0.5
        epipolar_map[mask.unsqueeze(-1).expand_as(epipolar_map)] = 1

        if (torch.any(torch.isnan(epipolar_map)) or
            torch.any(torch.isnan(distance)) or
            torch.any(torch.isnan(distance_weight)) or
            torch.any(torch.isnan(area)) or
            torch.any(torch.isnan(vector_len)) or        
            torch.any(torch.isnan(distance_weight)) or
            torch.any(torch.isnan(oi_to_pi_repeat)) or
            torch.any(torch.isnan(oi_to_coord_repeat))):
            print(f"find nan !!!")
            print(f"epipolar_map: {torch.any(torch.isnan(epipolar_map))}")
            print(f"distance_weight: {torch.any(torch.isnan(distance_weight)) }")
            print(f"distance: {torch.any(torch.isnan(distance)) }")
            print(f"vector_len: {torch.any(torch.isnan(vector_len)) }")
            print(f"area: {torch.any(torch.isnan(area)) }")
            print(f"oi_to_pi_repeat: {torch.any(torch.isnan(oi_to_pi_repeat))}")
            print(f"oi_to_coord_repeat: {torch.any(torch.isnan(oi_to_coord_repeat))}")
            print(f"pi_to_j: {torch.any(torch.isnan(pi_to_j))}")
            print(f"oi_to_j: {torch.any(torch.isnan(oi_to_j))}")
            print(f"pi_to_j_unnormalize has zero: {torch.any(torch.eq(pi_to_j_unnormalize[...,-1:],0))}")
            print(" ")
            print("break !")
            os._exit(0)



        return epipolar_map


    def forward(self, x, src_encode,intrinsic = None,c2w = None,pre_weightmap = None):
        # b, c, h, w = x.shape
        b,_,_ = x.shape

        h = 32
        w = 32

        """
        q: (b (h w) d)
        k: (b (h w) d)
        v: (b (h w) d)
        """

        q = x
        k = src_encode
        v = src_encode

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        #* 一般的 cross attention -> 得到 attention map
        cross_attend = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)

        weight_map = torch.ones_like(cross_attend)

        if pre_weightmap == None:

            if self.do_epipolar:
                #* 得到 epipolar weighted map (B,HW,HW)
                epipolar_map = self.get_epipolar(b,h,w,intrinsic.clone(),c2w[1],c2w[0])

                epipolar_map = repeat(epipolar_map,'b hw hw2 -> (b repeat) hw hw2',repeat = self.heads)

                weight_map = weight_map*epipolar_map
            
            if self.do_bidirectional_epipolar:
                #* 做反方向的epipolar
                epipolar_map = self.get_epipolar(b,h,w,intrinsic.clone(),c2w[0],c2w[1])

                epipolar_map_transpose = epipolar_map.permute(0,2,1)

                epipolar_map = repeat(epipolar_map_transpose,'b hw hw2 -> (b repeat) hw hw2',repeat = self.heads)

                weight_map = weight_map*epipolar_map

        else:
            weight_map = pre_weightmap

        cross_attend = cross_attend * weight_map
        att = cross_attend.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        # z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z.contiguous(),weight_map