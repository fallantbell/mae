import os
import argparse
import math
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from data_loader.data_loaders import re10k_DataLoader
from data_loader.re10K_dataset import Re10k_dataset
from torchinfo import summary


from model import *
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pth')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--resume', type=str)

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    # val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    # dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    train_dataloader = re10k_DataLoader(
        "../../../disk2/icchiu",
        load_batch_size,
        shuffle=True,
        num_workers=0,
        mode="train",
        max_interval = 1,        
    )
    test_dataset = Re10k_dataset(
        data_root = "../../../disk2/icchiu",
        mode = "test",
    )

    name = args.exp_name

    # writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
    model_path = f"{name}.pth"
    writer = SummaryWriter(os.path.join('logs', 're10k', f'mae-pretrain_{name}'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(
        image_size = 64,
        patch_size=2,
        mask_ratio=args.mask_ratio).to(device)

    summary(model, [(1, 3, 64, 64),(1,4,4),(1,2,4,4)])
    model = nn.DataParallel(model)

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    start_epoch = 0
    if args.resume!=None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['e']

    step_count = 0
    optim.zero_grad()
    for e in range(start_epoch,args.total_epoch):
        model.train()
        losses = []
        for data in tqdm(iter(train_dataloader)):
            step_count += 1
            img = data['img'].to(device)
            src_img_tensor = data['src_img'].to(device)
            intrinsic = data['intrinsics'].to(device)
            c2w = data['c2w'].to(device)

            prev_img = img[:,0]
            now_img = img[:,1]

            predicted_img, mask = model(prev_img,intrinsic,c2w)
            # loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            criterion = nn.MSELoss()
            loss = criterion(predicted_img,now_img)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            
            
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            prev_img = torch.stack([test_dataset[i]["img"][0] for i in range(4)])
            now_img = torch.stack([test_dataset[i]["img"][1] for i in range(4)])
            val_k = torch.stack([test_dataset[i]["intrinsics"] for i in range(4)])
            val_c2w = torch.stack([test_dataset[i]["c2w"][:2] for i in range(4)])
            prev_img = prev_img.to(device)
            now_img = now_img.to(device)
            val_k = val_k.to(device)
            val_c2w = val_c2w.to(device)
            predicted_val_img, mask = model(prev_img,val_k,val_c2w)
            # predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            # img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = torch.cat([now_img,predicted_val_img, prev_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
        ''' save model '''
        # torch.save(model.state_dict(), model_path)
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
        }, model_path)