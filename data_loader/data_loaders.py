from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from data_loader.re10K_dataset import Re10k_dataset


class re10k_DataLoader(DataLoader):
    """
    ACID dataloader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, 
                validation_split=0.0, num_workers=1, 
                mode="train",max_interval = 5,
                infer_len = 20, do_latent = False,
                ):

        self.data_dir = data_dir

        if mode == "train":
            self.dataset = Re10k_dataset(self.data_dir,"train",max_interval,do_latent = do_latent)
        elif mode == "validation":
            self.dataset = Re10k_dataset(self.data_dir,"validation")
        else:
            self.dataset = Re10k_dataset(self.data_dir,"test",infer_len=infer_len,do_latent = do_latent)
        

        collate_fn=default_collate

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            "drop_last": True,
        }

        super().__init__(**self.init_kwargs)