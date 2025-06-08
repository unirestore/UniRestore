import os
from glob import glob

import numpy as np
import random
import torch
import torch.utils
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
# import torchvision.transforms as T
# import torchvision.transforms.functional as TF
from litdata import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset
from lightning.pytorch import LightningDataModule

from torchvision.io import ImageReadMode, decode_image, read_image
from torchvision.io import read_file

from torch import Tensor
from torch.utils import data
from PIL import Image
from .utils import open_image, to_image
from .corruption import corrupt, init_corruption_function

class IRImageData(data.Dataset):
    # [degradation, clean, label]
    def __init__(self, listfile: str) -> None:
        super().__init__()
        self.listfile = listfile
        self.paths = []
        with open(listfile) as fin:
            for line in fin:
                line = line.strip().split()
                self.paths.append(line)

        self.paths = sorted(self.paths)
        print("ImageDataset:", listfile, len(self.paths))

    def __getitem__(self, index:int):
        lq_pth, hq_pth, label = self.paths[index]
        
        if lq_pth == 'None':
            lq_pth = None
        else: 
            fname = os.path.basename(lq_pth)
        
        if hq_pth == 'None':
            hq_pth = None
        else: 
            fname = os.path.basename(hq_pth)
        return lq_pth, hq_pth, label, fname

    def __len__(self):
        return len(self.paths)

# Different Type of Dataset
class IRRealDataset(data.Dataset):
    '''
    Single Real World Degradation Data
    '''
    def __init__(self, dataset, resolution=512, is_train=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.is_train = is_train
        self.resolution = resolution
        self.resize = T.Resize((resolution,), interpolation=3, antialias=False)
        self.transform = T.Compose(
            [
                T.RandomCrop((resolution, resolution)),
                T.RandomHorizontalFlip(),
            ]
        )
        self.final_transform = T.Compose(
            [
                T.ToDtype(torch.float32, scale=True),
                T.ToTensor(),
            ]
        )

    def __getitem__(self, index: int):
        # Read Sample
        lq_path, _, _, fname = self.dataset[index]
        try:
            lq = read_image(lq_path, mode=ImageReadMode.RGB)
        except:
            # Step 1: Load the BMP file using Pillow
            img = Image.open(lq_path).convert("RGB")  # Ensure 3 channels (RGB)

            # Step 2: Convert to a PyTorch tensor with shape [C, H, W] and dtype uint8
            lq = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            lq = lq.view(img.height, img.width, 3).permute(2, 0, 1)

        # Training Transform
        if self.is_train:
            # resize and augment
            _, h, w = hq.shape
            if h < self.resolution or w < self.resolution:
                lq = self.resize(lq)
            lq = self.transform(lq)
        
        # Final Transform
        lq = self.final_transform(lq)
        return lq, np.nan, np.nan, fname, 'ir'

    def __len__(self):
        return len(self.dataset)

class IRPairDataset(data.Dataset):
    def __init__(self, dataset, resolution=512, is_train=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.is_train = is_train
        self.resolution = resolution
        self.resize = T.Resize((resolution,), interpolation=3, antialias=False)
        self.final_transform = T.Compose(
            [
                T.ToDtype(torch.float32, scale=True),
                T.ToTensor(),
            ]
        )

    def __getitem__(self, index: int):
        # Read Sample
        lq_pth, hq_pth, _, fname = self.dataset[index]
        hq = read_image(hq_pth, mode=ImageReadMode.RGB)
        lq = read_image(lq_pth, mode=ImageReadMode.RGB)
        
        # Training Transform
        if self.is_train:
            # resize and augment
            _, h, w = hq.shape
            if h < self.resolution or w < self.resolution:
                hq = self.resize(lq)
                lq = self.resize(lq)
            hq, lq = self.pair_aug_transform(hq, lq)
            
        # Final Transform
        hq = self.final_transform(hq)
        lq = self.final_transform(lq)
        return lq, hq, np.nan, fname, 'ir'

    def __len__(self):
        return len(self.dataset)

    def pair_aug_transform(self, hq, lq):
        # RandomCrop
        i, j, h, w = T.RandomCrop.get_params(hq, output_size=(self.resolution, self.resolution))
        hq = TF.crop(hq, i, j, h, w)
        lq = TF.crop(lq, i, j, h, w)

        # RandomHorizontalFlip
        if random.random() > 0.5:
            hq = TF.hflip(hq)
            lq = TF.hflip(lq)
            
        return hq, lq

class IRCorruptDataset(data.Dataset):
    def __init__(self, dataset, resolution=512, is_train=True, crp_mode="common", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.resolution = resolution
        self.resize = T.Resize((resolution,), interpolation=3, antialias=False)
        self.transform = T.Compose(
            [
                T.RandomCrop((resolution, resolution)),
                T.RandomHorizontalFlip(),
            ]
        )
        self.final_transform = T.Compose(
            [
                T.ToDtype(torch.float32, scale=True),
                T.ToTensor(),
            ]
        )
        self.is_train = is_train
        self.corruption_funcs = init_corruption_function(crp_mode)

    def __getitem__(self, index: int):
        # read hq sample
        _, hq_pth, _, fname = self.dataset[index]
        hq = read_image(hq_pth, mode=ImageReadMode.RGB)

        # training transform
        if self.is_train:  
            # resize & augment
            _, h, w = hq.shape
            if h < self.resolution or w < self.resolution:
                hq = self.resize(hq)
            hq = self.transform(hq)

        # generate corruption sample
        corruption_mode = np.random.choice(self.corruption_funcs)
        severity = np.random.choice(5, p=[0.05, 0.25, 0.4, 0.25, 0.05]) + 1  # [1, 5]
        lq = self._degrade_image(hq, corruption_mode, severity)

        # final transform
        hq, lq = TF.to_image(hq), TF.to_image(lq)
        hq = self.final_transform(hq)
        lq = self.final_transform(lq)
        return lq, hq, np.nan, fname, 'ir'

    def _degrade_image(
        self,
        hq: torch.Tensor,
        corruption_mode: str, 
        severity: int
    ):
        """
        Resize to [patch_size//4, patch_size], corrupt, then resize back to original resolution.
        Input:
            hq: [0, 255] uint8 tensor (C, H, W)
        Output:
            lq: [0, 255] uint8 tensor (C, H, W)
        """
        if corruption_mode == "clean":
            return hq

        h, w = hq.shape[-2:]
        # random resize short edge to [resolution//4, resolution]
        size = int(torch.randint(self.resolution // 4, self.resolution, ()))
        lq = TF.resize(hq, (size,))
        # to np
        lq = lq.permute(1, 2, 0).numpy()  # [0, 255] uint8 np (H, W, C)
        # process lq image
        lq = corrupt(lq, corruption_name=corruption_mode, severity=severity)
        # to tensor
        lq = TF.to_image(lq)
        # resize back to original resolution
        lq = TF.resize(lq, (h, w))
        return lq

    def __len__(self):
        return len(self.dataset)

class IRNoiseDataset(data.Dataset):
    def __init__(self, dataset, resolution=512, is_train=True, noise_sigma=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.is_train = is_train
        self.resolution = resolution
        self.resize = T.Resize((resolution,), interpolation=3, antialias=False)
        # self.transform = T.Compose(
        #     [
        #         T.RandomCrop((self.resolution, self.resolution)),
        #         T.RandomHorizontalFlip(),
        #     ]
        # )
        self.final_transform = T.Compose(
            [
                T.ToDtype(torch.float32, scale=True),
                T.ToTensor(),
            ]
        )
        if noise_sigma:
            self.sigmas = [noise_sigma]
        else:
            self.sigmas = [15, 25, 50]

    def _add_gaussian_noise(self, img: Tensor, sigma: int):
        """
        Args:
            img: [0, 255] uint8 tensor (C, H, W)
            sigma: noise level [15, 25, 50, 100, ...]
        """
        noise = torch.randn(size=img.size())
        noised = torch.clamp(img + sigma * noise, 0, 255).to(torch.uint8)
        return noised

    def __getitem__(self, index: int):
        # read sample
        _, hq_path, _, fname = self.dataset[index]
        hq = read_image(hq_path, mode=ImageReadMode.RGB)
        
        # synthesize lq
        lq = self._add_gaussian_noise(hq, np.random.choice(self.sigmas))
        hq, lq = TF.to_image(hq), TF.to_image(lq)
        
        # Training Transform
        if self.is_train:
            # resize & augment
            _, h, w = hq.shape
            if h < self.resolution or w < self.resolution:
                hq = self.resize(hq)
                lq = self.resize(lq)
            hq, lq = self.pair_aug_transform(hq, lq)

        # Final Transform
        hq = self.final_transform(hq)
        lq = self.final_transform(lq)
        return lq, hq, np.nan, fname, 'ir'

    def __len__(self):
        return len(self.dataset)

    def pair_aug_transform(self, hq, lq):
        # RandomCrop
        i, j, h, w = T.RandomCrop.get_params(hq, output_size=(self.resolution, self.resolution))
        hq = TF.crop(hq, i, j, h, w)
        lq = TF.crop(lq, i, j, h, w)

        # RandomHorizontalFlip
        if random.random() > 0.5:
            hq = TF.hflip(hq)
            lq = TF.hflip(lq)
            
        return hq, lq

if __name__ == "__main__":
    from torchvision.utils import save_image
    data_dict = {"DIVF2KOST": {'train': r'/mnt/200/b/Dataset/IR/HR/train.list',  # (13774, dyn_crp)
                                'val': r'/mnt/200/b/Dataset/IR/HR/val.list'}}    # (100, pair)
    
    train_data = IRCorruptDataset(
        IRImageData(data_dict['DIVF2KOST']['train']),
        resolution=512, is_train=True
    )
    weights = ([1] * len(train_data))
    sampler = data.WeightedRandomSampler(
        weights=weights, num_samples=len(train_data), replacement=True
    )
    train_dl = data.DataLoader(
        train_data,
        sampler=sampler,
        batch_size=1,
        num_workers=1,
        # prefetch_factor=2,
        drop_last=False,
        pin_memory=True,
    )

    val_data = IRPairDataset(
        IRImageData(data_dict['DIVF2KOST']['val']),
        resolution=512, is_train=True
    )
    val_dl = data.DataLoader(
        val_data,
        batch_size=1,
        num_workers=1,
        # prefetch_factor=2,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    with torch.no_grad():
        print("Train", len(train_dl))
        for ct, train in enumerate(train_dl, 1):
            print("%d / %d"%(ct, len(train_dl)), end='\r')
            if ct > 50: 
                break
        print()

        print("Val", len(val_dl))        
        vis_lq, vis_hq = [], []
        for ct, val in enumerate(val_dl, 1):
            print("%d / %d"%(ct, len(val_dl)), end='\r')
            lq, hq, label, fname = val 
            vis_lq.append(lq)
            vis_hq.append(hq)
            if ct == 8:
                break
        print()
        save_image(torch.cat(vis_lq+vis_hq), "vis_ir.png", nrow=8)