import os
from glob import glob

import numpy as np
import torch
import torch.utils
# import torchvision.transforms.v2 as T
# import torchvision.transforms.v2.functional as TF
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils import data

from lightning.pytorch import LightningDataModule
from litdata import StreamingDataLoader, StreamingDataset

from .corruption import corrupt, init_corruption_function
from .dataset_ir import IRImageData, IRRealDataset, IRPairDataset, IRCorruptDataset, IRNoiseDataset
from .dataset_cls import CLSImageData, CLSPairDataset, CLSCorruptDataset, CLSRealDataset
from .dataset_seg import SEGImageData,  CityscapesCorruptDataset, CityscapesPairDataset, SEGRealDataset
from .dataset_det import DetImageData, CoCoCorruptDataset, CoCoPairDataset, CoCoRealDataset, custom_collate_fn

dataset_dict = {
    # Classification
    "ImageNet": {"train": r'./dataset/Classification/ImageNet/train.list',     # (80000, dyn_crp)
                 "val": r'./dataset/Classification/ImageNet/val.list'},        # (20000, pair)
    
    # Segmentation
    "Cityscapes": {"train": r'./dataset/Segmentation/Cityscapes/train.list',          # (2975, dyn_crp)
                   "val": r'./dataset/Segmentation/Cityscapes/val.list'},             # (500, pair)
    "FoggyCityscapes": {"train": r'./dataset/Segmentation/FoggyCityscapes/train.list',# (2975, foggy & dyn_crp)
                        "val": r'./dataset/Segmentation/FoggyCityscapes/val.list'},   # (500, foggy & dyn_crp)
    "Foggy_Zurich": {"val": r'./dataset/Segmentation/Foggy_Zurich/val.list'},         # (40, real-world)
    "ACDC": {"train": r'./dataset/Segmentation/ACDC/train.list',                      # (1600, real-world)
             "val_fog": r'./dataset/Segmentation/ACDC/val_fog.list',
             "val_rain": r'./dataset/Segmentation/ACDC/val_rain.list',
             "val_snow": r'./dataset/Segmentation/ACDC/val_snow.list',
             "val_night": r'./dataset/Segmentation/ACDC/val_night.list',
             "val": r'./dataset/Segmentation/ACDC/val.list',},                        # (406, real-world)

    # Object Detection
    "COCO": {'train': r'./dataset/Detection/COCO/train.list', 
             'val': r'./dataset/Detection/COCO/val.list', 
             'test': r'./dataset/Detection/COCO/test.list'},
    "RTTS": {'test': r'./dataset/Detection/RTTS/test.list'},

    # Image Restoration
    "DIVF2KOST": {'train': r'./dataset/PIR/DIVF2KOST/train.list',  # (13774, dyn_crp)
                  'val': r'./dataset/PIR/DIVF2KOST/val.list'},     # (100, pair)
    ## Derain (RainTrainL, Rain100L, LHPRain, UHDRain, Practical)
    ## Dehaze (SOTS, OTS, 4kID, Unann, NH-Haze)
    ## Denoise (BSD400, WED, BSD68, Urban, CBSD68, Kodak, McMaster, Set12)
    ## Desnow (UHDSnow, Snow100k)                 
    ## Deblur (GoPro, HIDE, RealBlur-J, RealBlur-R)
    ## Lowlight (LOL, DICM, MEF, NPE, LIME, VV)
    ## Others (UDC, poled, toled)
}

class DatasetEngine(LightningDataModule):
    """
    'ir': {'train': ['div2kost',],     # default: air for sota 
           'val': ['inference', 'pair', 'real']}
    'cls': {'train': None,                      # default: ImageNet-C
            'val': ['inference', 'CUB']}
    'seg': {'train': None,                      # default: Cityscape-C
            'val': ['inference', 'fog1', 'fog2', 'fog3', 'Foggy_Zurich', 'ACDC']}
    """
    def __init__(
        self,
        task: str,
        train: dict,
        val: dict,
        crp_mode: str = "common", 
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.task = task
        self.train = train
        self.val = val
        # others
        self.resolution = self.train['resolution']
        self.crp_mode = crp_mode
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor       

    def train_dataloader(self):
        if self.task == 'mtl':
            if self.train["type"] == 'all':
                trainsets = [
                    # Corrupt IN-HD384k
                    CLSCorruptDataset(
                        CLSImageData(dataset_dict['ImageNet']['train']),
                        resolution=self.resolution, is_train=True, crp_mode=self.crp_mode
                    ),
                    # Corrupt Cityscapes
                    CityscapesCorruptDataset(
                        SEGImageData(dataset_dict['FoggyCityscapes']['train']),
                        resolution=self.resolution, is_train=True, crp_mode=self.crp_mode
                    ),
                    # Corrupt DF2kOST
                    IRCorruptDataset(
                        IRImageData(dataset_dict['DIVF2KOST']['train']),
                        resolution=self.resolution, is_train=True, crp_mode=self.crp_mode
                    )
                ]
                dataset = data.ConcatDataset(trainsets)
                """
                Num. of Sample: 80000, 2975, 13774 = 96749
                Ratio: 1.2093625, 32.5206722689, 7.02403078263 = 40.7540655515
                Weights: 0.02967464677, 0.7979, 0.1724
                """
                weights = (
                    [0.2] * len(trainsets[0])
                    + [10] * len(trainsets[1])
                    + [1] * len(trainsets[2])
                )
                sampler = data.WeightedRandomSampler(
                    weights=weights, num_samples=len(dataset), replacement=True
                )
            else:
                raise KeyError("In MTL Task, training dataloader {} is not defined!"%(self.train["type"]))
            
            loader = data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.train["batch_size"],
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                drop_last=True,
                pin_memory=True,
            )

        elif self.task == 'ir':
            if self.train["type"] == 'div2kost':
                trainsets = [
                    # Corrupt DF2kOST
                    IRCorruptDataset(
                        IRImageData(dataset_dict['DIVF2KOST']['train']),
                        resolution=self.resolution, is_train=True, crp_mode=self.crp_mode
                    )
                ]
                dataset = data.ConcatDataset(trainsets)
                weights = (
                    [1] * len(trainsets[0])
                )
                sampler = data.WeightedRandomSampler(
                    weights=weights, num_samples=len(dataset), replacement=True
                )
            else:
                raise NotImplemented

            loader = data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.train["batch_size"],
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                drop_last=True,
                pin_memory=True,
            )

        elif self.task == 'cls':
            if self.train["type"] == 'all':
                dataset = CLSCorruptDataset(
                    CLSImageData(dataset_dict['ImageNet']['train_all']),
                    resolution=self.resolution, is_train=True, crp_mode=self.crp_mode
                )
            else:
                dataset = CLSCorruptDataset(
                    CLSImageData(dataset_dict['ImageNet']['train']),
                    resolution=self.resolution, is_train=True, crp_mode=self.crp_mode
                )
            loader = data.DataLoader(
                dataset,
                batch_size=self.train["batch_size"],
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            )

        elif self.task == 'seg':
            dataset = CityscapesCorruptDataset(
                SEGImageData(dataset_dict['FoggyCityscapes']['train']),
                resolution=self.resolution, is_train=True, crp_mode=self.crp_mode
            )
            loader = data.DataLoader(
                dataset,
                batch_size=self.train["batch_size"],
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            )

        elif self.task == 'det':
            dataset = CoCoCorruptDataset(
                DetImageData(dataset_dict['COCO']['train']),
                resolution=self.resolution, is_train=True, crp_mode=self.crp_mode
            )
            loader = data.DataLoader(
                dataset,
                batch_size=self.train["batch_size"],
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                collate_fn = custom_collate_fn
            )

        return loader

    def val_dataloader(self, ):
        if self.task == 'mtl':
            if self.val["type"] == "val":
                valsets = [
                    # # IR
                    IRPairDataset(
                        IRImageData(dataset_dict['DIVF2KOST']['val']),
                        resolution=self.resolution, is_train=False),
                    # CLS
                    CLSPairDataset(
                        CLSImageData(dataset_dict['ImageNet']['val']),
                        resolution=self.resolution, is_train=False),
                    # SEG
                    CityscapesPairDataset(
                        SEGImageData(dataset_dict['Cityscapes']['val']),
                        resolution=self.resolution, is_train=False
                    )
                ]
                dataset = data.ConcatDataset(valsets)
            else:
                raise KeyError("In MTL Task, val dataloader {} is not defined!"%(self.train["type"]))
            
            loader = data.DataLoader(
                dataset,
                batch_size=self.val["batch_size"],
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )

        elif self.task == 'ir':
            if self.val["type"] == "val":
                valset = IRPairDataset(
                        IRImageData(dataset_dict['DIVF2KOST']['val']),
                        resolution=self.resolution, is_train=False)
            elif self.val["type"] == "pair":
                datasets = []
                for dset in self.val["val_list"]:
                    datasets.append(IRPairDataset(
                                        IRImageData(dataset_dict[dset]['test']),
                                        resolution=self.resolution, is_train=False))
                valset = data.ConcatDataset(datasets)
            elif self.val["type"] == "real":
                datasets = []
                for dset in self.val["val_list"]:
                    datasets.append(IRRealDataset(
                                        IRImageData(dataset_dict[dset]['test']),
                                        resolution=self.resolution, is_train=False))
                valset = data.ConcatDataset(datasets)
            elif self.val["type"] == "noise":
                datasets = []
                for dset in self.val["val_list"]:
                    datasets.append(IRNoiseDataset(
                                        IRImageData(dataset_dict[dset]['test']),
                                        resolution=self.resolution, is_train=False, noise_sigma=50)) 
                    # noise_sigma = [15, 20, 50]
                valset = data.ConcatDataset(datasets)
            else:
                raise NotImplemented

            loader = data.DataLoader(
                valset,
                batch_size=self.val["batch_size"],
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )

        elif self.task == 'cls':
            if self.val["type"] == "val":
                dataset = CLSPairDataset(
                    CLSImageData(dataset_dict['ImageNet']['val']),
                    resolution=self.resolution, is_train=False
                )
            elif self.val["type"] == "CUB":
                dataset = CLSPairDataset(
                    CLSImageData(dataset_dict['CUB']['val']),
                    resolution=self.resolution, is_train=False
                )
            else:
                raise NotImplemented

            loader = data.DataLoader(
                dataset,
                batch_size=self.val["batch_size"],
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )

        elif self.task == 'seg':
            if self.val["type"] == 'val':
                valset = CityscapesPairDataset(
                    SEGImageData(dataset_dict['Cityscapes']['val']),
                    resolution=self.resolution, is_train=False
                )
            elif self.val["type"] in ['fog1', 'fog2', 'fog3']:
                valset = CityscapesPairDataset(
                    SEGImageData(dataset_dict['FoggyCityscapes']['val']),
                    resolution=self.resolution, is_train=False, crp_mode=self.val["type"]
                )
            elif self.val["type"] in ['Foggy_Zurich', 'ACDC']:
                valset = SEGRealDataset(
                    SEGImageData(dataset_dict[self.val["type"]]['val']),
                    resolution=self.resolution, is_train=False
                )
            elif self.val["type"] in ["ACDC_fog", "ACDC_rain", "ACDC_snow", "ACDC_night"]:
                valset = SEGRealDataset(
                    SEGImageData(dataset_dict['ACDC']["val_%s"%(self.val["type"].split("_")[-1])]),
                    resolution=self.resolution, is_train=False
                )
            else:
                raise NotImplemented

            loader = data.DataLoader(
                    valset,
                    batch_size=self.val["batch_size"],
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                )

        elif self.task == 'det':
            if self.val["type"] == 'val':
                valset = CoCoPairDataset(
                    DetImageData(dataset_dict['COCO']['val']),
                    resolution=self.resolution, is_train=False
                )

            elif self.val["type"] == 'RTTS':
                valset = CoCoRealDataset(
                    DetImageData(dataset_dict['RTTS']['test']),
                    resolution=self.resolution, is_train=False, crp_mode='common', 
                    ann="RTTS"
                )
            
            loader = data.DataLoader(
                    valset,
                    batch_size=self.val["batch_size"],
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                    collate_fn = custom_collate_fn
                )

        return loader

if __name__ == "__main__":
    from torchvision.utils import save_image
    dataenginer = DatasetEngine(task='ir', 
                                train_type='in1k', 
                                val_type='inference', val_list=[], 
                                resolution=512, crp_mode="common", batch_size=1, val_batch_size=1, num_workers=4)
    train_dl = dataenginer.train_dataloader()
    val_dl = dataenginer.val_dataloader()
    with torch.no_grad():
        print("Train", len(train_dl))
        for ct, train in enumerate(train_dl, 1):
            print("%d / %d"%(ct, len(train_dl)), end='\r')
            if ct > 50: 
                break
       
        vis_lq, vis_hq = [], []
        T2P_trsnf = TF.to_pil_image
        P2T_trsnf = T.ToTensor()
        for ct, val in enumerate(val_dl, 1):
            print("%d / %d"%(ct, len(val_dl)), end='\r')
            lq, hq, label, fname = val 
            vis_lq.append(P2T_trsnf(TF.crop(T2P_trsnf(lq[0]), 0, 0, 512, 512)).unsqueeze(0))
            vis_hq.append(P2T_trsnf(TF.crop(T2P_trsnf(hq[0]), 0, 0, 512, 512)).unsqueeze(0))
            if ct == 8:
                break
        print()
        save_image(torch.cat(vis_lq+vis_hq), "vis.png", nrow=8)