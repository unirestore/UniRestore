import torch
import torchvision.transforms.v2 as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from .deeplabv3 import modeling
from .refinenetlw.refinenetlw import rf_lw101


def build_segmentation_model(model_type: str):
    # dlv3pr50, dlv3pr50_ft, rflwr101, rflwr101_ft, rflwr101_fifo
    if "dlv3pr50" in model_type:
        model = modeling.__dict__["deeplabv3plus_resnet50"](
            num_classes=19,
            output_stride=16,
            pretrained_backbone=False,
        )
        # load ckpt
        if model_type.endswith("_ft"):
            ckpt_path = "logs/segmentation/deeplabv3plus_ep500_csmix/version_1/checkpoints/epoch=11-val=0.3284.ckpt"
        else:
            ckpt_path = "src/models/segmentation/best_deeplabv3plus_resnet50_cityscapes_os16.pth"

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state" in ckpt:
            ckpt = ckpt["model_state"]
        else:
            ckpt = ckpt["state_dict"]
            ckpt = {k[8:]: v for k, v in ckpt.items() if k.startswith("model.1.")}
        model.load_state_dict(ckpt)
        return nn.Sequential(
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD), model
        )
    elif "rflwr101" in model_type:
        model = rf_lw101(num_classes=19, imagenet=False)
        # load ckpt
        if model_type.endswith("_ft"):
            ckpt_path = "logs/segmentation/rf_lw101_ep500_csmix/version_0/checkpoints/epoch=17-val=0.3919.ckpt"
        elif model_type.endswith("_fifo"):
            ckpt_path = "src/models/segmentation/FIFO_final_model.pth"
        else:
            ckpt_path = "src/models/segmentation/Cityscapes_pretrained_rflwr101.pth"

        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        nn.modules.utils.consume_prefix_in_state_dict_if_present(ckpt, prefix="model.")
        model.load_state_dict(ckpt)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model
