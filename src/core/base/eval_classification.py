from typing import Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from einops import rearrange, repeat
from modules.rvt.robust_models import rvt_base_plus
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import resnet18, resnet50, resnet101, swin_v2_b, vgg16, vit_b_16, efficientnet_v2_l
import torchvision
import timm
from transformers import pipeline
from .base import BaseEvaluator
from .task import TaskMetric


class ClassificationEvaluator(BaseEvaluator):
    """
    Task-specific Evaluator: Classification
        1. setup task tools
        2. define evaluation_step
    """

    def __init__(self, save_image: bool = False, eval_mode: str = 'single', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_image = save_image
        self.eval_mode = eval_mode

    def configure_model(self):
        super().configure_model()
        # Define task_metric
        eval_types = ["hq", "lq"]
        if self.eval_mode == "all":
            model_types = ["r50v1", "r101v1", "vgg", "swin", "vit", "rvt"]
        elif self.eval_mode == "all_ft":
            model_types = ["r50v1_ft", "r50v2_ft", "vgg_ft", "swin_ft", "vit_ft", "rvt"]
        elif self.eval_mode == "single":
            model_types = ["r50v1", "r50v2"]
        elif self.eval_mode == "bare":
            model_types = None
        elif self.eval_mode == "CUB":
            model_types = ['cub_r18', 'cub_r50', 'cub_conv', 'cub_vitb', 'cub_swin'] # 'cub_vitL'
        else:
            raise ValueError(f"Unknown eval_mode: {self.eval_mode}")

        self.task_metric = ClassificationMetric(
            eval_types=eval_types,
            model_types=model_types,
        )
        self.task_metric.requires_grad_(False).eval()

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        # 1. parse batch
        lq, hq, gt, fname, task = batch
        lq = self.crop_tensor(lq)
        hq = self.crop_tensor(hq)
        
        # 2. Inference
        preds = self.forward([hq, lq], 'cls')
        preds = [pred.mul(255).round_().clamp_(0, 255).div_(255) for pred in preds]
        # 3. Visualize
        if self.eval_mode != "bare" and batch_idx == 0:
            self.visualize(hq, lq, preds)
        # 4. Update metrics
        self.task_metric.update_metrics(torch.cat(preds, dim=0), gt)

        # 5. save image
        if self.save_image and len(fname) == 1:
            enh_hq = preds[0]
            enh_lq = preds[1]
            logdir = self.logger.log_dir
            torchvision.utils.save_image(
                enh_hq, os.path.join(logdir, "hq", f"{fname[0]}.png")
            )
            torchvision.utils.save_image(
                enh_lq, os.path.join(logdir, "lq", f"{fname[0]}.png")
            )

    def on_validation_epoch_start(self) -> None:
        logdir = self.logger.log_dir
        if self.save_image:
            os.makedirs(os.path.join(logdir, "hq"), exist_ok=True)
            os.makedirs(os.path.join(logdir, "lq"), exist_ok=True)

    def on_validation_epoch_end(self):
        metrics = super().on_validation_epoch_end()
        # val_monitor
        if self.eval_mode == "bare":
            acc = metrics["val_lq/acc"]
        elif self.eval_mode == "all_ft":
            acc = metrics["val_lq/r50v1_ft"]
        elif self.eval_mode == 'CUB':
            acc = metrics["val_lq/cub_r50"]
        else:
            acc = metrics["val_lq/r50v1"]
        # sync_dist to avoid warning
        self.log("val_monitor", acc, sync_dist=True)
        print(acc)

    def crop_tensor(self, image, base=16):
        # center crop
        if image.ndim == 3:
            b, h, w = image.shape
            crop_h = h
            crop_w = w
            if h > 960:
                crop_h = 960
            # else:
            #     crop_h = (h // base) * base
            if w > 1664:
                crop_w = 1664
            # else:
            #     crop_w = (w // base) * base
            return image[:, h//2 - crop_h//2 : h//2 + crop_h//2, w//2 - crop_w//2 : w//2 + crop_w//2]
        elif image.ndim == 4:
            b, c, h, w = image.shape
            crop_h = h
            crop_w = w
            if h > 960:
                crop_h = 960
            # else:
            #     crop_h = (h // base) * base
            if w > 1664:
                crop_w = 1664
            # else:
            #     crop_w = (w // base) * base
            return image[:, :, h//2 - crop_h//2 : h//2 + crop_h//2, w//2 - crop_w//2 : w//2 + crop_w//2]
        else:
            raise NotImplemented
            
    def visualize(self, hq, lq, preds, n_img=8):
        if self.trainer.is_global_zero and self.logger:
            writer = self.logger.experiment
            # log input only once
            if self.global_step == 0:
                vis = torch.cat([hq[:n_img], lq[:n_img]])
                vis = TF.resize(vis, (256, 256)).clamp(0, 1)
                writer.add_images(
                    "val/input",
                    vis.float(),
                    global_step=self.global_step,
                    dataformats="NCHW",
                )

            # log predition
            vis = torch.cat([img[:n_img] for img in preds])
            vis = TF.resize(vis, (256, 256)).clamp(0, 1)
            writer.add_images(
                "val/preds",
                vis.float(),
                global_step=self.global_step,
                dataformats="NCHW",
            )


class ClassificationLoss(nn.Module):
    def __init__(self, model_type: str):
        super().__init__()
        # downstream task
        self.preprocess = T.Compose(
            [
                T.Resize((224, 224)),
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        if model_type == "r50v1":
            self.model = resnet50(weights="IMAGENET1K_V1")
        elif model_type == "r50v2":
            self.model = resnet50(weights="IMAGENET1K_V2")
        elif model_type == "vgg16":
            self.model = vgg16(weights="IMAGENET1K_V1")
        elif model_type == "swin":
            self.model = swin_v2_b(weights="IMAGENET1K_V1")
        elif model_type == "vit":
            self.model = vit_b_16(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unknown model name: {model_type}")

        self.model.requires_grad_(False).eval()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        logits = self.model(self.preprocess(input))
        return F.cross_entropy(logits, target)


class ClassificationMetric(TaskMetric):
    def __init__(
        self,
        eval_types: list[str] = ["hq", "lq"],
        model_types: Optional[list[str]] = None,
    ):
        super().__init__(eval_types=eval_types)
        self.model_types = model_types
        if self.model_types[-1][:3] == 'cub':
            self.n_class = 200
        else:
            self.n_class = 1000
        if self.model_types is not None:
            self._setup_model()
        self._setup_metrics()

    def _setup_model(self):
        self.preprocess = T.Compose(
            [
                T.Resize((224, 224)),
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        ft_ckpt_path = {
            "r50v1": "logs/classification/r50v1/version_0/checkpoints/epoch=6-val=0.6855.ckpt",
            "r50v2": "logs/classification/r50v2/version_0/checkpoints/epoch=7-val=0.7375.ckpt",
            "vit": "logs/classification/vitb16v1/version_1/checkpoints/epoch=8-val=0.7380.ckpt",
            "vgg": "logs/classification/vgg16v1_aug_de/version_4/checkpoints/epoch=83-val=0.6188.ckpt",
            "swin": "logs/classification/swinv2bv1/version_1/checkpoints/epoch=6-val=0.7960.ckpt",
        }
        self.classifiers = nn.ModuleDict()
        for model_type in self.model_types:
            print(model_type)
            # CUB
            if model_type == "cub_vitb":
                classifier = timm.create_model("hf_hub:anonauthors/cub200-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True) # 0.6354808807373047
            elif model_type == 'cub_conv':
                classifier = timm.create_model("hf_hub:anonauthors/cub200-timm-convnext_base.fb_in1k", pretrained=True) # 0.7065525054931641 
            elif model_type == 'cub_swin':
                classifier = timm.create_model("hf_hub:anonauthors/cub200-timm-swin_base_patch4_window7_224.ms_in22k_ft_in1k", pretrained=True) # 0.7432680726051331
            elif model_type == 'cub_vitL':
                classifier = pipeline("image-classification", model="HaotianZG/vit-cub-200-2011-bird",device=0)# 0.9775383472442627
            elif model_type == 'cub_r18':
                classifier = resnet18(weights='IMAGENET1K_V1')
                pretrained_weights = torch.load(r'path_to_CUB_r18.ckpt.pt')
                classifier.fc = nn.Linear(512, 200)
                classifier.load_state_dict(pretrained_weights, strict=True)
                classifier = classifier.requires_grad_(False).eval()
            elif model_type == 'cub_r50':
                classifier = resnet50(weights='IMAGENET1K_V1').requires_grad_(False).eval() 
                pretrained_weights = torch.load(r'path_to_CUB_r50.ckpt.pt')
                classifier.fc = nn.Linear(2048, 200)
                classifier.load_state_dict(pretrained_weights, strict=True)
                classifier = classifier.requires_grad_(False).eval()
            # For ImageNet1k
            elif "r18" in model_type:
                classifier = resnet18(pretrained=True)
            elif "r50v1" in model_type:
                classifier = resnet50(weights="IMAGENET1K_V1")
            elif "r50v2" in model_type:
                classifier = resnet50(weights="IMAGENET1K_V2")
            elif "r101v1" in model_type:
                classifier = resnet101(weights="IMAGENET1K_V1").requires_grad_(False).eval()
            elif "vit" in model_type:
                classifier = vit_b_16(weights="IMAGENET1K_V1")
            elif "vgg" in model_type:
                classifier = vgg16(weights="IMAGENET1K_V1")
            elif "swin" in model_type:
                classifier = swin_v2_b(weights="IMAGENET1K_V1")
            elif "rvt" in model_type:
                classifier = rvt_base_plus(pretrained=True)
            elif "eff" in model_type:
                classifier = efficientnet_v2_l(weights="IMAGENET1K_V1").requires_grad_(False).eval()
            else:
                raise ValueError(f"Unknown claddifier name: {model_type}")
            # load finetuned ckpt
            if model_type.endswith("_ft"):
                self._load_ckpt(classifier, ft_ckpt_path[model_type[:-3]])
            
            self.classifiers[model_type] = classifier

    def _setup_metrics(self):
        wrapper = self.metric_wrapper
        if self.model_types is None:
            metrics = {
                "acc": wrapper(MulticlassAccuracy(num_classes=self.n_class, top_k=1)),
            }
        else:
            metrics = {
                name: wrapper(MulticlassAccuracy(num_classes=self.n_class, top_k=1))
                for name in self.model_types
            }
        self.metrics = nn.ModuleDict(metrics)

    def _load_ckpt(self, model: nn.Module, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(ckpt, "model.")
        model.load_state_dict(ckpt)

    def update_metrics(self, preds: Tensor, target: Tensor) -> None:
        """
        Args:
            preds: logits (k*B, N)
            target: (B)
        """
        num_outputs = len(self.eval_types)
        target = repeat(target, "b -> b k", k=num_outputs)

        # post-process
        if self.model_types is None:  # bare
            preds = rearrange(preds, "(k b) n -> b n k", k=num_outputs)
            self.metrics["acc"].update(preds, target)
        else:
            to_clf = self.preprocess(preds)
            for name, clf in self.classifiers.items():
                logit = rearrange(clf(to_clf), "(k b) n -> b n k", k=num_outputs)
                self.metrics[name].update(logit, target)


class CrossEntropy(Metric):
    """Torchmetrics cross entropy loss implementation."""

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update the state with new predictions and targets.
        Args:
            preds: (B, C)
            targets: (B)
        """
        self.sum_loss += F.cross_entropy(preds, targets, reduction="sum")
        self.total += targets.numel()

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        # Return average loss over entire validation dataset
        return self.sum_loss / self.total
