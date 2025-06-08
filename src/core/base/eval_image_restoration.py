import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from einops import rearrange, repeat
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.image import FrechetInceptionDistance as FID
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
import pyiqa

# from torchmetrics.image import PeakSignalNoiseRatio as PSNR
# from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from .base import BaseEvaluator
from .task import TaskMetric

class ImageRestorationEvaluator(BaseEvaluator):
    """
    Task-specific Trainer: ImageRestoration
        1. setup task tools
        2. define evaluation_step
    """

    def __init__(self, save_image: bool = False, eval_mode: str = 'FR', need_crop: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_image = save_image
        self.eval_mode = eval_mode # FR, NR, ALL
        self.need_crop = need_crop
        if eval_mode == 'NR':
            self.eval_types = ["lq"]
        else:
            self.eval_types = ["hq", "lq"]

    def configure_model(self):
        super().configure_model()
        # Define task_metric
        self.task_metric = ImageRestorationMetric(eval_types=self.eval_types, eval_mode=self.eval_mode)
        self.task_metric.requires_grad_(False).eval()

    def on_validation_epoch_start(self) -> None:
        logdir = self.logger.log_dir
        if self.save_image:
            if "hq" in self.eval_types:
                os.makedirs(os.path.join(logdir, "hq"), exist_ok=True)
            if "lq" in self.eval_types:
                os.makedirs(os.path.join(logdir, "lq"), exist_ok=True)

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        # 1. parse batch
        lq, hq, gt, fname, task = batch
        if self.need_crop:
            if "lq" in self.eval_types:
                lq = self.crop_tensor(lq)
            if "hq" in self.eval_types:
                hq = self.crop_tensor(hq)

        # 2. Inference
        inputs = []
        # ["hq", "lq"]
        if "hq" in self.eval_types:
            inputs.append(hq)
        if "lq" in self.eval_types:
            inputs.append(lq)
        preds = self.forward(inputs, 'ir') 
        preds = [pred.mul(255).round_().clamp_(0, 255).div_(255) for pred in preds]
        
        # 3. Visualize
        if batch_idx == 0:
            if "hq" in self.eval_types:
                self.visualize(hq, lq, preds)
            else:
                self.visualize(lq, lq, preds)
        
        # 4. Update metrics
        self.task_metric.update_metrics(torch.cat(preds, dim=0), hq) 
        
        # save image
        if self.save_image and len(fname) == 1:
            logdir = self.logger.log_dir
            if "hq" in self.eval_types:
                enh_hq = preds[0]
                torchvision.utils.save_image(
                    enh_hq, os.path.join(logdir, "hq", f"{fname[0]}.png")
                )
            if "lq" in self.eval_types:
                if "hq" in self.eval_types:
                    enh_lq = preds[1]
                else:
                    enh_lq = preds[0]
                torchvision.utils.save_image(
                    enh_lq, os.path.join(logdir, "lq", f"{fname[0]}.png")
                )

    def on_validation_epoch_end(self):
        metrics = super().on_validation_epoch_end()
        # val_monitor
        if self.eval_mode in ["FR", "ALL"]:
            psnr = metrics["val_lq/psnr"]
            val_monitor = psnr
        elif self.eval_mode in ["NR"]:
            niqe = metrics["val_lq/niqe"]
            val_monitor = niqe
        # sync_dist to avoid warning
        self.log("val_monitor", val_monitor, sync_dist=True)
        print(val_monitor)
    
    def crop_tensor(self, image, base=16):
        # center crop
        # upper_h, upper_w = 960, 1664
        upper_h, upper_w = 512, 512
        if image.ndim == 3:
            b, h, w = image.shape
            crop_h = h
            crop_w = w
            if h > upper_h:
                crop_h = upper_h
            if w > upper_w:
                crop_w = upper_w
            return image[:, h//2 - crop_h//2 : h//2 + crop_h//2, w//2 - crop_w//2 : w//2 + crop_w//2]
        elif image.ndim == 4:
            b, c, h, w = image.shape
            crop_h = h
            crop_w = w
            if h > upper_h:
                crop_h = upper_h
            if w > upper_w:
                crop_w = upper_w
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

class ImageRestorationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target)

class ImageRestorationMetric(TaskMetric):
    def __init__(self, eval_types: list[str] = ["lq"], eval_mode: str = "FR"):
        super().__init__(eval_types=eval_types)
        self.preprocess = T.Lambda(lambda x: torch.clamp(x, 0, 1))
        self.eval_mode = eval_mode # "FR", "NR", "ALL"
        self._setup_metrics()

    def _setup_metrics(self):
        wrapper = self.metric_wrapper
        self._update_fid_real = True
        ir_metrics = {}
        if self.eval_mode == 'FR' or self.eval_mode == 'ALL':
            ir_metrics['ir'] = MetricCollection({
                        "psnr": wrapper(SKPSNR(data_range=1.0)),
                        "ssim": wrapper(SKSSIM(data_range=1.0)),
                        "lpips": wrapper(LPIPS(net_type="alex", normalize=True)),
                    })
            ir_metrics['fid'] = wrapper(
                    FID(reset_real_features=self._update_fid_real, normalize=True))
                    
        if self.eval_mode == 'NR' or self.eval_mode == 'ALL':
            ir_metrics['pyiqa'] = MetricCollection({
                        "clipiqa": wrapper(PyNRMetric('clipiqa')),
                        "musiq": wrapper(PyNRMetric('musiq')),
                        "musiq-ava": wrapper(PyNRMetric('musiq-ava')),
                        "musiq-paq2piq": wrapper(PyNRMetric('musiq-paq2piq')),
                        "musiq-spaq": wrapper(PyNRMetric('musiq-spaq')),

                        "nima-koniq": wrapper(PyNRMetric('nima-koniq')),
                        "maniqa": wrapper(PyNRMetric('maniqa')),
                        "hyperiqa": wrapper(PyNRMetric('hyperiqa')),

                        "pi": wrapper(PyNRMetric('pi')),
                        "niqe": wrapper(PyNRMetric('niqe')),
                    })

        self.metrics = nn.ModuleDict(ir_metrics)

    def _rgb2ycbcr_y(self, x: Tensor) -> Tensor:
        y = (16.0 + 65.481 * x[:, 0] + 128.553 * x[:, 1] + 24.966 * x[:, 2]) / 255.0
        return y.unsqueeze(1)

    def update_metrics(self, preds, target) -> None:
        """
        Args:
            preds: (k*B, C, H, W)
            target: (B, C, H, W)
        """
        num_outputs = len(self.eval_types)
        device = preds.device
        with torch.autocast(device_type="cuda", enabled=False):
            preds = torch.clamp(preds, 0, 1).to(torch.float32)
            preds = rearrange(
                self.preprocess(preds), "(k b) c h w -> b c h w k", k=num_outputs
            )

            if self.eval_mode == 'FR' or self.eval_mode == 'ALL':
                target = torch.clamp(target, 0, 1).to(torch.float32)
                target = repeat(
                    self.preprocess(target), "b c h w -> b c h w k", k=num_outputs
                )

                self.metrics["ir"].update(preds, target)
                self.metrics["fid"].update(
                    preds, real=torch.zeros(num_outputs, dtype=bool, device=device)
                )
                if self._update_fid_real:
                    self.metrics["fid"].update(
                        target, real=torch.ones(num_outputs, dtype=bool, device=device)
                    )

            if self.eval_mode == 'NR' or self.eval_mode == 'ALL':
                self.metrics["pyiqa"].update(preds)

    def reset_metrics(self, reset_fid_real: bool = True):
        """
        Args:
            reset_fid_real: Reset real features for FID computation.
            Set True for sanity_checking round.
        """
        self._update_fid_real = reset_fid_real
        if self.eval_mode == 'FR' or self.eval_mode == 'ALL':
            for metric in self.metrics["fid"].metrics:
                metric.reset_real_features = reset_fid_real
        super().reset_metrics()

class SKPSNR(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range
        self.add_state("sum_psnr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update the state with new predictions and targets.
        Args:
            preds: (B, C, H, W)
            targets: (B, C, H, W)
        """
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        for pred, target in zip(preds, targets):
            self.sum_psnr += peak_signal_noise_ratio(
                target, pred, data_range=self.data_range
            )
        self.total += len(targets)

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        # Return average loss over entire validation dataset
        return self.sum_psnr / self.total

class SKSSIM(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range
        self.add_state("sum_ssim", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update the state with new predictions and targets.
        Args:
            preds: (B, C, H, W)
            targets: (B, C, H, W)
        """
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        for pred, target in zip(preds, targets):
            self.sum_ssim += structural_similarity(
                pred, target, data_range=self.data_range, channel_axis=0
            )
        self.total += len(targets)

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        # Return average loss over entire validation dataset
        return self.sum_ssim / self.total

class PyNRMetric(Metric):
    # is_differentiable: bool = False
    # higher_is_better: bool = True
    # full_state_update: bool = False

    def __init__(self, metric_name: str):
        super().__init__()
        self.metric_name = metric_name
        self.metric = pyiqa.create_metric(metric_name).cuda()
        self.add_state("iqa", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor) -> None:
        """Update the state with new predictions and targets.
        Args:
            preds: (B, C, H, W)
            targets: (B, C, H, W)
        """
        with torch.cuda.amp.autocast():
            for pred in preds:
                self.iqa += self.metric(pred.unsqueeze(0)).item()
        self.total += len(preds)

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""
        # Return average loss over entire validation dataset
        return self.iqa / self.total
