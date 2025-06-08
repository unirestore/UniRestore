from typing import Optional

import torch
from lightning.pytorch import LightningModule
from timm.optim.optim_factory import create_optimizer_v2
from torch import optim

from .task import TaskMetric


class BaseTrainer(LightningModule):
    """
    BaseTrainer: A task-agnoistic trainer inheriting from LightningModule.
        1. setup generic trainer utilities
        2. define optimizer & lr_scheduler
    Enhancer Model Trainer
        1. setup enhancer model
        2. define training_step
        3. define forward for inference
    """

    def __init__(
        self,
        model_kwargs: Optional[dict] = None,
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model_kwargs = model_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        # skip saving tasks' weights @ self.on_save_checkpoint
        self.no_checkpoint: list[str] = ["task_metric", "criterion"]
        self.no_ckpt_exception: list[str] = []
        # since some weights are omitted, must disable strict_loading for resuming
        self.strict_loading = False
        # self.automatic_optimization = False

    def configure_model(self):
        """
        # 1. Define model
        # 2. Load ckpt
        # 3. Configure trainable params
        """
        super().configure_model()

    def configure_optimizers(self):
        # 1. Parse optimizer & scheduler kwargs
        opt_kwargs, sched_kwargs, total_steps = self._parse_optim_kwargs()

        # 2. Setup optimizer
        optimizer = create_optimizer_v2(self.model, **opt_kwargs)
        if sched_kwargs is None:
            return optimizer

        # 3. Setup scheduler
        sched = sched_kwargs.pop("sched")
        if sched == "onecycle":
            lr_scheduler = {
                "scheduler": optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=[pg["lr"] for pg in optimizer.param_groups],
                    total_steps=total_steps,
                    pct_start=0.1,
                    anneal_strategy="cos",
                    div_factor=10,
                ),
                "interval": "step",
            }
        elif sched == "step":
            lr_scheduler = {
                "scheduler": optim.lr_scheduler.StepLR(optimizer, **sched_kwargs),
                "interval": "epoch",
            }
        else:
            raise ValueError(f"Unknown scheduler: {sched}")

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def _parse_optim_kwargs(self):
        """
        Returns:
            opt_kwargs: dict, optimizer kwargs
            sched_kwargs: dict, lr_scheduler kwargs
            total_steps: int, total training steps
        """
        # 1. get optim params
        opt_kwargs = self.optimizer_kwargs
        opt = opt_kwargs["opt"]
        base_bsz = opt_kwargs["base_bsz"]
        base_lr = opt_kwargs["base_lr"]
        weight_decay = opt_kwargs.get("weight_decay", -1)
        momentum = opt_kwargs.get("momentum", -1)

        sched_kwargs = self.lr_scheduler_kwargs
        sched = sched_kwargs["sched"] if sched_kwargs else ""

        # 2. get effective optim params
        self.trainer.fit_loop.setup_data()
        total_steps = self.trainer.estimated_stepping_batches
        batch_size = self.trainer.train_dataloader.batch_size
        accum_iter = self.trainer.accumulate_grad_batches
        num_devices = self.trainer.num_devices

        eff_batch_size = batch_size * accum_iter * num_devices
        eff_lr = base_lr * (eff_batch_size / base_bsz) ** 0.5

        opt_kwargs = {"opt": opt, "lr": eff_lr}
        if weight_decay >= 0:
            opt_kwargs["weight_decay"] = weight_decay
        if momentum >= 0:
            opt_kwargs["momentum"] = momentum

        # 3. log
        if self.trainer.is_global_zero:
            print("#" * 20)
            print(f"[optimizer]: {opt}, [scheduler]: {sched}")
            print(f"[bsz] base->actual: {base_bsz}->{eff_batch_size}")
            print(f"[lr]  base->actual: {base_lr:.2e}->{eff_lr:.2e}")
            print(f"[weight_decay]: {weight_decay:.2e}")
            print(f"[momentum]: {momentum:.2e}")
            print("#" * 20)
        return opt_kwargs, sched_kwargs, total_steps

    def on_save_checkpoint(self, checkpoint) -> None:
        # remove frozen weights in self.no_checkpoint
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not any(k.startswith(f"{kw}.") for kw in self.no_checkpoint)
            or any(k.startswith(f"{kw}.") for kw in self.no_ckpt_exception)
        }

class BaseEvaluator(LightningModule):
    task_metric: TaskMetric

    def configure_model(self):
        super().configure_model()
        # self.task_metric: TaskMetric = ...

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            inputs: [hq, lq, ...]
        Returns:
            preds: [enh_hq, enh_lq, ...]
        """
        return inputs

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_validation_epoch_end(self):
        # compute & reset
        metrics = self.task_metric.compute_metrics(prefix="val")
        self.task_metric.reset_metrics(reset_fid_real=self.trainer.sanity_checking)
        # log
        self.log_dict(metrics, sync_dist=True)
        if self.trainer.is_global_zero and self.trainer.state.fn == "validate":
            TaskMetric.print_metrics(metrics)
        # empty cache
        torch.cuda.empty_cache()
        return metrics

    def visualize(self):
        raise NotImplementedError
