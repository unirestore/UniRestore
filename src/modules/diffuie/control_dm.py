import sys

sys.path.append("/mnt/201/a/ycc6/src/modules")

import torch
import torch.nn as nn
from diffuie.base_model import ControlledUNet
from diffuie.controller import Controller, stablesr_config
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from torch import Tensor

from .cfrm import AdaNAF, AdaNAFV2, NAFBlock, my_vae_encoder_fwd


class DiffUIE(nn.Module):
    def __init__(self, control_type: str, tasks: int = 2):
        super().__init__()
        model_id = "stabilityai/sd-turbo"
        self.tasks = tasks

        # modules
        ##### noise scheduler #####
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        # scheduler.set_timesteps(1, device="cuda")
        # scheduler.alphas_cumprod = scheduler.alphas_cumprod.cuda()

        ##### unet #####
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.base_model = ControlledUNet(unet, control_type=control_type)

        ##### vae #####
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

        ########## vae encoder ##########
        self.vae.encoder.forward = my_vae_encoder_fwd.__get__(
            self.vae.encoder, self.vae.encoder.__class__
        )
        self.vae.encoder.fp_blocks = nn.ModuleList(
            [
                nn.Identity(),
                nn.Sequential(*[NAFBlock(128) for _ in range(0)], AdaNAFV2(128)),
                nn.Sequential(*[NAFBlock(256) for _ in range(0)], AdaNAFV2(256)),
                nn.Sequential(*[NAFBlock(512) for _ in range(8)], AdaNAFV2(512)),
            ]
        )
        ########## vae encoder ##########

        ##### controller #####
        self.controller = Controller(**stablesr_config)

    def encode_latents(self, images: Tensor):
        """Adapted from https://github.com/huggingface/diffusers/blob/v0.25.1/examples/text_to_image/train_text_to_image.py#890

        Args:
            images (torch.Tensor): shape = (b, c, h, w). value in [0, 1].
        Returns:
            latents (torch.Tensor): shape = (b, 4, h//8, w//8)
        """
        images = images * 2 - 1  # value in [-1, 1]
        latents = self.vae.encode(images, return_dict=False)[0].sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents: Tensor):
        """
        Args:
            latents (torch.Tensor): shape = (b, c, h//8, w//8)
            encoder_res_samples (list[torch.Tensor]): list of encoder residual samples
        Returns:
            images (torch.Tensor): shape = (b, c, h, w)
        """
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents, return_dict=False)[0]
        images = (images + 1) / 2
        return images

    def forward_ldm(
        self, latents: Tensor, conditions: Tensor, num_inference_steps: int
    ):
        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        for t in self.scheduler.timesteps:
            timesteps = t.reshape(-1)
            control = self.controller(conditions, timesteps)
            pred_eps = self.base_model(latents, control, timesteps)
            latents = self.scheduler.step(pred_eps, t, latents)
            latents = latents.prev_sample
        return latents

    def forward(self, images: Tensor, task_ids=None):
        """Adapted from diffusers.StableDiffusionPipeline.__call__()"""
        device = images.device

        # vae encoder
        self.vae.encoder.enable_fp = True
        latents = self.encode_latents(images)

        # prepare dm input
        noise = torch.randn_like(latents)
        timesteps = 999 * torch.ones((len(images),), dtype=int, device=device)
        noised_latents = self.scheduler.add_noise(latents, noise, timesteps)
        # forward dm
        latents = self.forward_ldm(noised_latents, latents, 1)

        preds = self.decode_latents(latents)
        return preds


if __name__ == "__main__":
    import torch.nn.functional as F

    device = "cuda:0"

    model = DiffUIE(control_type="scedit", tasks=2).to(device)
    model.scheduler.set_timesteps(1, device=device)
    model.scheduler.alphas_cumprod = model.scheduler.alphas_cumprod.to(device)

    model.requires_grad_(False).eval()
    model.controller.requires_grad_(True).train()
    # for name, module in model.base_model.named_modules():
    #     if "spade" in name:
    #         module.requires_grad_(True).train()
    model.base_model.csc_editors.requires_grad_(True).train()

    # show n param
    # print(sum(p.numel() for p in model.parameters()))
    images = torch.rand(2, 3, 512, 512, device=device)
    bsz = images.shape[0]

    with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
        l0 = model.encode_latents(images)

        # prepare dm input
        noise = torch.randn_like(l0)
        timesteps = 999 * torch.ones((bsz,), dtype=int, device=device)
        zt = model.scheduler.add_noise(l0, noise, timesteps)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        control = model.controller(l0, timesteps)
        pred_noise = model.base_model(zt, control, timesteps)

        loss = F.mse_loss(pred_noise, torch.zeros_like(pred_noise))

    print("vram", torch.cuda.memory_allocated(device) / 1024**3)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(name)
    print("Done")
