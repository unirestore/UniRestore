from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from torch import Tensor

from .autoencoder import SkipConnectedAutoEncoder
from .base_model import ControlledUNet
from .controller import Controller, stablesr_config

# def hook_fn(grad):
#     print(f"Gradient computed for: {grad}")

class DiffUIE(nn.Module):
    def __init__(
        self,
        frenc: Optional[dict] = None,
        cnet: Optional[dict] = None,
        tedit: Optional[dict] = None
    ):
        super().__init__()
        model_id = "stabilityai/sd-turbo"
        # model_id = "stabilityai/stable-diffusion-2"
        # model_id = "stabilityai/sdxl-turbo"

        self.fr_type = frenc["type"] if frenc else None
        self.control_type = cnet["type"] if cnet else None
        self.tedit = tedit if tedit else None

        # modules
        ##### vae #####
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.ae = SkipConnectedAutoEncoder(vae, self.fr_type, self.tedit)

        # testing the computational cost
        ############################################
        from calflops import calculate_flops
        input_shape = (1, 3, 512, 512)
        self.ae.cuda()
        flops, macs, params = calculate_flops(model = self.ae, 
                                              input_shape=input_shape,
                                              output_as_string=True,
                                              output_precision=4)
        print("UniRestore.AEs FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
        raise TypeError

        # for name, param in self.ae.vae.decoder.named_parameters():
        #     param.register_hook(lambda grad, name=name: print(f"Gradient computed for: {name}"))

        if self.control_type: 
            ##### unet #####
            unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
            self.controller = Controller(**stablesr_config)
            self.base_model = ControlledUNet(unet, control_type=self.control_type)

            ##### noise scheduler #####
            self.register_buffer(
                "train_timesteps",
                torch.tensor([249, 499, 749, 999, 999, 999], dtype=int),
            ) 
            self.ddpm = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
            self.scheduler = DDIMScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            )
            self.scheduler.set_timesteps(
                cnet["num_inference_steps"], device=self.train_timesteps.device
            )

    def diffuse(
        self, latents: Tensor, timesteps: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        if timesteps is None:
            # random timesteps
            tidx = torch.randint(
                0, len(self.train_timesteps), (latents.size(0),), device=latents.device
            )
            timesteps = self.train_timesteps[tidx]

        noise = torch.randn_like(latents)
        noised = self.ddpm.add_noise(latents, noise, timesteps)
        return noised, noise, timesteps

    def predict_z0(
        self, latents: Tensor, conditions: Tensor, timesteps: Tensor
    ) -> Tensor:
        # forward dm
        control = self.controller(conditions, timesteps)
        pred_eps = self.base_model(latents, control, timesteps)
        # predict original sample
        bsz = latents.size(0)
        alphas_cumprod = self.ddpm.alphas_cumprod
        alpha_prod_t = alphas_cumprod[timesteps].view(bsz, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (
            latents - beta_prod_t ** (0.5) * pred_eps
        ) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def forward(self, images: Tensor, task: str):
        # preprocess size
        h, w = org_h, org_w = images.shape[-2:]

        #########################################################
        # pad to > 512 and multiple of 64
        # h = h + (64 - h % 64) % 64 if h > 512 else 512
        # w = w + (64 - w % 64) % 64 if w > 512 else 512
        # pad_h = h - org_h
        # pad_w = w - org_w
        # pad_left = pad_w // 2
        # pad_right = pad_w - pad_left
        # pad_top = pad_h // 2
        # pad_bottom = pad_h - pad_top
        # images = F.pad(images, (0, pad_w, 0, pad_h), mode="reflect")
        #########################################################

        if h < 512 or w < 512:
            scale_factor = 512 / min(h, w)
            h, w = round(h * scale_factor), round(w * scale_factor)
            images = F.interpolate(
                images, (h, w), mode="bicubic", align_corners=False, antialias=False
            )
        if h % 64 != 0 or w % 64 != 0:
            # padding to multiple of 64
            pad_w = (64 - w % 64) % 64
            pad_h = (64 - h % 64) % 64
            images = F.pad(images, pad=(0, pad_w, 0, pad_h), mode="reflect")
            
        # encode
        z0, z0_mids = self.ae.encode(images, enable_fr=(self.fr_type is not None))

        if self.control_type:
            # diffuse
            timesteps = 999 * torch.ones(
                (len(images),), dtype=int, device=images.device
            )
            zt, _, timesteps = self.diffuse(z0, timesteps)
            # denoise
            for t in self.scheduler.timesteps:
                timesteps = t.reshape(-1)
                control = self.controller(z0, timesteps)
                pred_eps = self.base_model(zt, control, timesteps)
                zt = self.scheduler.step(pred_eps, t, zt).prev_sample
        else:
            zt = z0

        # # decode
        preds = self.ae.decode(zt, z0_mids, task)

        # postprocess size
        #########################################################
        # preds = preds[..., pad_top : h - pad_bottom, pad_left : w - pad_right]
        # preds = preds[..., :org_h, :org_w]
        #######################################################

        # unpad
        preds = preds[..., :h, :w]
        # resize
        preds = F.interpolate(
            preds, (org_h, org_w), mode="bicubic", align_corners=False, antialias=False
        )
        return preds
