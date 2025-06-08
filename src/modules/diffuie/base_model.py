import os

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.resnet import ResnetBlock2D

from .scedit import CSCEAdapter, CSCEAdapterV2, CSCEAdapterV3
from .spade import SPADE

# ref: https://github.com/IceClear/StableSR/blob/main/ldm/modules/diffusionmodules/openaimodel.py


class ControlledUNet(nn.Module):
    def __init__(self, unet: UNet2DConditionModel, control_type: str):
        super().__init__()

        # 1. prepare unet
        self.unet = unet
        # embedding for null string prompt
        # get current file dir

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.register_buffer(
            "null_embeds",
            torch.load(os.path.join(cur_dir, "sd_null_emb.pt"), map_location="cpu"),
        )

        # 2. insert control mechanism into SD's resnet blocks
        semb_channels = 256  # control signal channel

        if control_type == "spade":
            for module in self.unet.modules():
                if isinstance(module, ResnetBlock2D):
                    out_channels = module.conv2.out_channels
                    module.spade = SPADE(out_channels, semb_channels)
            self.controlled_resnet = self.spade_resnet
        elif control_type == "scedit":
            chans = [320] * 4 + [640] * 3 + [1280] * 5
            self.csc_editors = nn.ModuleList(
                [CSCEAdapter(chan, chan, semb_channels) for chan in chans]
            )
            self.controlled_resnet = self._resnet
        else:
            raise ValueError(f"control_type '{control_type}' not supported")

    def _resnet(
        self,
        resnet: ResnetBlock2D,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        _,  # ignored s_cond
    ): # resnet, sample, emb, control
        return resnet(input_tensor, temb)

    def spade_resnet(
        self,
        resnet: ResnetBlock2D,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        control: dict[int, torch.Tensor],
    ) -> torch.FloatTensor:
        # Adapted from diffusers/models/resnet.py ResnetBlock2D
        # In Stable Diffusion 2.1:
        #     self.upsample is None and self.downsample is None
        #     self.time_embedding_norm == 'default'
        #     self.output_scale_factor = 1
        hidden_states = input_tensor

        hidden_states = resnet.norm1(hidden_states)
        hidden_states = resnet.nonlinearity(hidden_states)
        hidden_states = resnet.conv1(hidden_states)

        if resnet.time_emb_proj is not None:
            if not resnet.skip_time_act:
                temb = resnet.nonlinearity(temb)
            temb = resnet.time_emb_proj(temb)[:, :, None, None]

        hidden_states = hidden_states + temb
        hidden_states = resnet.norm2(hidden_states)
        hidden_states = resnet.nonlinearity(hidden_states)
        hidden_states = resnet.dropout(hidden_states)
        hidden_states = resnet.conv2(hidden_states)

        ###### spade control ######
        hidden_states = resnet.spade(hidden_states, control[hidden_states.shape[-1]])
        ###########################

        if resnet.conv_shortcut is not None:
            input_tensor = resnet.conv_shortcut(input_tensor)
        output_tensor = (input_tensor + hidden_states) / resnet.output_scale_factor
        return output_tensor

    def forward_unet_encoder(
        self,
        sample: torch.FloatTensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control: dict[int, torch.Tensor],
        # added_cond_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ):
        """Adapted from diffusers/models/unets/unet_2d_uncondition.py UNet2DConditionModel.forward()"""
        # 1. process timesteps
        temb = self.unet.time_proj(timesteps)  # (b, ) -> (b, 320): sinusoidal
        temb = temb.to(dtype=sample.dtype)
        emb = self.unet.time_embedding(temb)  # (b, 320) -> (b, 1280): mlp

        # 1-2. get_aug_embed (for SDXL)
        # aug_emb = None
        # if self.unet.config.addition_embed_type == "text_time":
        #     # SDXL - style
        #     text_embeds = added_cond_kwargs["text_embeds"]
        #     time_ids = added_cond_kwargs["time_ids"]
        #     time_embeds = self.unet.add_time_proj(time_ids.flatten())
        #     time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        #     add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        #     add_embeds = add_embeds.to(emb.dtype)
        #     aug_emb = self.unet.add_embedding(add_embeds)

        # emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = self.unet.conv_in(sample)  # b, 4, h, w -> b, 320, h, w

        # unet
        # 3. downsample -> several CrossAttnDownBlock2D, DownBlock2D
        down_block_res_samples = [sample]
        for downsample_block in self.unet.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                # CrossAttnDownBlock2D
                for resnet, attn in zip(
                    downsample_block.resnets, downsample_block.attentions
                ):
                    sample = self.controlled_resnet(resnet, sample, emb, control)
                    sample = attn(sample, encoder_hidden_states, return_dict=False)[0]
                    down_block_res_samples += [sample]
            else:
                # DownBlock2D
                for resnet in downsample_block.resnets:
                    sample = self.controlled_resnet(resnet, sample, emb, control)
                    down_block_res_samples += [sample]

            if downsample_block.downsamplers is not None:
                # downsampler: Downsample2D
                for downsample in downsample_block.downsamplers:
                    sample = downsample(sample)
                down_block_res_samples += [sample]

        # 4. mid: UNetMidBlock2DCrossAttn res->attn->res
        sample = self.controlled_resnet(
            self.unet.mid_block.resnets[0], sample, emb, control
        )
        for attn, resnet in zip(
            self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]
        ):
            sample = attn(sample, encoder_hidden_states, return_dict=False)[0]
            sample = self.controlled_resnet(resnet, sample, emb, control)

        return sample, emb, down_block_res_samples

    def forward_unet_decoder(
        self,
        sample: torch.FloatTensor,
        emb: torch.Tensor,
        down_block_res_samples: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        control: dict[int, torch.Tensor],
    ):  # sample, temb, down_block_res_samples, null_embeds, control
        # 5. upsample -> UpBlock2D, CrossAttnUpBlock2D
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                # CrossAttnUpBlock2D
                for resnet, attn in zip(
                    upsample_block.resnets, upsample_block.attentions
                ):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    sample = torch.cat([sample, res_hidden_states], dim=1)
                    sample = self.controlled_resnet(resnet, sample, emb, control)
                    sample = attn(sample, encoder_hidden_states, return_dict=False)[0]
            else:
                # UpBlock2D
                for resnet in upsample_block.resnets:
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    sample = torch.cat([sample, res_hidden_states], dim=1)
                    sample = self.controlled_resnet(resnet, sample, emb, control)

            if upsample_block.upsamplers is not None:
                # upsampler: Upsample2D
                for upsample in upsample_block.upsamplers:
                    sample = upsample(sample)

        # 6. post-process
        sample = self.unet.conv_norm_out(sample)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        return sample

    def forward(
        self,
        sample: torch.FloatTensor,  # zt latent
        control: dict[int, torch.FloatTensor],  # l0 condition
        timesteps: torch.Tensor,    # timesteps
        # added_cond_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ):
        # broadcasting
        bsz = sample.shape[0]
        # duplicate null_embeds and added_cond_kwargs to match bsz
        null_embeds = self.null_embeds.expand(bsz, -1, -1)
        # # sdxl
        # if added_cond_kwargs is not None:
        #     for k, v in added_cond_kwargs.items():
        #         added_cond_kwargs[k] = v.repeat(bsz, *([1] * (len(v.shape) - 1)))

        # Unet - down & mid
        sample, temb, down_block_res_samples = self.forward_unet_encoder(
            sample, timesteps, null_embeds, control
        )

        # cscedit
        if hasattr(self, "csc_editors"):
            for i, csce in enumerate(self.csc_editors):
                down_block_res_samples[i] = csce(
                    down_block_res_samples[i],
                    control[down_block_res_samples[i].shape[-1]],
                )

        # # Unet - up
        sample = self.forward_unet_decoder(
            sample, temb, down_block_res_samples, null_embeds, control
        )

        return sample


if __name__ == "__main__":
    import torchvision
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

    device = "cuda:0"
    torch.manual_seed(0)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae").to(
        device
    )

    print(f"n params: {sum(p.numel() for p in vae.parameters()):,}    ")

    def encode_latents(images):
        images = images * 2 - 1  # value in [-1, 1]
        latents = vae.encode(images, return_dict=False)[0].sample()
        latents = latents * vae.config.scaling_factor
        return latents

    def decode_latents(latents):
        latents = latents / vae.config.scaling_factor
        images = vae.decode(latents, return_dict=False)[0]
        images = (images + 1) / 2
        return images

    # setup
    # model_id = "stabilityai/stable-diffusion-2-1-base"
    # model_id = "stabilityai/sdxl-turbo"
    model_id = "stabilityai/sd-turbo"
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler.set_timesteps(1, device=device)

    # pseudo input
    bsz = 1
    ht = torch.randn(bsz, 4, 64, 64, device=device)
    control = {
        64: torch.randn(bsz, 256, 64, 64, device=device),
        32: torch.randn(bsz, 256, 32, 32, device=device),
        16: torch.randn(bsz, 256, 16, 16, device=device),
        8: torch.randn(bsz, 256, 8, 8, device=device),
    }
    prompt_embeds = torch.randn(bsz, 77, 1024, device=device)
    timestep = 499 * torch.ones((bsz,), device=device)
    # added_cond_kwargs = {
    #     "text_embeds": torch.randn(1, 1280, device=device),
    #     "time_ids": torch.tensor(
    #         [[512, 512, 0, 0, 512, 512]], device=device, dtype=torch.float32
    #     ),
    # }

    # prepare models
    base_model = ControlledUNet(unet, "scedit")
    base_model.to(device)

    base_model.requires_grad_(False).eval()
    with torch.no_grad():
        img = torchvision.io.read_image(
            "/mnt/201/a/ycc6/dataset/imagenetc/newval_sub/ILSVRC2012_val_00000077.png"
        )
        img = (
            torchvision.transforms.functional.resize(img, (512, 512))
            .unsqueeze(0)
            .to(device)
            / 255.0
        )
        ht = scheduler.add_noise(
            encode_latents(img),
            torch.randn(bsz, 4, 64, 64, device=device),
            timestep.long(),
        )
        mid, scs = base_model(ht, control, timestep)

        h0 = scheduler.step(mid, 499, ht).pred_original_sample

        img = decode_latents(h0)

        torchvision.utils.save_image(img, "test.png")
        import ipdb

        ipdb.set_trace()

    # test output
    # base_model.eval()
    # with torch.inference_mode():
    #     out_latents = base_model(ht, l0, timestep, prompt_embeds)
    # print(out_latents.shape, out_latents.sum())

    # find unused parameters
    base_model.requires_grad_(True).train()
    out_latents = base_model(ht, control, timestep)

    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    for name, param in base_model.named_parameters():
        assert param.grad is None

    loss = out_latents.sum()
    loss.backward()
    for name, param in base_model.named_parameters():
        if param.grad is None:
            print(name)
    exit()
