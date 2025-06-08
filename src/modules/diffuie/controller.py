import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.unets.unet_2d_blocks import (
    Attention,
    ResnetBlock2D,
    Transformer2DModel,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)

sdxl_config = {
    "in_channels": 4,
    "model_channels": 256,
    "out_channels": 256,
    "num_res_blocks": 2,
    "dropout": 0,
    "channel_mult": (1, 2, 4),
    "downsample_type": "conv",
    "num_heads": 4,
    "down_block_types": (
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
}

stablesr_config = {
    "in_channels": 4,
    "model_channels": 256,
    "out_channels": 256,
    "num_res_blocks": 2,
    "dropout": 0,
    "channel_mult": (1, 1, 2, 2),
    "downsample_type": "conv",
    "num_heads": 4,
    "down_block_types": (
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    "mid_block_type": "UNetMidBlock2D",
}

mysd2_config = {
    "in_channels": 4,
    "model_channels": 256,
    "out_channels": 256,
    "num_res_blocks": 2,
    "dropout": 0,
    "channel_mult": (1, 1, 2, 2),
    "downsample_type": "conv",
    "num_heads": 4,
    "down_block_types": (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    "mid_block_type": "UNetMidBlock2DCrossAttn",
}

class Controller(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        dropout,
        channel_mult,
        downsample_type,
        num_heads,
        down_block_types,
        mid_block_type,
    ) -> None:
        super().__init__()

        # text
        self.text_embed_dim = 1024

        # time
        time_embed_dim = model_channels * 4
        self.time_proj = Timesteps(model_channels, True, 0)
        self.time_embedding = TimestepEmbedding(
            model_channels, time_embed_dim, act_fn="silu"
        )

        # down
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([])
        input_block_chans = []

        _output_channel = model_channels
        for i, down_block_type in enumerate(down_block_types):
            input_channel = _output_channel
            _output_channel = model_channels * channel_mult[i]
            is_final_block = i == len(channel_mult) - 1
            down_block = get_down_block(
                # shared args
                down_block_type,
                num_layers=num_res_blocks,
                in_channels=input_channel,
                out_channels=_output_channel,
                temb_channels=time_embed_dim,
                dropout=dropout,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                resnet_groups=32,
                downsample_padding=1,
                # attn args
                attention_head_dim=_output_channel // num_heads,
                downsample_type=downsample_type,
                # cross attn args
                transformer_layers_per_block=1,
                cross_attention_dim=self.text_embed_dim,
                num_attention_heads=num_heads,
                dual_cross_attention=False,
                use_linear_projection=True,
                only_cross_attention=False,
                upcast_attention=False,
            )

            self.down_blocks.append(down_block)
            input_block_chans.append(_output_channel)

        # mid
        match mid_block_type:
            case "UNetMidBlock2D":
                self.middle_block = UNetMidBlock2D(
                    in_channels=_output_channel,
                    temb_channels=time_embed_dim,
                    dropout=dropout,
                    resnet_eps=1e-5,
                    resnet_act_fn="silu",
                    resnet_groups=32,
                    attention_head_dim=_output_channel // num_heads,
                )
            case "UNetMidBlock2DCrossAttn":
                self.middle_block = UNetMidBlock2DCrossAttn(
                    transformer_layers_per_block=1,
                    in_channels=_output_channel,
                    temb_channels=time_embed_dim,
                    dropout=dropout,
                    cross_attention_dim=self.text_embed_dim,
                    num_attention_heads=num_heads,
                    resnet_groups=32,
                    dual_cross_attention=False,
                    use_linear_projection=True,
                )
            case _:
                raise NotImplementedError

        # feature transform
        self.fea_tran = nn.ModuleList([])
        for i in range(len(input_block_chans)):
            self.fea_tran.append(
                ResnetBlock2D(
                    in_channels=input_block_chans[i],
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=time_embed_dim,
                    groups=32,
                    eps=1e-5,
                    non_linearity="silu",
                )
            )

        # zero conv
        # apply zero conv to last conv of ResnetBlock2D and Attention
        for module in self.modules():
            if isinstance(module, ResnetBlock2D):
                nn.init.zeros_(module.conv2.weight)
                nn.init.zeros_(module.conv2.bias)
            elif mid_block_type == "UNetMidBlock2DCrossAttn" and isinstance(
                module, Transformer2DModel
            ):
                nn.init.zeros_(module.proj_out.weight)
                nn.init.zeros_(module.proj_out.bias)
            elif mid_block_type == "UNetMidBlock2D" and isinstance(module, Attention):
                nn.init.zeros_(module.to_out[0].weight)
                nn.init.zeros_(module.to_out[0].bias)

        # # check zero_module
        # for name, param in self.named_parameters():
        #     if (param == 0).all():
        #         print(name)
        # exit()

    def forward(self, x, timesteps, encoder_hidden_states=None):
        emb = self.time_embedding(self.time_proj(timesteps))

        result_list = []
        # down
        hidden = self.conv_in(x)
        for module in self.down_blocks:
            hidden, output = (
                module(hidden, emb, encoder_hidden_states)
                if encoder_hidden_states is not None
                else module(hidden, emb)
            )
            result_list.append(output[-2])
        # mid
        if isinstance(self.middle_block, UNetMidBlock2D):
            hidden = self.middle_block(hidden, emb)
        else:
            hidden = self.middle_block(hidden, emb, encoder_hidden_states)
        result_list[-1] = hidden  # replace the last one

        # post process controller features
        results = {}
        for i in range(len(result_list)):
            results[result_list[i].size(-1)] = self.fea_tran[i](
                result_list[i].contiguous(), emb
            )

        return results


if __name__ == "__main__":
    import ipdb

    # hyper
    torch.manual_seed(0)
    device = "cuda"

    # pseudo input
    bsz = 2
    x = torch.randn(bsz, 4, 64, 64, device=device)  # lq_latent
    timesteps = torch.randint(0, 1000, (bsz,), device=device)  # timesteps
    # of shape (b, 77, 1024)
    prompt = torch.randn(bsz, 77, 1024, device=device)

    # setup model by config
    stablesr_config or mysd2_config
    model = Controller(**stablesr_config)

    # params count
    print(sum(p.numel() for p in model.parameters()))
    exit()

    # load pre-trained weights
    model.load_stablesr_controller()

    model.requires_grad_(True)
    model.to(device)

    y = model(x, timesteps, prompt)

    for k, v in y.items():
        print(k, v.shape, v.sum().item())

    loss = sum([v.sum() for v in y.values()])
    loss.backward()

    # check grad non-zero
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)
    print(loss.item())

    # time_embed: pass
    # emb = model.time_embedding(model.time_proj(timesteps))
    # print(emb.sum())

    # fea_tran: pass
    # t_emb = torch.randn(2, 1024, device=device)
    # x1 = torch.randn(2, 256, 64, 64, device=device)
    # x2 = torch.randn(2, 256, 32, 32, device=device)
    # x3 = torch.randn(2, 512, 16, 16, device=device)
    # x4 = torch.randn(2, 512, 8, 8, device=device)
    # for fea_tran, x in zip(model.fea_tran, [x1, x2, x3, x4]):
    #     x = fea_tran(x, t_emb)
    #     print(x.sum())

    # mid
    # x = torch.randn(2, 512, 8, 8, device=device)
    # t_emb = torch.randn(2, 1024, device=device)
    # x = model.middle_block(x, t_emb)
    # print(x.sum())
    # mid = torch.load('mid.pt')
    # # compute their l2 distance
    # print(torch.norm(x[0] - mid[0]))
    # print(torch.norm(x[1] - mid[1]))

    # down
    # x = torch.randn(2, 4, 64, 64, device=device)
    # t_emb = torch.randn(2, 1024, device=device)
    # x = model.conv_in(x)
    # for down_block in model.down_blocks:
    #     x = down_block(x, t_emb)[0]
    #     print(x.sum())

    exit()
