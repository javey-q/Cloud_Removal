# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .arch_util import LayerNorm2d, find_class_in_module
from .local_arch import  Local_Base_CR
from .sgfa import SGFA

import argparse
from utils.parser_option import parse_option
from einops import repeat

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, emb_channels=0, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        if emb_channels!=0:
            self.emb_channels = emb_channels
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    c,
                ),
            )
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, id_emb=None):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        if id_emb!=None:
            emb_out = self.emb_layers(id_emb).type(y.dtype)
            while len(emb_out.shape) < len(y.shape):
                emb_out = emb_out[..., None]
            y = y + emb_out

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma



def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def image_id_embedding(image_id, dim, max_period=10000, repeat_only=False, accelerator=None):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        if accelerator!=None:
            freqs = freqs.to(device=accelerator.device)
        else:
            freqs = freqs.to(device=image_id.device)
        args = image_id[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(image_id, 'b -> b d', d=dim)
    return embedding


class NAF_ID_Net(nn.Module):

    def __init__(self, block_type='Baseline', optical_channel=3, sar_channel=1, output_channel=3, optical_width=16, optical_middle_blk_num=1, optical_enc_blks=[],
                 optical_dec_blks=[], optical_dw_expand=1, optical_ffn_expand=2, sar_width=16, sar_middle_blk_num=1, sar_enc_blks=[],
                 sar_dec_blks=[], sar_dw_expand=1, sar_ffn_expand=2, model_channels=256):
        super().__init__()

        self.model_channels = model_channels
        id_embed_dim = model_channels * 4
        self.id_embed = nn.Sequential(
            nn.Linear(model_channels, id_embed_dim),
            nn.SiLU(),
            nn.Linear(id_embed_dim, id_embed_dim),
        )

        self.optical_intro = nn.Conv2d(in_channels=optical_channel, out_channels=optical_width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        # self.optical_ending = nn.Conv2d(in_channels=width, out_channels=output_channel, kernel_size=3, padding=1, stride=1, groups=1,
        #                       bias=True)
        self.optical_encoders = nn.ModuleList()
        self.optical_decoders = nn.ModuleList()
        self.optical_middle_blks = nn.ModuleList()
        self.optical_ups = nn.ModuleList()
        self.optical_downs = nn.ModuleList()
        self.optical_first_decoders = nn.ModuleList()

        self.sar_intro = nn.Conv2d(in_channels=sar_channel, out_channels=sar_width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        # self.ending = nn.Conv2d(in_channels=width, out_channels=output_channel, kernel_size=3, padding=1, stride=1, groups=1,
        #                       bias=True)

        self.sar_encoders = nn.ModuleList()
        self.sar_decoders = nn.ModuleList()
        self.sar_middle_blks = nn.ModuleList()
        self.sar_ups = nn.ModuleList()
        self.sar_downs = nn.ModuleList()
        self.sar_first_decoders = nn.ModuleList()

        block_name = block_type + 'Block'
        block_cls = find_class_in_module(block_name, 'arch.NAF_ID_CR_net')

        # self.sgfa_module1 = SGFA(kernel_size=1, stride=1, rate=1, softmax_scale=10.0)
        # self.sgfa_module2 = SGFA(kernel_size=3, stride=1, rate=2, softmax_scale=10.0)
        # self.sgfa_module3 = SGFA(kernel_size=3, stride=1, rate=2, softmax_scale=10.0)

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(in_channels=optical_width + sar_width, out_channels=optical_width, kernel_size=3, stride=1, padding=1, groups=1,
                              bias=True),
            nn.GELU(),
        )
        self.output_layer = nn.Conv2d(in_channels=optical_width, out_channels=output_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        optical_chan = optical_width
        # (B, 64, 256, 256) ->(B, 128, 128, 128)  ->(B, 256, 64, 64) ->(B, 512, 32, 32) ->(B, 1024, 16, 16)
        for num in optical_enc_blks:
            self.optical_encoders.append(
                nn.Sequential(
                    *[block_cls(optical_chan, optical_dw_expand, optical_ffn_expand) for _ in range(num)]
                )
            )
            self.optical_downs.append(
                nn.Conv2d(optical_chan, 2*optical_chan, 2, 2)
            )
            optical_chan = optical_chan * 2

        self.optical_middle_first_blk = block_cls(optical_chan, optical_dw_expand, optical_ffn_expand, id_embed_dim)
        self.optical_middle_blks = \
            nn.Sequential(
                *[block_cls(optical_chan, optical_dw_expand, optical_ffn_expand) for _ in range(optical_middle_blk_num-2)]
            )
        self.optical_middle_last_blk = block_cls(optical_chan, optical_dw_expand, optical_ffn_expand, id_embed_dim)

        for num in optical_dec_blks:
            self.optical_ups.append(
                nn.Sequential(
                    nn.Conv2d(optical_chan, optical_chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            optical_chan = optical_chan // 2
            self.optical_first_decoders.append(
                block_cls(optical_chan, optical_dw_expand, optical_ffn_expand, id_embed_dim)
            )
            self.optical_decoders.append(
                nn.Sequential(
                    block_cls(optical_chan, optical_dw_expand, optical_ffn_expand, id_embed_dim),
                    *[block_cls(optical_chan, optical_dw_expand, optical_ffn_expand) for _ in range(num-1)]
                )
            )

        sar_chan = sar_width
        id_embed_dim = model_channels * 4
        # sar
        for num in sar_enc_blks:
            self.sar_encoders.append(
                nn.Sequential(
                    *[block_cls(sar_chan, sar_dw_expand, sar_ffn_expand) for _ in range(num)]
                )
            )
            self.sar_downs.append(
                nn.Conv2d(sar_chan, 2*sar_chan, 2, 2)
            )
            sar_chan = sar_chan * 2

        self.sar_middle_first_blk = block_cls(sar_chan, sar_dw_expand, sar_ffn_expand, id_embed_dim)
        self.sar_middle_blks = \
            nn.Sequential(
                *[block_cls(sar_chan, sar_dw_expand, sar_ffn_expand) for _ in range(sar_middle_blk_num)],
            )
        self.sar_middle_last_blk = block_cls(sar_chan, sar_dw_expand, sar_ffn_expand, id_embed_dim)

        for num in sar_dec_blks:
            self.sar_ups.append(
                nn.Sequential(
                    nn.Conv2d(sar_chan, sar_chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            sar_chan = sar_chan // 2
            self.sar_first_decoders.append(
                block_cls(sar_chan, sar_dw_expand, sar_ffn_expand, id_embed_dim)
            )
            self.sar_decoders.append(
                nn.Sequential(
                    *[block_cls(sar_chan, sar_dw_expand, sar_ffn_expand) for _ in range(num-1)]
                )
            )
        self.padder_size = 2 ** len(self.optical_encoders)

    def forward(self, optical, sar, image_id=None, accelerator=None):
    # def forward(self, input):
    #     optical, sar = input[0], input[1]
        B, C, H, W = optical.shape
        optical = self.check_image_size(optical)
        sar = self.check_image_size(sar)

        optical_x = self.optical_intro(optical)
        sar_x = self.sar_intro(sar)

        id_emb = image_id_embedding(image_id, self.model_channels, repeat_only=False, accelerator=accelerator)
        emb = self.id_embed(id_emb)

        optical_encs = []
        sar_encs = []
        i = 0
        for optical_encoder, optical_down, sar_encoder, sar_down in \
                zip(self.optical_encoders, self.optical_downs, self.sar_encoders, self.sar_downs):
            optical_x = optical_encoder(optical_x)
            optical_encs.append(optical_x)
            optical_x = optical_down(optical_x)

            sar_x = sar_encoder(sar_x)
            sar_encs.append(sar_x)
            sar_x = sar_down(sar_x)
            i += 1
            # if i==2:
                # mask_s = nn.functional.interpolate(mask, (optical_x.shape[2], optical_x.shape[3]))
                # optical_x = self.sgfa_module2(optical_x, sar_x)

        optical_x = self.optical_middle_first_blk(optical_x, emb)
        optical_x = self.optical_middle_blks(optical_x)
        optical_x = self.optical_middle_last_blk(optical_x, emb)

        sar_x = self.sar_middle_first_blk(sar_x, emb)
        sar_x = self.sar_middle_blks(sar_x)
        sar_x = self.sar_middle_last_blk(sar_x, emb)


        # mask_s = nn.functional.interpolate(mask, (optical_x.shape[2], optical_x.shape[3]))
        # optical_x = self.sgfa_module1(optical_x, sar_x, mask_s)
        i = 0
        for optical_first_decoder, optical_decoder, optical_up, optical_enc_skip, sar_first_decoder, sar_decoder, sar_up, sar_enc_skip in \
                zip(self.optical_first_decoders, self.optical_decoders, self.optical_ups, optical_encs[::-1], self.sar_first_decoders, self.sar_decoders, self.sar_ups, sar_encs[::-1]):
            optical_x = optical_up(optical_x)
            optical_x = optical_x + optical_enc_skip
            optical_x = optical_first_decoder(optical_x, emb)
            optical_x = optical_decoder(optical_x)

            sar_x = sar_up(sar_x)
            sar_x = sar_x + sar_enc_skip
            sar_x = sar_first_decoder(sar_x, emb)
            sar_x = sar_decoder(sar_x)
            i += 1
            # if i == 2:
                # mask_s = nn.functional.interpolate(mask, (optical_x.shape[2], optical_x.shape[3]))
                # optical_x = self.sgfa_module3(optical_x, sar_x)

        fusion = self.fusion_layer(torch.cat((optical_x, sar_x), dim=1))
        output = self.output_layer(fusion)

        output = output + optical
        return output[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAF_ID_Local_CR(Local_Base_CR, NAF_ID_Net):
    def __init__(self, *args, train_size=(1, 3, 256, 256), sar_size=(1, 2, 256, 256), mask_size=(1, 1, 256, 256), fast_imp=False, **kwargs):
        Local_Base_CR.__init__(self)
        NAF_ID_Net.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, sar_size=sar_size, mask_size=mask_size, useid=True, fast_imp=fast_imp)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='../options/NAF_simple_crop_config.yml',
                        help='the path of options file.')
    args = parser.parse_args()
    opt = parse_option(args.opt)

    return opt

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    opt = parse_args()
    opt_network = opt['network_g']
    opt_network['block_type'] = 'NAF'
    opt_network.pop('name')
    model = NAF_ID_Local_CR(**opt_network).cuda()

    cloudy = torch.rand(1, 3, 256, 256).cuda()
    cloudfree = torch.rand(1, 3, 256, 256).cuda()
    s1_sar = torch.rand(1, 2, 256, 256).cuda()

    output = model(cloudy, s1_sar)
    print(output.shape)

    # from ptflops import get_model_complexity_info
    #
    #
    # def prepare_input(input_size):
    #     """
    #         input_size: including batch_size.
    #         For threeD, input_size = [(2, 3, 80, 192, 160), (2, 1, 80, 192, 160)]
    #     """
    #     x1 = torch.FloatTensor(input_size[0])
    #     x2 = torch.FloatTensor(input_size[1])
    #
    #     return dict(x=(x1, x2))
    # macs, params = get_model_complexity_info(net, (optical_shape, sar_shape),  as_strings=True, verbose=False, print_per_layer_stat=False, input_constructor=prepare_input)
    #
    # params = float(params[:-3])
    # macs = float(macs[:-4])
    #
    # print(macs, params)
