import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch_util import extract_patches,extract_image_patches

def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x

class SGFA(nn.Module):
    '''Region affinity learning.'''

    def __init__(self, kernel_size=3, stride=1, rate=2, softmax_scale=10., use_cuda=True):
        super(SGFA, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale
        self.use_cuda = use_cuda

    def forward(self, background, foreground, mask=None):

        # accelerated calculation
        if self.rate > 1:
            foreground = F.interpolate(foreground, scale_factor=1. / self.rate, mode='bilinear',
                                       align_corners=True,recompute_scale_factor=True)  # (n, c, h/r, h/r)
            if mask:
                mask = F.interpolate(mask, scale_factor=1. / self.rate, mode='bilinear',
                                           align_corners=True, recompute_scale_factor=True)  # (n, c, h/r, h/r)
                mask_size = list(mask.size())
        foreground_size, background_size = list(foreground.size()), list(background.size())  # (n, c, h, h) , (n, c, h/r, h/r)

        background_kernel_size = 2 * self.rate
        background_patches = extract_image_patches(background, kernel_size=background_kernel_size,
                                             stride=self.stride * self.rate)
        background_patches = background_patches.view(background_size[0], -1,
                                                     background_size[1], background_kernel_size,
                                                     background_kernel_size)  # (n, m_b, c, 2*r, 2*r) m_b = (h/r)*(h/r)
        background_patches_list = torch.split(background_patches, 1,
                                              dim=0)  # 划分tensor为块 -> list(n) :(1, m_b, c, 2*r, 2*r)

        foreground_list = torch.split(foreground, 1, dim=0)  # list(n) :(1, c, h/r, h/r)
        foreground_patches = extract_image_patches(foreground, kernel_size=self.kernel_size, stride=self.stride)
        foreground_patches = foreground_patches.view(foreground_size[0], -1,
                                                     foreground_size[1], self.kernel_size,
                                                     self.kernel_size)  # (n, m_f, c, k, k) m_k=(h/r)*(h/r)
        foreground_patches_list = torch.split(foreground_patches, 1, dim=0)  # list(n) :(1, m_f, c, k, k)

        if mask:
            mask_patches = extract_image_patches(mask, kernel_size=self.kernel_size, stride=self.stride)
            mask_patches = mask_patches.view(mask_size[0], -1,
                                                         mask_size[1], self.kernel_size,
                                                         self.kernel_size)  # (n, m_f, c, k, k) m_k=(h/r)*(h/r)
            mask_patches_list = torch.split(mask_patches, 1, dim=0)  # list(n) :(1, m_f, c, k, k)


        output_list = []
        padding = 0 if self.kernel_size == 1 else 1

        for i, (foreground_item, foreground_patches_item, background_patches_item) in enumerate(zip(
                foreground_list, foreground_patches_list, background_patches_list)):
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                device = foreground_patches_item.device
                escape_NaN = escape_NaN.to(device)
            foreground_patches_item = foreground_patches_item[0]  # (m_f, c, k, k) m_f = (h/r)*(h/r)
            foreground_patches_item_normed = foreground_patches_item / torch.max(
                torch.sqrt((foreground_patches_item * foreground_patches_item).sum([1, 2, 3], keepdim=True)),
                escape_NaN)
            # same_padding ?
            score_map = F.conv2d(foreground_item, foreground_patches_item_normed, stride=1, padding=padding)
            # (1, c, h/r, h/r) conv (m_f, c, k, k) -> (1, m_f, h/r, h/r) m_f = (h/r)*(h/r)
            score_map = score_map.view(1, foreground_size[2] // self.stride * foreground_size[3] // self.stride,
                                       foreground_size[2], foreground_size[3])  # (1, m_f, h/r, h/r) m_f = (h/r)*(h/r)
            if mask:
                mask_patches_item = mask_patches_list[i][0]
                mm = (reduce_mean(mask_patches_item, axis=[1, 2, 3], keepdim=True) >= 0.9).to(torch.float32)
                # print(mm.sum())
                mm = mm.permute(1, 0, 2, 3)
                mm = mm * (-1000)
                mm = mm.repeat(1, 1, foreground_size[2], foreground_size[3])
                attention_map = F.softmax(score_map * self.softmax_scale + mm, dim=1)
            else:
                attention_map = F.softmax(score_map * self.softmax_scale, dim=1)
            attention_map = attention_map.clamp(min=1e-8)

            background_patches_item = background_patches_item[0]  # (m_b, c, 2*r, 2*r)  m_b=(h/r)*(h/r)
            if self.rate==1:
                background_patches_item = F.pad(background_patches_item, [0, 1, 0, 1])
            output_item = F.conv_transpose2d(attention_map, background_patches_item, stride=self.rate, padding=1) / 4.
            # （1, m_f, h/r, h/r) conv_transpose (m_b, c, 2*r, 2*r) -> (1, c, h, h)
            # h_out = (h/r - 1)*r - 2*p + (b_k - 1) + 1 = h - r - 2*p + 2*r = h + r - 2*p
            output_list.append(output_item)

        output = torch.cat(output_list, dim=0)
        output = output.view(background_size)  # (n, c, h, h)
        return output

#