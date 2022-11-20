# copyright (c) 2021 torchtorch Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from collections import Callable
# from torch import ParamAttr
# from torch.nn.initializer import KaimingNormal
# from torch.regularizer import L2decay
import numpy as np
import torch
import torch.nn as nn
# from torch.nn.initializer import TruncatedNormal, Constant, Normal
from mmocr.models.builder import BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# trunc_normal_ = TruncatedNormal(std=.02)
# normal_ = Normal
# zeros_ = Constant(value=0.)
# ones_ = Constant(value=1.)


def to_2tuple(x):
    return tuple([x] * 2)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = torch.tensor(1 - drop_prob,device=x.device)
    shape = (x.size()[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype)
    random_tensor = torch.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.reshape([B, C, H // H_sp, H_sp, W // W_sp, W_sp])
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape([-1, H_sp * W_sp, C])
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.reshape([B, H // H_sp, W // W_sp, H_sp, W_sp, -1])
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().reshape([B, H, W, -1])
    return img


class ConvMixer(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 HW=[8, 25],
                 local_k=[3, 3],
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(dim, dim, local_k, 1, [local_k[0] // 2, local_k[1] // 2], groups=num_heads,
                                     )

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.permute(0, 2, 1).contiguous().reshape([-1, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.reshape([-1, self.dim, h * w]).permute(0, 2, 1).contiguous()
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mixer='Global',
                 HW=[8, 25],
                 local_k=[7, 11],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if mixer == 'Local':
            H = HW[0]
            W = HW[1]
            hk = local_k[0]
            wk = local_k[1]
            mask = np.ones([H * W, H * W])
            for h in range(H):
                for w in range(W):
                    for kh in range(-(hk // 2), (hk // 2) + 1):
                        for kw in range(-(wk // 2), (wk // 2) + 1):
                            if H > (h + kh) >= 0 and W > (w + kw) >= 0:
                                mask[h * W + w][(h + kh) * W + (w + kw)] = 0
            mask_torch = torch.tensor(mask, dtype=torch.float32)
            mask_inf = torch.full([H * W, H * W], float('-inf'))
            self.mask = torch.where(mask_torch < 1, mask_torch, mask_inf)
        self.mixer = mixer

    def forward(self, x):
        # B= torch.shape(x)[0]
        N, C = x.shape[1:]
        self.N = N
        self.C = C
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q.matmul(k.permute(0, 1, 3, 2)))
        if self.mixer == 'Local':
            attn += self.mask.cuda()
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).permute(0, 2, 1, 3).contiguous().reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self):
        flops = self.N * self.N * self.C * 2 + 4 * self.N * self.C * self.C
        return flops

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mixer='Global',
                 local_mixer=[7, 11],
                 HW=[8, 25],
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-6):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim)
        elif isinstance(norm_layer, Callable):
            self.norm1 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or torch.nn.layer.Layer class")
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        if mixer == 'Conv':
            self.mixer = ConvMixer(dim, num_heads=num_heads, HW=HW, local_k=local_mixer)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim)
        elif isinstance(norm_layer, Callable):
            self.norm2 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or torch.nn.layer.Layer class")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        self.N, self.C = x.shape[1:]
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def flops(self):
        flops = self.attn.flops() + self.N * self.C * 2 + 2 * self.N * self.C * self.C * self.mlp_ratio
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=[32, 100], in_channels=3, embed_dim=768, sub_num=2):
        super().__init__()
        # img_size = to_2tuple(img_size)
        num_patches = (img_size[1] // (2 ** sub_num)) * \
                      (img_size[0] // (2 ** sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if sub_num == 2:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 2, 3, 2, 1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1, ),
                nn.BatchNorm2d(embed_dim),
                nn.GELU())
        if sub_num == 3:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 4, 3, 2, 1,),
                nn.BatchNorm2d(embed_dim // 4),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 2, 1, ),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1,),
                nn.BatchNorm2d(embed_dim)
            )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).permute(0, 2, 1).contiguous()
        return x

    def flops(self):
        Ho, Wo = self.img_size
        flops = Ho // 2 * Wo // 2 * 3 * self.embed_dim // 2 * (3 * 3) \
                + Ho // 4 * Wo // 4 * self.embed_dim // 2 * self.embed_dim * (3 * 3) \
                + Ho * Wo * self.embed_dim * 2
        return flops


class SubSample(nn.Module):
    def __init__(self, in_channels, out_channels, types='Pool', stride=[2, 1], sub_norm='nn.LayerNorm', act=None):
        super().__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2d(kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2d(kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                  )
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):

        if self.types == 'Pool':
            # x = x.transpose((0, 2, 1))
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).permute(0, 2, 1).contiguous())
        else:
            # self.H, self.W = x.shape[2:]
            x = self.conv(x)
            out = x.flatten(2).permute(0, 2, 1).contiguous()
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 num_groups=1,
                 sub_norm='nn.LayerNorm',
                 act=None):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=num_groups)

        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None
        # self.hardswish = nn.Hardswish()

    def forward(self, x):
        self.H, self.W = x.shape[2:]
        x = self.conv(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        # x = self.hardswish(x)
        return x

    def flops(self):
        flops = self.H // self.stride[0] * self.W // self.stride[1] * 3 * 3 * self.in_channels * self.out_channels
        flops += self.H // self.stride[0] * self.W // self.stride[1] * self.out_channels
        return flops


@BACKBONES.register_module()
class PVTR(BaseModule):
    def __init__(self,
                 img_size=[32, 128],
                 in_channels=3,
                 embed_dim=[64, 128, 256],
                 depth=[3, 6, 3],
                 num_heads=[2, 4, 8],
                 mixer=['Local', 'Local', 'Local', 'Local', 'Local', 'Local', 'Global', 'Global', 'Global', 'Global',
                        'Global', 'Global'],  # Local atten, Global atten, Conv
                 local_mixer=[[7, 11], [7, 11], [7, 11]],
                 patch_merging='Conv',  # Conv, Pool, None
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 last_drop=0.1,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer='nn.LayerNorm',
                 sub_norm='nn.LayerNorm',
                 epsilon=1e-6,
                 out_channels=192,
                 out_char_num=32,
                 block_unit='Block',
                 act='nn.GELU',
                 last_stage=True,
                 sub_num=2,
                 **kwargs):
        super().__init__()
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num)
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // (2 ** sub_num), img_size[1] // (2 ** sub_num)]
        x = torch.zeros(1, num_patches, embed_dim[0])
        self.pos_embed = nn.parameter.Parameter(
            x,requires_grad=True)
        # self.add_parameter("pos_embed", self.pos_embed)
        # self.cls_token = self.create_parameter(
        #     shape=(1, 1, embed_dim), default_initializer=zeros_)
        # self.add_parameter("cls_token", self.cls_token)
        self.pos_drop = nn.Dropout(p=drop_rate)
        Block_unit = eval(block_unit)
        if block_unit == 'CSWinBlock':
            split_size_h = [1, 2, 2]
            split_size_w = [5, 5, 25]
            ex_arg = [{'reso': [img_size[0] // 4, img_size[1] // 4],
                       'split_size_h': split_size_h[0],
                       'split_size_w': split_size_w[0]},
                      {'reso': [img_size[0] // 8, img_size[1] // 4],
                       'split_size_h': split_size_h[1],
                       'split_size_w': split_size_w[1]},
                      {'reso': [img_size[0] // 16, img_size[1] // 4],
                       'split_size_h': split_size_h[2],
                       'split_size_w': split_size_w[2]}
                      ]
        else:
            ex_arg = [{'epsilon': epsilon},
                      {'epsilon': epsilon},
                      {'epsilon': epsilon}]
        dpr = np.linspace(0, drop_path_rate, sum(depth))
        self.blocks1 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mixer=mixer[0:depth[0]][i],
                HW=self.HW,
                local_mixer=local_mixer[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[0:depth[0]][i],
                norm_layer=norm_layer,
                **ex_arg[0],
            ) for i in range(depth[0])
        ])
        if patch_merging is not None:
            self.sub_sample1 = SubSample(embed_dim[0], embed_dim[1], sub_norm=sub_norm, stride=[2, 1],
                                         types=patch_merging)  # ConvBNLayer(embed_dim[0], embed_dim[1], kernel_size=3, stride=[2, 1], sub_norm=sub_norm)
            HW = [self.HW[0] // 2, self.HW[1]]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mixer=mixer[depth[0]:depth[0] + depth[1]][i],
                HW=HW,
                local_mixer=local_mixer[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0]:depth[0] + depth[1]][i],
                norm_layer=norm_layer,
                **ex_arg[1]) for i in range(depth[1])
        ])
        if patch_merging is not None:
            self.sub_sample2 = SubSample(embed_dim[1], embed_dim[2], sub_norm=sub_norm, stride=[2, 1],
                                         types=patch_merging)  # ConvBNLayer(embed_dim[1], embed_dim[2], kernel_size=3, stride=[2, 1], sub_norm=sub_norm)
            HW = [self.HW[0] // 4, self.HW[1]]
        else:
            HW = self.HW
        self.blocks3 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mixer=mixer[depth[0] + depth[1]:][i],
                HW=HW,
                local_mixer=local_mixer[2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + depth[1]:][i],
                norm_layer=norm_layer,
                **ex_arg[2]) for i in range(depth[2])
        ])
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d([1, out_char_num])
            self.linear = nn.Linear(384,512)
            self.last_conv = nn.Conv2d(
                in_channels=embed_dim[2],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0)

            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop)
        self.norm = eval(norm_layer)(embed_dim[-1])

        # Classifier head
        # self.head = nn.Linear(embed_dim,
        #                       class_num) if class_num > 0 else Identity()

        trunc_normal_(self.pos_embed)
        # trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # B = x.shape[0]

        B = x.size()[0]
        x = self.patch_embed(x)
        # cls_tokens = self.cls_token.expand((B, -1, -1))
        # x = torch.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(x.permute(0, 2, 1).contiguous().reshape([B, self.embed_dim[0], self.HW[0], self.HW[1]]))
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(x.permute(0, 2, 1).contiguous().reshape([B, self.embed_dim[1], self.HW[0] // 2, self.HW[1]]))
        for blk in self.blocks3:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.last_stage:
            x = self.linear(x)
        #     B = x.size()[0]
        #     if self.patch_merging is not None:
        #         h = self.HW[0] // 4
        #     else:
        #         h = self.HW[0]
        #     x = self.avg_pool(x.permute(0, 2, 1).contiguous().reshape([B, self.embed_dim[2], h, self.HW[1]]))
        #     x = self.last_conv(x)
        #     x = self.hardswish(x)
        #     x = self.dropout(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = self.patch_embed.flops()
        for blk in self.blocks1:
            flops += blk.flops()
        for blk in self.blocks2:
            flops += blk.flops()
        for blk in self.blocks3:
            flops += blk.flops()
        return flops + self.img_size[0] // 16 * self.img_size[1] // 4 * self.embed_dim[-1] + \
               self.img_size[1] // 4 * self.embed_dim[-1] * self.out_channels

# class CSWinBlock(nn.Module):
#
#     def __init__(self, dim, reso, num_heads,
#                  split_size_h=1, split_size_w=5, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer='nn.LayerNorm',
#                  last_stage=False):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.patches_resolution = reso
#
#         self.mlp_ratio = mlp_ratio
#         self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
#         self.norm1 = eval(norm_layer)(dim)
#
#         if self.patches_resolution[0] == split_size_h:
#             last_stage = True
#         if last_stage:
#             self.branch_num = 1
#         else:
#             self.branch_num = 2
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(drop)
#
#         if last_stage:
#             self.attns = nn.ModuleList([
#                 LePEAttention(
#                     dim, resolution=self.patches_resolution, idx=-1,
#                     split_size_h=split_size_h, split_size_w=split_size_w, num_heads=num_heads,
#                     dim_out=dim,
#                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#                 for i in range(self.branch_num)])
#         else:
#             self.attns = nn.ModuleList([
#                 LePEAttention(
#                     dim // 2, resolution=self.patches_resolution, idx=i,
#                     split_size_h=split_size_h, split_size_w=split_size_w,
#                     num_heads=num_heads // 2, dim_out=dim // 2,
#                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#                 for i in range(self.branch_num)])
#
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp_ratio = mlp_ratio
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
#                        drop=drop)
#         self.norm2 = eval(norm_layer)(dim)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#
#         H, W = self.patches_resolution
#
#         B, L, C = x.shape
#         self.L = L
#         self.C = C
#         # print(L, H, W, C)
#         assert L == H * W, "flatten img_tokens has wrong size"
#         img = self.norm1(x)
#         qkv = self.qkv(img).reshape([B, -1, 3, C]).transpose([2, 0, 1, 3])  # 3, B , L C
#
#         if self.branch_num == 2:
#             x1 = self.attns[0](qkv[:, :, :, :C // 2])
#             x2 = self.attns[1](qkv[:, :, :, C // 2:])
#             attened_x = torch.concat([x1, x2], axis=2)
#         else:
#             attened_x = self.attns[0](qkv)
#         attened_x = self.proj(attened_x)
#         x = x + self.drop_path(attened_x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#
#         return x
#
#     def flops(self):
#         flops = self.L * self.C * 2 + self.L * self.C * self.C * 3
#         if self.branch_num == 2:
#             flops += self.attns[0].flops()
#             flops += self.attns[1].flops()
#         else:
#             flops += self.attns[0].flops()
#         flops += self.L * self.dim * self.dim
#         flops += 2 * self.L * self.C * self.C * self.mlp_ratio
#
#         return flops

# class LePEAttention(nn.Module):
#     def __init__(self, dim, resolution, idx, split_size_h=1, split_size_w=5, dim_out=None, num_heads=8,
#                  attn_drop=0., proj_drop=0.,
#                  qk_scale=None):
#         super().__init__()
#         self.dim = dim
#         self.dim_out = dim_out or dim
#         self.resolution = resolution
#
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#         if idx == -1:
#             H_sp, W_sp = self.resolution[0], self.resolution[1]
#         elif idx == 0:
#             H_sp, W_sp = self.resolution[0], split_size_w
#         elif idx == 1:
#             W_sp, H_sp = self.resolution[1], split_size_h
#         else:
#             print("ERROR MODE", idx)
#             exit(0)
#         self.H_sp = H_sp
#         self.W_sp = W_sp
#         # print(idx, H_sp, W_sp)
#         self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#
#     def im2cswin(self, x):
#         B, N, C = x.shape
#         H, W = self.resolution
#         x = x.transpose([0, 2, 1]).reshape([B, C, H, W])
#         x = img2windows(x, self.H_sp, self.W_sp)
#         x = x.reshape([-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])
#         return x
#
#     def get_lepe(self, x, func):
#         B, N, C = x.shape
#         H, W = self.resolution
#         x = x.transpose([0, 2, 1]).reshape([B, C, H, W])
#
#         H_sp, W_sp = self.H_sp, self.W_sp
#         # print(H, W, H_sp, W_sp)
#         x = x.reshape([B, C, H // H_sp, H_sp, W // W_sp, W_sp])
#         x = x.transpose([0, 2, 4, 1, 3, 5]).reshape([-1, C, H_sp, W_sp])  ### B', C, H', W'
#
#         lepe = func(x)  ### B', C, H', W'
#         lepe = lepe.reshape([-1, self.num_heads, C // self.num_heads, H_sp * W_sp]).transpose([0, 1, 3, 2])
#
#         x = x.reshape([-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp]).transpose([0, 1, 3, 2])
#         return x, lepe
#
#     def forward(self, qkv):
#         """
#         x: B L C
#         """
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         ### Img2Window
#         H, W = self.resolution
#         B, L, C = q.shape
#         self.L = self.H_sp * self.W_sp
#         self.C = C
#         assert L == H * W, "flatten img_tokens has wrong size"
#
#         q = self.im2cswin(q)
#         k = self.im2cswin(k)
#         v, lepe = self.get_lepe(v, self.get_v)
#
#         q = q * self.scale
#         attn = (q @ k.transpose([0, 1, 3, 2]))  # B head N C @ B head C N --> B head N N
#         attn = nn.functional.softmax(attn, axis=-1, dtype=attn.dtype)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v) + lepe
#         x = x.transpose([0, 2, 1, 3]).reshape([-1, self.H_sp * self.W_sp, C])  # B head N N @ B head N C
#
#         ### Window2Img
#         x = windows2img(x, self.H_sp, self.W_sp, H, W).reshape([B, -1, C])  # B H' W' C
#
#         return x
#
#     def flops(self):
#
#         flops = 2 * self.L * self.L * self.C + self.L * 3 * 3 * self.dim
#
#         return flops
