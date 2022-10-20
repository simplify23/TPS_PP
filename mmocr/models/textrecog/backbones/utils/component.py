
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from einops import rearrange, reduce, repeat
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, Sequential
from mmocr.models.builder import BACKBONES
from mmocr.models.textrecog.decoders.transformer_mask import TransDecoderLayer, TFCommonDecoderLayer


class Unet(nn.Module):
# For mini-Unet
    def __init__(self, in_channels=512,
                 num_channels=64,
                 attn_mode='nearest',
                 stride = 2,
                 ratio = [1,1,1],
                 u_channel = 2,
                 ):
        super().__init__()
        self.stride = stride
        self.k_encoder = nn.Sequential(
            self._encoder_layer(in_channels * u_channel, num_channels * ratio[0], stride=1),
            self._encoder_layer(num_channels * ratio[0], num_channels* ratio[1], stride=2),
            self._encoder_layer(num_channels* ratio[1], num_channels* ratio[2], stride=stride),
            # self._encoder_layer(num_channels, num_channels, stride=2),
            # self._encoder_layer(num_channels, num_channels, stride=(1,2))
            )

        # self.trans = TFCommonDecoderLayer(d_model=64,
        #          d_inner=64,
        #          n_head=4,
        #          d_k=16,
        #          d_v=16,
        #          ifmask=False,)
        self.atten = CBAM(num_channels* ratio[2])

        self.k_decoder = nn.Sequential(
            # self._decoder_layer(num_channels, num_channels, scale_factor=(1,2), mode=attn_mode),
            # self._decoder_layer(num_channels, num_channels, scale_factor=2, mode=attn_mode),

            self._decoder_layer(
                num_channels * ratio[2], num_channels * ratio[1], scale_factor=stride, mode=attn_mode),
            self._decoder_layer(
                num_channels * ratio[1], num_channels * ratio[0], scale_factor=2, mode=attn_mode),
            # self._decoder_layer(
            #     num_channels, num_channels, scale_factor=2, mode=attn_mode),
            self._decoder_layer(
                num_channels * ratio[0],
                in_channels,
                scale_factor=1,
                mode=attn_mode),
        )
    def _encoder_layer(self,
                       in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=2,
                       padding=1):
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,)

    def _decoder_layer(self,
                       in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       mode='nearest',
                       scale_factor=None,
                       size=None):
        align_corners = None if mode == 'nearest' else True
        return nn.Sequential(
            nn.Upsample(
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding))

    def forward(self,k):
    # Apply mini U-Net on k
    #     k = k.transpose(1,2)
        features = []
        for i in range(len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        point = features[-1]
        B,C,H,W = point.size()

        k = self.atten(point)

        # point = rearrange(point, 'b c h w -> b (h w) c')
        # point = self.trans(point,point,point,mask=None, ifmask=False)
        # point = point.transpose(1, 2).contiguous().reshape(-1, C, H, W)

        for i in range(len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)
        return {'feature':k, 'point':point}

class Unet_down3(Unet):
    def __init__(self, in_channels=512,
                 num_channels=64,
                 attn_mode='nearest',
                 stride=2,
                 ratio=[1, 1, 1],
                 u_channel=2,
                 ):
        super().__init__()
        self.stride = stride
        self.k_encoder = nn.Sequential(
            self._encoder_layer(in_channels * u_channel, num_channels * ratio[0], stride=1),
            self._encoder_layer(num_channels * ratio[0], num_channels * ratio[1], stride=2),
            self._encoder_layer(num_channels * ratio[1], num_channels * ratio[2], stride=stride),
            # self._encoder_layer(num_channels, num_channels, stride=2),
            # self._encoder_layer(num_channels, num_channels, stride=(1,2))
        )

        # self.trans = TFCommonDecoderLayer(d_model=64,
        #          d_inner=64,
        #          n_head=4,
        #          d_k=16,
        #          d_v=16,
        #          ifmask=False,)
        self.atten = CBAM(num_channels * ratio[2])

        self.k_decoder = nn.Sequential(
            # self._decoder_layer(num_channels, num_channels, scale_factor=(1,2), mode=attn_mode),
            # self._decoder_layer(num_channels, num_channels, scale_factor=2, mode=attn_mode),

            self._decoder_layer(
                num_channels * ratio[2], num_channels * ratio[1], scale_factor=stride, mode=attn_mode),
            self._decoder_layer(
                num_channels * ratio[1], num_channels * ratio[0], scale_factor=2, mode=attn_mode),
            # self._decoder_layer(
            #     num_channels, num_channels, scale_factor=2, mode=attn_mode),
            self._decoder_layer(
                num_channels * ratio[0],
                in_channels,
                scale_factor=1,
                mode=attn_mode),
        )

class Unet_Tiny(Unet):
    def __init__(self, in_channels=512,
                 num_channels=64,
                 attn_mode='nearest',
                 stride=2):
        super().__init__()
        self.k_encoder = nn.Sequential(
            # self._encoder_layer(in_channels, num_channels, stride=1),
            # self._encoder_layer(num_channels, num_channels, stride=2),
            self._encoder_layer(num_channels, num_channels, stride=2),
            self._encoder_layer(num_channels, num_channels, stride=2))


        self.k_decoder = nn.Sequential(
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=attn_mode),
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=attn_mode),)

    def forward(self,k):
    # Apply mini U-Net on k
    #     k = k.transpose(1,2)
        features = []
        for i in range(len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        point = features[-1]
        B,C,H,W = point.size()

        point = self.atten(point)
        k = point

        for i in range(len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)
        return {'feature':k, 'point':point}

class Unet_Base(Unet):
    def __init__(self, in_channels=512,
                 num_channels=64,
                 attn_mode='nearest',
                 stride=2,
                 scale = [1,1,1,1,1],
                 # scale = [1,2,2,3,3],
                 ):
        super().__init__()

        self.k_encoder = nn.Sequential(
            self._encoder_layer(in_channels * 2, num_channels * scale[0], stride=1),
            self._encoder_layer(num_channels * scale[0], num_channels * scale[1], stride=2),
            self._encoder_layer(num_channels * scale[1], num_channels * scale[2], stride=1),
            # self._encoder_layer(num_channels, num_channels, stride=2),
            self._encoder_layer(num_channels * scale[2], num_channels * scale[3], stride=2),
            self._encoder_layer(num_channels * scale[3], num_channels * scale[4], stride=1),
        )

        self.atten = CBAM(num_channels * scale[4])

        self.k_decoder = nn.Sequential(
            self._encoder_layer(num_channels * scale[4], num_channels * scale[3], stride=1),
            self._decoder_layer(
                num_channels * scale[3], num_channels * scale[2], scale_factor=stride, mode=attn_mode),

            self._encoder_layer(num_channels * scale[2], num_channels * scale[1], stride=1),
            self._decoder_layer(
                num_channels * scale[1], num_channels * scale[0], scale_factor=2, mode=attn_mode),
            # self._decoder_layer(
            #     num_channels, num_channels, scale_factor=2, mode=attn_mode),
            self._decoder_layer(
                num_channels * scale[0],
                in_channels,
                scale_factor=1,
                mode=attn_mode))

    def forward(self,k):
    # Apply mini U-Net on k
    #     k = k.transpose(1,2)
        features = []
        for i in range(len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        point = features[-1]
        B,C,H,W = point.size()

        point = self.atten(point)
        k = point

        for i in range(len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)
        return {'feature':k, 'point':point}

class PointNet(nn.Module):
# For mini-Unet
    def __init__(self, in_channels=512,
                 num_channels=64,
                 attn_mode='nearest',
                 stride = 2):
        super().__init__()
        self.stride = stride
        self.point_conv = nn.Sequential(
            self.group_conv(in_channels, in_channels, stride=2,groups=in_channels),
            self.group_conv(in_channels, in_channels, stride=2,groups=in_channels),
            # self._encoder_layer(num_channels, num_channels, stride=2),
            self.group_conv(in_channels, in_channels * 2, stride=1),
            self.group_conv(in_channels * 2, in_channels, stride=1),
        )

        self.atten = CBAM(in_channels)

    def group_conv(self,
                       in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                        groups=1):
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )

    def forward(self,k):
    # Apply mini U-Net on k
    #     k = k.transpose(1,2)
        res = k
        for i in range(len(self.point_conv)):
            k = self.point_conv[i](k)
            # features.append(k)
        point = self.atten(k)

        return {'feature':res, 'point':point}

class Fuser(nn.Module):
# For mini-Unet
    def __init__(self, num_img_channel=512,
                 ):
        super().__init__()
        self.w_att = ConvModule(2 * num_img_channel, num_img_channel,3, 1, 1)

    def forward(self,l_feature, v_feature):
        f = torch.cat((l_feature, v_feature), dim=1)
        f_att = torch.sigmoid(self.w_att(f))
        output = f_att * v_feature + (1 - f_att) * l_feature
        return output


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if ratio > 0 :
            self.shared_MLP = nn.Sequential(
                nn.Conv2d(channel, channel // ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channel // ratio, channel, 1, bias=False)
            )
        else:
            self.shared_MLP = nn.Sequential(
                nn.Conv2d(channel, channel * -ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channel * -ratio, channel, 1, bias=False)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel,ratio=16):
        super(CBAM, self).__init__()
        self.ratio = ratio
        self.channel_attention = ChannelAttentionModule(channel,ratio)
        self.spatial_attention = SpatialAttentionModule()
        if ratio < 0:
            self.down = nn.Conv2d(channel, 1, 1, bias=False)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        if self.ratio<0:
            out = self.down(out).squeeze(1)
        return out