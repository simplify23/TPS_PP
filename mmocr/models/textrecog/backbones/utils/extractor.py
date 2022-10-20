
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from einops import rearrange, reduce, repeat
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, Sequential, ModuleList
from mmocr.models.builder import BACKBONES
from mmocr.models.textrecog.backbones.utils.component import Unet, Unet_down3, Unet_Tiny, PointNet, Unet_Base
from mmocr.models.textrecog.decoders.transformer_mask import TransDecoderLayer, TFCommonDecoderLayer


class AttentionNetwork(nn.Module):
    """Localization Network of RARE, which predicts C' (K x 2) from input
    (img_width x img_height)

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        num_img_channel (int): Number of channels of the input image.
    """

    def __init__(self, num_img_channel,point_size,p_stride):
        super().__init__()
        self.tf_layers = 1
        # give stride & padding
        # if p_stride == 1:
        #     p_padding = 1
        # else:
        #     p_padding = 1
        self.conv_feature = ConvModule(num_img_channel,64,3,1,1)
        self.conv_point = ConvModule(64,64,1,stride=p_stride)

        self.num_img_channel = num_img_channel

        self.Atten = ModuleList([
            TransDecoderLayer(d_model=64,
                              d_inner=num_img_channel if num_img_channel < 64 else 64,
                              n_head=4,
                              d_k=16,
                              d_v=16,
                              ifmask=False, )
            for _ in range(self.tf_layers)
        ])

        self.point_x = point_size[1]
        self.point_y = point_size[0]
        self.num_fiducial = self.point_y * self.point_x
        self.localization_fc1 = nn.Linear(64*self.num_fiducial, self.num_fiducial*16)
        self.localization_fc2 = nn.Linear(16*self.num_fiducial, self.num_fiducial*2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        # ctrl_pts_x = np.linspace(0.0, self.point_x , num=int(self.point_x)) / self.point_x
        # ctrl_pts_x = np.linspace(0.0, self.point_y , num=int(self.point_y)) / self.point_y #X * Y * 2
        ctrl_pts_x = np.linspace(-1.0, 1.0, num=int(self.point_x))
        ctrl_pts_y = np.linspace(-1.0, 1.0, num=int(self.point_y))  # X * Y * 2
        initial_bias = np.stack(np.meshgrid(ctrl_pts_x, ctrl_pts_y), axis=2)
        # initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(
            initial_bias).float().view(-1)

    def forward(self, batch_img,p_feature):
        """
        Args:
            batch_img (Tensor): Batch input image of shape
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Predicted coordinates of fiducial points for input batch.
            The shape is :math:`(N, F, 2)` where :math:`F` is ``num_fiducial``.
        """
        # B,T,C -> B,C,H,W

        # p_feature = p_feature.view(B,C,self.point_y,self.point_x)
        # stride(1,2)
        p_feature = self.conv_point(p_feature)
        B, C, H, W = p_feature.size()
        # shrink channel
        i_feature = self.conv_feature(batch_img)

        p_feature = rearrange(p_feature, 'b c h w -> b (h w) c')
        i_feature = rearrange(i_feature, 'b c h w -> b (h w) c')
        point_coord = self.Atten(p_feature,i_feature,i_feature)

        p_feature = point_coord.transpose(1, 2).contiguous().reshape(-1, C, H, W)
        # p_feature= point_coord.view(B,C,H,W)
        # batch_img = logits['feature']
        # logits = self.conv(batch_img).view(batch_size, -1)
        # x*y*2
        batch_C_prime = self.localization_fc2(self.localization_fc1(
            point_coord.view(B,-1))).view(B,self.num_fiducial, 2)
        return {"point": batch_C_prime,"p_feature":p_feature}

class PointNetwork(nn.Module):
    """Localization Network of RARE, which predicts C' (K x 2) from input
    (img_width x img_height)

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        num_img_channel (int): Number of channels of the input image.
    """

    def __init__(self, num_img_channel,point_size,p_stride):
        super().__init__()

        self.num_img_channel = num_img_channel
        self.point_x = point_size[1]
        self.point_y = point_size[0]

        self.conv = PointNet(
                            in_channels=num_img_channel,
                            # num_channels = 64,
                            # stride = p_stride
        )
        # self.point_y = point_size[0]
        self.point_x = point_size[1]
        self.num_fiducial = self.point_y * self.point_x

        self.atten = TransDecoderLayer(d_model=64,
                          d_inner=128,
                          n_head=4,
                          d_k=16,
                          d_v=16,
                          ifmask=False, )

        self.localization_fc1 = nn.Sequential(
            # nn.Linear(num_img_channel if num_img_channel < 64 else 64, 512),
            nn.Linear(128, 2),
            # nn.Linear(256, 2),
            nn.ReLU(True),
        )

        # self.localization_fc2 = nn.Linear(512*self.num_fiducial, self.num_fiducial*2)
        self.localization_fc2 = nn.Linear(2 * self.num_fiducial, self.num_fiducial * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(0.1, self.point_x-0.1 , num=int(self.point_x)) / self.point_x
        ctrl_pts_y = np.linspace(0.1, self.point_y-0.1 , num=int(self.point_y)) / self.point_y #X * Y * 2
        initial_bias = np.stack(np.meshgrid(ctrl_pts_x, ctrl_pts_y), axis=2)
        self.localization_fc2.bias.data = torch.from_numpy(
            initial_bias).float().view(-1)

        # count param
        self.count_param(self.conv,'Unet')
        self.count_param(self.localization_fc1, 'localization_fc1')
        self.count_param(self.localization_fc2, 'localization_fc2')

    def count_param(self, model,name):
        print("{} have {}M paramerters in total".format(name,sum(x.numel() for x in model.parameters())/1e6))


    def forward(self, batch_img):
        """
        Args:
            batch_img (Tensor): Batch input image of shape
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Predicted coordinates of fiducial points for input batch.
            The shape is :math:`(N, F, 2)` where :math:`F` is ``num_fiducial``.
        """
        batch_size = batch_img.size(0)
        logits = self.conv(batch_img)
        point = logits['point']
        # B,C,H,W = point.size()
        point = rearrange(point, 'b c h w -> b (h w) c')
        batch_C_prime = self.localization_fc2(
            self.localization_fc1(point).view(batch_size,-1)).view(batch_size,
                                                  self.num_fiducial, 2)
        return {"point": batch_C_prime,"feature": logits['feature'],"point_feat":point}

class Unet_wrap(nn.Module):
    """Localization Network of RARE, which predicts C' (K x 2) from input
    (img_width x img_height)

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        num_img_channel (int): Number of channels of the input image.
    """

    def __init__(self, num_img_channel,in_channel,p_stride):
        super().__init__()

        self.num_img_channel = num_img_channel
        # self.point_x = point_size[1]
        # self.point_y = point_size[0]

        self.conv = Unet(
                            in_channels=num_img_channel,
                            num_channels= in_channel,
                            stride = p_stride
        )
        # # self.point_y = point_size[0]
        # self.point_x = point_size[1]
        # self.num_fiducial = self.point_y * self.point_x


        # count param
        self.count_param(self.conv,'Unet')
        # self.count_param(self.localization_fc1, 'localization_fc1')
        # self.count_param(self.localization_fc2, 'localization_fc2')

    def count_param(self, model,name):
        print("{} have {}M paramerters in total".format(name,sum(x.numel() for x in model.parameters())/1e6))


    def forward(self, batch_img):
        """
        Args:
            batch_img (Tensor): Batch input image of shape
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Predicted coordinates of fiducial points for input batch.
            The shape is :math:`(N, F, 2)` where :math:`F` is ``num_fiducial``.
        """
        # batch_size = batch_img.size(0)
        logits = self.conv(batch_img)

        return logits


class UNetwork(nn.Module):
    """Localization Network of RARE, which predicts C' (K x 2) from input
    (img_width x img_height)

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        num_img_channel (int): Number of channels of the input image.
    """

    def __init__(self, num_img_channel,point_size,p_stride,u_channel = 2,tf_ratio=2,tf_layers=1):
        super().__init__()

        self.num_img_channel = num_img_channel
        self.point_x = point_size[1]
        self.point_y = point_size[0]
        # self.tf_ratio = tf_ratio
        # self.tf_layers = tf_layers
        self.tf_ratio = 4
        self.tf_layers = 1
        # if point_size[0] == 1:
        #     tuple_stride = (1,2)
        #     self.point_y = point_size[0] * 2
        # else:
        #     tuple_stride = 2
        #     self.point_y = point_size[0]
        # if self.point_x // self.point_y == 2:
        #     tuple_stride = (1, 2)
        # else:
        #     tuple_stride = 2
        # self.conv = Unet_Tiny(
        self.conv = Unet_down3(
        # self.conv = Unet(
        # self.conv = Unet_Base(
                            in_channels=num_img_channel,
                            # num_channels= num_img_channel if num_img_channel<64 else 64,
                            num_channels = 64,
                            stride = p_stride,
                            u_channel = u_channel,
        )
        # self.point_y = point_size[0]
        self.point_x = point_size[1]
        self.num_fiducial = self.point_y * self.point_x

        self.atten = ModuleList([
            TFCommonDecoderLayer(d_model=num_img_channel,
                                 d_inner=num_img_channel * self.tf_ratio,
                                 n_head=4,
                                 d_k=16,
                                 d_v=16,
                                 ifmask=False, )
            for _ in range(self.tf_layers)
        ])

        # self.sa_atten = ModuleList([
        #     TFCommonDecoderLayer(d_model=num_img_channel,
        #                          d_inner=num_img_channel * self.tf_ratio,
        #                          n_head=4,
        #                          d_k=16,
        #                          d_v=16,
        #                          ifmask=False, )
        #     for _ in range(self.tf_layers)
        # ])

        # self.localization_fc1 = nn.Sequential(
        #     nn.Linear(num_img_channel if num_img_channel<64 else 64, 512),
        #     nn.ReLU(True),
        # )
        self.localization_fc1 = nn.Sequential(
            # nn.Linear(num_img_channel if num_img_channel < 64 else 64, 512),
            nn.Linear(num_img_channel, 256),
            nn.ReLU(True),
            nn.Linear(256,2),
            nn.ReLU(True),
        )

        # self.down_linear = nn.Linear(num_img_channel * 4 , num_img_channel)
        # self.localization_fc2 = nn.Linear(512*self.num_fiducial, self.num_fiducial*2)
        self.localization_fc2 = nn.Linear(2 * self.num_fiducial, self.num_fiducial * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(0.1, self.point_x-0.1 , num=int(self.point_x)) / self.point_x
        ctrl_pts_y = np.linspace(0.1, self.point_y-0.1 , num=int(self.point_y)) / self.point_y #X * Y * 2
        # ctrl_pts_x = np.linspace(-1.0, 1.0, num=int(self.point_x))
        # ctrl_pts_y = np.linspace(-1.0, 1.0, num=int(self.point_y))  # X * Y * 2
        initial_bias = np.stack(np.meshgrid(ctrl_pts_x, ctrl_pts_y), axis=2)
        # initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(
            initial_bias).float().view(-1)

        # count param
        self.count_param(self.conv,'Unet')

        self.count_param(self.atten, 'atten')

        # self.count_param(self.localization_fc2, 'localization_fc2')

    def count_param(self, model,name):
        print("{} have {}M paramerters in total".format(name,sum(x.numel() for x in model.parameters())/1e6))


    def forward(self, batch_img):
        """
        Args:
            batch_img (Tensor): Batch input image of shape
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Predicted coordinates of fiducial points for input batch.
            The shape is :math:`(N, F, 2)` where :math:`F` is ``num_fiducial``.
        """
        batch_size = batch_img.size(0)
        logits = self.conv(batch_img)
        point = logits['point']
        B, C, H, W = logits['feature'].size()
        point = rearrange(point, 'b c h w -> b (h w) c')
        feat = rearrange(logits['feature'], 'b c h w -> b (h w) c')
        # point = self.down_linear(point)
        # for sa_atten_layer in self.sa_atten:
        #     point = sa_atten_layer(point,point,point, mask=None, ifmask=False)
        for atten_layer in self.atten:
            feat = atten_layer(feat,point,point, mask=None, ifmask=False)
        feat = rearrange(feat, 'b (h w) c -> b c h w',h=H,w=W)

        # batch_img = logits['feature']
        # logits = self.conv(batch_img).view(batch_size, -1)

        batch_C_prime = self.localization_fc2(
            self.localization_fc1(point).view(batch_size,-1)).view(batch_size,
                                                  self.num_fiducial, 2)
        return {"point": batch_C_prime,"feature": feat,"point_feat":point}


class UNetwork_Warp(nn.Module):
    """Localization Network of RARE, which predicts C' (K x 2) from input
    (img_width x img_height)

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        num_img_channel (int): Number of channels of the input image.
    """

    def __init__(self, num_img_channel,point_size,p_stride,u_channel=2, unet_type="8_8"):
        super().__init__()

        self.num_img_channel = num_img_channel
        self.point_x = point_size[1]
        self.point_y = point_size[0]
        # self.tf_ratio = tf_ratio
        # self.tf_layers = tf_layers
        self.tf_ratio = 4
        self.tf_layers = 1
        # self.conv = Unet_Tiny(
        # self.conv = Unet_down3(
        if unet_type =="4_4":
            self.conv = Unet(
                                in_channels=num_img_channel,
                                num_channels = 64,
                                stride = p_stride,
                                u_channel = u_channel,
            )
        elif unet_type =="8_8":
            self.conv = Unet_down3(
                                in_channels=num_img_channel,
                                num_channels = 64,
                                stride = p_stride,
                                u_channel = u_channel,
            )
        # self.point_y = point_size[0]
        self.point_x = point_size[1]
        self.num_fiducial = self.point_y * self.point_x

        self.atten = ModuleList([
            TFCommonDecoderLayer(d_model=num_img_channel,
                                 d_inner=num_img_channel * self.tf_ratio,
                                 n_head=4,
                                 d_k=16,
                                 d_v=16,
                                 ifmask=False, )
            for _ in range(self.tf_layers)
        ])

        self.localization_fc1 = nn.Sequential(
            # nn.Linear(num_img_channel if num_img_channel < 64 else 64, 512),
            nn.Linear(num_img_channel, 256),
            nn.ReLU(True),
            nn.Linear(256,2),
            nn.ReLU(True),
        )

        # self.down_linear = nn.Linear(num_img_channel * 4 , num_img_channel)
        # self.localization_fc2 = nn.Linear(512*self.num_fiducial, self.num_fiducial*2)
        self.localization_fc2 = nn.Linear(2 * self.num_fiducial, self.num_fiducial * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(0.1, self.point_x-0.1 , num=int(self.point_x)) / self.point_x
        ctrl_pts_y = np.linspace(0.1, self.point_y-0.1 , num=int(self.point_y)) / self.point_y #X * Y * 2
        # ctrl_pts_x = np.linspace(-1.0, 1.0, num=int(self.point_x))
        # ctrl_pts_y = np.linspace(-1.0, 1.0, num=int(self.point_y))  # X * Y * 2
        initial_bias = np.stack(np.meshgrid(ctrl_pts_x, ctrl_pts_y), axis=2)
        # initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(
            initial_bias).float().view(-1)

    def forward(self, batch_img):
        """
        Args:
            batch_img (Tensor): Batch input image of shape
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Predicted coordinates of fiducial points for input batch.
            The shape is :math:`(N, F, 2)` where :math:`F` is ``num_fiducial``.
        """
        batch_size = batch_img.size(0)
        logits = self.conv(batch_img)
        point = logits['point']
        B, C, H, W = logits['feature'].size()
        point = rearrange(point, 'b c h w -> b (h w) c')
        feat = rearrange(logits['feature'], 'b c h w -> b (h w) c')
        # point = self.down_linear(point)
        # for sa_atten_layer in self.sa_atten:
        #     point = sa_atten_layer(point,point,point, mask=None, ifmask=False)
        for atten_layer in self.atten:
            feat = atten_layer(feat,point,point, mask=None, ifmask=False)
        feat = rearrange(feat, 'b (h w) c -> b c h w',h=H,w=W)

        # batch_img = logits['feature']
        # logits = self.conv(batch_img).view(batch_size, -1)

        batch_C_prime = self.localization_fc2(
            self.localization_fc1(point).view(batch_size,-1)).view(batch_size,
                                                  self.num_fiducial, 2)
        return {"point": batch_C_prime,"feature": feat,"point_feat":point}


class LocalizationNetwork(nn.Module):
    """Localization Network of RARE, which predicts C' (K x 2) from input
    (img_width x img_height)

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        num_img_channel (int): Number of channels of the input image.
    """

    def __init__(self, num_fiducial, num_img_channel):
        super().__init__()
        self.num_fiducial = num_fiducial
        self.num_img_channel = num_img_channel
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_img_channel,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 64 x img_height/2 x img_width/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 128 x img_h/4 x img_w/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 256 x img_h/8 x img_w/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)  # batch_size x 512
        )

        self.localization_fc1 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.num_fiducial * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)


        ctrl_pts_x = np.linspace(-1.0, 1.0, int(num_fiducial / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(num_fiducial / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(num_fiducial / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(
            initial_bias).float().view(-1)

    def forward(self, batch_img):
        """
        Args:
            batch_img (Tensor): Batch input image of shape
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Predicted coordinates of fiducial points for input batch.
            The shape is :math:`(N, F, 2)` where :math:`F` is ``num_fiducial``.
        """
        batch_size = batch_img.size(0)
        features = self.conv(batch_img).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(
            self.localization_fc1(features)).view(batch_size,
                                                  self.num_fiducial, 2)
        return batch_C_prime
