# Modified from https://github.com/clovaai/deep-text-recognition-benchmark
#
# Licensed under the Apache License, Version 2.0 (the "License");s
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn import ModuleList

# from mmocr.models import PREPROCESSOR
from mmocr.models.textrecog.backbones.tps_pp.DGAB import DGAB
from mmocr.models.builder import PREPROCESSOR
from .base_preprocessor import BasePreprocessor

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

class Encoder_Decoder_Feature_Extractor(nn.Module):
    def __init__(self, in_channels=512,
                 num_channels=64,
                 attn_mode='nearest',
                 stride=2,
                 ratio=[1, 2, 4],
                 u_channel=2,
                 ):
        super().__init__()
        self.stride = stride
        self.k_encoder = nn.Sequential(
            self._encoder_layer(in_channels * u_channel, num_channels * ratio[0], stride=2),
            self._encoder_layer(num_channels * ratio[0], num_channels * ratio[1], stride=2),
            self._encoder_layer(num_channels * ratio[1], num_channels * ratio[2], stride=stride),
            self._encoder_layer(num_channels * ratio[2], num_channels * ratio[2], stride=(2,1)),
            # self._encoder_layer(num_channels, num_channels, stride=(1,2))
        )

        self.atten = CBAM(num_channels * ratio[2])

        self.k_decoder = nn.Sequential(
            self._decoder_layer(num_channels * ratio[2], num_channels * ratio[2], scale_factor=(2,1), mode=attn_mode),
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
                scale_factor=2,
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
        features = []
        for i in range(len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        point = features[-1]

        k = self.atten(point)

        for i in range(len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)
        return {'decoded_feature':k, 'encoded_feature':point}


class Multi_Scale_Fearue_Aggregation(nn.Module):
    """
    (img_width x img_height)

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        num_img_channel (int): Number of channels of the input image.
    """

    def __init__(self, num_img_channel,point_size,p_stride,u_channel = 2):
        super().__init__()

        self.num_img_channel = num_img_channel
        self.point_x = point_size[1]
        self.point_y = point_size[0]
        # self.tf_ratio = tf_ratio
        # self.tf_layers = tf_layers
        self.tf_ratio = 4
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
        self.conv = Encoder_Decoder_Feature_Extractor(
                            in_channels=num_img_channel,
                            num_channels = 64,
                            stride = p_stride,
                            u_channel = u_channel,
        )
        self.num_fiducial = self.point_y * self.point_x

        # count param
        self.count_param(self.conv,'Extractor')

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
        logits = self.conv(batch_img)
        en_feat = logits['encoded_feature']
        de_feat = logits['decoded_feature']

        return {"de_feat": de_feat,"en_feat":en_feat}

class Transformation_Parameter_Estimation(nn.Module):
    """
    (img_width x img_height)

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        num_img_channel (int): Number of channels of the input image.
    """

    def __init__(self, img_channel, point_channel,num_img_channel,point_size):
        super().__init__()

        self.num_img_channel = num_img_channel
        self.point_x = point_size[1]
        self.point_y = point_size[0]
        self.tf_layers = 1
        self.scale = num_img_channel ** -0.5
        self.without_as = False

        self.num_fiducial = self.point_y * self.point_x


        self.p_linear = nn.Sequential(
                            nn.Linear(point_channel,32),
                            nn.Linear(32, 64 * 2),
        )
        # self.down_feat = ConvModule(self.img_channel * 3, self.img_channel, kernel_size=1, stride=1,)
        self.feat_linear = nn.Sequential(
            nn.Linear(img_channel, 32),
            nn.Linear(32, 64 * 2),
        )

        self.atten = ModuleList([
            DGAB(num_img_channel*2, point = self.num_fiducial)
            for _ in range(self.tf_layers)
        ])
        print(self.atten)

        self.localization_fc1 = nn.Sequential(
            nn.Linear(num_img_channel, 256),
            nn.ReLU(True),
            nn.Linear(256,2),
            nn.ReLU(True),
        )
        self.localization_fc2 = nn.Linear(2 * self.num_fiducial, self.num_fiducial * 2)
        self.down_mlp = nn.Linear(num_img_channel * 4, num_img_channel)
        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(0.1, self.point_x-0.1 , num=int(self.point_x)) / self.point_x
        ctrl_pts_y = np.linspace(0.1, self.point_y-0.1 , num=int(self.point_y)) / self.point_y #X * Y * 2
        initial_bias = np.stack(np.meshgrid(ctrl_pts_x, ctrl_pts_y), axis=2)
        # initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(
            initial_bias).float().view(-1)

        # count param
        self.count_param(self.atten, 'DGAB_block')

    def count_param(self, model, name):
        print("{} have {}M paramerters in total".format(name, sum(x.numel() for x in model.parameters()) / 1e6))

    def atten_score(self, a, b):
        attn = torch.einsum('b m c, b n c -> b m n', a, b)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        # attn = F.softmax(attn, dim=2)
        # attn = torch.sigmoid(attn)
        attn = torch.tanh(attn)

        return attn

    def get_score(self, point, feat):
        feat = rearrange(feat, 'b c h w -> b (h w) c')
        p1 = self.p_linear(point)
        f = self.feat_linear(feat)
        # cc_score =self.atten_score(p1,p2)
        pc_score = self.atten_score(f, p1)

        if self.without_as == True:
            pc_score = torch.zeros_like(pc_score)
        return pc_score


    def forward(self, en_feat,de_feat):
        batch_size,_,_,_ = en_feat.size()
        en_feat = rearrange(en_feat, 'b c h w -> b (h w) c')
        en_feat = self.down_mlp(en_feat)
        for atten_layer in self.atten:
            de_feat = atten_layer(de_feat, en_feat)

        control_point = self.localization_fc2(
            self.localization_fc1(en_feat).view(batch_size, -1)).view(batch_size,
                                                                      self.num_fiducial, 2)
        atten_score = self.get_score(en_feat, de_feat)
        return control_point,atten_score


class Attention_Enhanced_TPS(nn.Module):
    """Grid Generator of TPS_PP, which produces P_prime by multiplying T with P.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        rectified_img_size (tuple(int, int)):
            Size :math:`(H_r, W_r)` of the rectified image.
    """

    def __init__(self, rectified_img_size,point_size):
        """Generate P_hat and inv_delta_C for later."""
        super().__init__()
        self.eps = 1e-6
        self.thela = 0.5
        self.point_size = point_size

        self.point_y = point_size[0]
        # if self.point_y == 1 :
        #     self.point_y = self.point_y * 2
        self.point_x = point_size[1]
        self.num_fiducial = self.point_y * self.point_x

        self.rectified_img_height = rectified_img_size[0]
        self.rectified_img_width = rectified_img_size[1]
        # self.num_fiducial = num_fiducial
        self.C = self._build_C()  # num_fiducial x 2
        self.P = self._build_P(self.rectified_img_width,
                               self.rectified_img_height)
        # for multi-gpu, you need register buffer
        self.register_buffer(
            'hat_C',
            torch.tensor(self._build_hat_C(
                self.num_fiducial,
                self.C)).float())  # num_fiducial+3 x num_fiducial+3
        self.register_buffer('P_hat',
                             torch.tensor(
                                 self._build_P_hat(
                                     self.num_fiducial, self.C,
                                     self.P)).float())  # n x num_fiducial+3

    def _build_C(self):
        """Return coordinates of fiducial points in rectified_img; C.
        +++++
        -----
        +++++
        """
        # ctrl_pts_x = np.linspace(-0.9, 0.9, num=int(self.point_x))
        # ctrl_pts_y = np.linspace(-0.9, 0.9, num=int(self.point_y))  # X * Y * 2
        ctrl_pts_x = np.linspace(0.5, self.point_x - 0.5, num=int(self.point_x)) / self.point_x
        ctrl_pts_y = np.linspace(0.5, self.point_y - 0.5, num=int(self.point_y)) / self.point_y #X * Y * 2
        C = np.stack(np.meshgrid(ctrl_pts_x, ctrl_pts_y), axis=2).reshape([-1,2])

        return C  # num_fiducial x 2

    def _build_hat_C(self, num_fiducial, C):
        """Return inv_delta_C which is needed to calculate T."""
        hat_C = np.zeros((num_fiducial, num_fiducial), dtype=float)
        for i in range(0, num_fiducial):
            for j in range(i, num_fiducial):
                r = np.linalg.norm(C[i] - C[j])  #sqrt(i-j)^2 [euclidean distance]
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)  #主对角线填充1
        hat_C = (hat_C**2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # num_fiducial+3 x num_fiducial+3
            [
                np.concatenate([np.ones((num_fiducial, 1)), C, hat_C],
                               axis=1),  # num_fiducial x num_fiducial+3
                np.concatenate([np.zeros(
                    (2, 3)), np.transpose(C)], axis=1),  # 2 x num_fiducial+3
                np.concatenate([np.zeros(
                    (1, 3)), np.ones((1, num_fiducial))],
                               axis=1)  # 1 x num_fiducial+3
            ],
            axis=0)
        inv_delta_C = np.linalg.inv(delta_C) #delta_C 取反
        return inv_delta_C  # num_fiducial+3 x num_fiducial+3
        # return hat_C

    def build_inv_delta_C(self,hat_C, cc_score,device):
        '''
            hat_C : B, num_fiducial, num_fiducial
            cc_score: B, num_fiducial, num_fiducial
            C : num_fiducial ,2
        '''
        # print(C.shape, hat_C.shape)
        B, num_fiducial, _ = hat_C.size()
        C = torch.tensor(self.C).float().to(device).unsqueeze(0).repeat(B, 1, 1)
        # hat_C = hat_C * cc_score + hat_C
        # cc_score = cc_score * 2
        hat_C = hat_C * (cc_score * 0.1 + 1)
        # hat_C = hat_C * (cc_score * 2 + self.eps)
        delta_C = torch.cat(  # num_fiducial+3 x num_fiducial+3
            [
                torch.cat([torch.ones((B, num_fiducial, 1)).to(device), C, hat_C],
                          dim=2),  # num_fiducial x num_fiducial+3
                torch.cat([torch.zeros(
                    (B, 2, 3)).to(device), C.transpose(2, 1)], dim=2),  # 2 x num_fiducial+3
                torch.cat([torch.zeros(
                    (B, 1, 3)).to(device), torch.ones((B, 1, num_fiducial)).to(device)],
                          dim=2)  # 1 x num_fiducial+3
            ],
            dim=1)
        #     print(delta_C)
        inv_delta_C = torch.inverse(delta_C)
        # inv_delta_C = torch.linalg.inv(delta_C) #delta_C 取反
        return inv_delta_C  # num_fiducial+3 x num_fiducial+3

    def _build_P(self, rectified_img_width, rectified_img_height):
        '''
        meshgrid for P in rectified_img
        '''

        rectified_img_grid_x = np.linspace(0.5, rectified_img_width - 0.5, num=int(rectified_img_width)) / rectified_img_width
        rectified_img_grid_y = np.linspace(0.5, rectified_img_height - 0.5, num=int(rectified_img_height)) / rectified_img_height

        P = np.stack(  # self.rectified_img_w x self.rectified_img_h x 2
            np.meshgrid(rectified_img_grid_x, rectified_img_grid_y),
            axis=2)
        return P.reshape([
            -1, 2
        ])  # n (= self.rectified_img_width x self.rectified_img_height) x 2

    def _build_P_hat(self, num_fiducial, C, P):
        n = P.shape[
            0]  # n (= self.rectified_img_width x self.rectified_img_height)
        P_tile = np.tile(np.expand_dims(P, axis=1),
                         (1, num_fiducial,
                          1))  # n x 2 -> n x 1 x 2 -> n x num_fiducial x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x num_fiducial x 2
        P_diff = P_tile - C_tile  # n x num_fiducial x 2
        rbf_norm = np.linalg.norm(
            P_diff, ord=2, axis=2, keepdims=False)  # n x num_fiducial
        P_hat = np.multiply(np.square(rbf_norm),
                          np.log(rbf_norm + self.eps))  # n x num_fiducial
        # P_hat = np.concatenate([np.ones((n, 1)), P, P_hat], axis=1)
        return P_hat  # n x num_fiducial+3

    def P_hat_score_process(self, P_hat,  pc_score, device):
        '''
        P_hat: B, n , num_fiducial
        '''
        B,n,_ = pc_score.size()
        P = torch.tensor(self.P).float().to(device).unsqueeze(0).repeat(B, 1, 1)
        # pc_score = pc_score * 2
        P_hat = P_hat * (pc_score * self.thela + 1)
        # P_hat = P_hat * (pc_score + 1)
        # P_hat = P_hat * (pc_score * 2 + self.eps)
        P_hat = torch.cat([torch.ones((B, n, 1)).to(device), P, P_hat], dim=2)

        return P_hat

    def build_P_prime(self, batch_C_prime, pc_score,device='cuda'):
        """Generate Grid from batch_C_prime [batch_size x num_fiducial x 2]"""
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.hat_C.to(device).repeat(batch_size, 1, 1)

        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat_score_process(batch_P_hat,pc_score, device)

        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)),
            dim=1)  # batch_size x num_fiducial+3 x 2
        batch_T = torch.bmm(
            batch_inv_delta_C,
            batch_C_prime_with_zeros)  # batch_size x num_fiducial+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2


@PREPROCESSOR.register_module()
class TPS_PPv2(BasePreprocessor):
    '''
    new multi-layers
    for paper tps_pp final version
    '''
    def __init__(self,
                 num_fiducial=64,
                 img_size=(32, 128),
                 rectified_img_size=(32, 128),
                 num_img_channel=64,
                 point_size=(2,16),
                 p_stride=2,
                 visual_point=False,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_fiducial, int)
        assert num_fiducial > 0
        assert isinstance(img_size, tuple)
        assert isinstance(rectified_img_size, tuple)
        # assert isinstance(num_img_channel, int)

        self.heads = 16
        pc_ratio = 1
        ic_ratio = 1

        self.visual_point = visual_point
        self.num_fiducial = point_size[0]*point_size[1]
        self.img_size = img_size
        self.point_size = point_size
        self.rectified_img_size = rectified_img_size
        self.num_img_channel = num_img_channel
        self.point_channel = num_img_channel * pc_ratio
        self.img_channel = num_img_channel * ic_ratio

        self.MSFA = Multi_Scale_Fearue_Aggregation(self.num_img_channel,self.point_size,p_stride, u_channel = 1)
        self.TPE = Transformation_Parameter_Estimation(self.point_channel,self.img_channel,self.num_img_channel,self.point_size)
        # self.scale = self.num_img_channel ** -0.5

        #ResNet45:
        self.down0 = ConvModule(3, self.img_channel, kernel_size=1, stride=1)

        self.atten_tps = Attention_Enhanced_TPS(self.rectified_img_size,self.point_size)
        self.count_param(self.MSFA, 'TPS_PP')

    def count_param(self, model, name):
        print("{} have {}M paramerters in total".format(name, sum(x.numel() for x in model.parameters()) / 1e6))

    def forward(self, batch_img):
        """
        Args:
            batch_img (Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Rectified image with size :math:`(N, C, H_r, W_r)`.
        """
        feat_cat = self.down0(batch_img)

        # res = batch_img
        logits = self.MSFA(
            feat_cat)
        de_feat = logits['de_feat']
        en_feat = logits['en_feat']

        control_point, atten_score = self.TPE(en_feat,de_feat)
        # pc_score = self.get_score(en_feat,de_feat)

        build_P_prime = self.atten_tps.build_P_prime(
            control_point, atten_score, batch_img.device
        )

        build_P_prime_reshape = build_P_prime.reshape([
            build_P_prime.size(0), self.rectified_img_size[0],
            self.rectified_img_size[1], 2
        ])

        batch_rectified_img = F.grid_sample(
            batch_img,
            build_P_prime_reshape,
            padding_mode='border',
            align_corners=True)

        # return result
        return batch_rectified_img