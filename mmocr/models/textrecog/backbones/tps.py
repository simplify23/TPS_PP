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
import einops
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList, load_checkpoint
from thop import profile
from einops import rearrange, reduce, repeat
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, Sequential
from timm.models.layers import to_2tuple, trunc_normal_
from mmocr.utils.model import revert_sync_batchnorm
from mmocr.models.builder import BACKBONES
from mmocr.models.textrecog.backbones.utils.component import Fuser, CBAM, Unet
from mmocr.models.textrecog.backbones.utils.extractor import UNetwork, AttentionNetwork, Unet_wrap, PointNetwork, \
    LocalizationNetwork, UNetwork_Warp
from mmocr.models.textrecog.decoders.transformer_mask import TransDecoderLayer, TFCommonDecoderLayer
from tools.data.textrecog.visual_feat import draw_point_map


@BACKBONES.register_module()
class Deform_net(BaseModule):
    """Rectification Network of RARE, namely TPS based STN in
    https://arxiv.org/pdf/1603.03915.pdf.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        img_size (tuple(int, int)): Size :math:`(H, W)` of the input image.
        rectified_img_size (tuple(int, int)): Size :math:`(H_r, W_r)` of
            the rectified image.
        num_img_channel (int): Number of channels of the input image.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 num_fiducial=20,
                 img_size=(32, 100),
                 rectified_img_size=(32, 100),
                 num_img_channel=1,
                 point_size=(4,16),
                 p_stride=1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_fiducial, int)
        assert num_fiducial > 0
        assert isinstance(img_size, tuple)
        assert isinstance(rectified_img_size, tuple)
        assert isinstance(num_img_channel, int)

        self.num_fiducial = point_size[0]*point_size[1]
        self.img_size = img_size
        self.point_size = point_size
        self.rectified_img_size = rectified_img_size
        self.num_img_channel = num_img_channel
        self.deform_net = nn.Sequential(
                            ConvModule(num_img_channel,num_img_channel//2,3,1,1),
                            CBAM(num_img_channel//2),
                            ConvModule(num_img_channel//2, 2, 3, 1, 1),
        )
        self.fc2 = nn.Linear(2 * self.num_fiducial, self.num_fiducial * 2)

        self.atten = Fuser(num_img_channel)
        # self.GridGenerator = GridGenerator(self.rectified_img_size,self.point_size)

        # Init fc2 in LocalizationNetwork
        self.fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(-1.0, 1.0, num=int(self.point_size[0]))
        ctrl_pts_y = np.linspace(-1.0, 1.0, num=int(self.point_size[1]))  # X * Y * 2
        initial_bias = np.stack(np.meshgrid(ctrl_pts_x, ctrl_pts_y), axis=2)
        # initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.fc2.bias.data = torch.from_numpy(
            initial_bias).float().view(-1)

    def forward(self, batch_img,point=None):
        """
        Args:
            batch_img (Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Rectified image with size :math:`(N, C, H_r, W_r)`.
        """
        residual = batch_img

        offset = self.deform_net(
            batch_img)
        # batch_C_prime = logits['point']

        offset = rearrange(offset, 'b c h w -> b (h w c)')
        offset = self.fc2(offset)
        offset = offset.reshape([
            -1, self.rectified_img_size[0],
            self.rectified_img_size[1], 2
        ])

        batch_rectified_img = F.grid_sample(
            batch_img,
            offset,
            padding_mode='border',
            align_corners=True)

        output = self.atten(batch_rectified_img,residual)
        # return batch_rectified_img,p_feature
        return output

@BACKBONES.register_module()
class TPSnet_Warp(BaseModule):
    """Rectification Network of RARE, namely TPS based STN in
    https://arxiv.org/pdf/1603.03915.pdf.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        img_size (tuple(int, int)): Size :math:`(H, W)` of the input image.
        rectified_img_size (tuple(int, int)): Size :math:`(H_r, W_r)` of
            the rectified image.
        num_img_channel (int): Number of channels of the input image.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 num_fiducial=20,
                 img_size=(32, 100),
                 rectified_img_size=(32, 100),
                 num_img_channel=1,
                 point_size=(4,16),
                 p_stride=1,
                 down_iter=0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_fiducial, int)
        assert num_fiducial > 0
        assert isinstance(img_size, tuple)
        assert isinstance(rectified_img_size, tuple)
        assert isinstance(num_img_channel, int)

        self.heads = 16
        self.hidden_unit = 64
        self.num_fiducial = point_size[0]*point_size[1]
        self.img_size = img_size
        self.point_size = point_size
        self.rectified_img_size = rectified_img_size
        self.num_img_channel = num_img_channel
        # self.atten = Fuser(num_img_channel)
        # self.Unet = UNetwork(self.num_img_channel,self.point_size,p_stride)
        self.scale = self.num_img_channel ** -0.5
        self.down_layers = ModuleList([
            ConvModule(self.hidden_unit, self.hidden_unit, kernel_size=3, stride=2, padding=1)
            for _ in range(down_iter-2)
        ])
        self.up_layers = ModuleList([
                            self._decoder_layer(2, 2, scale_factor=2, mode='nearest')
                            for _ in range(down_iter - 2)
                        ])
        # self.GridGenerator_atten = GridGenerator_Atten(self.rectified_img_size,self.point_size)
        self.down_conv = nn.Sequential(
                        ConvModule(num_img_channel, self.hidden_unit, kernel_size=3, stride=1, padding=1),
                        # ConvModule(self.hidden_unit, self.hidden_unit, kernel_size=3, stride=2, padding=1),
        )
        self.up_conv = nn.Sequential(
                        ConvModule(2,num_img_channel,kernel_size=3,stride=1,padding=1),
                        ConvModule(num_img_channel, 2, kernel_size=3, stride=1, padding=1),
                        # self._decoder_layer(2, 2, scale_factor=2, mode='nearest'),
        )
        self.scale = self.num_img_channel ** -0.5

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



    def count_param(self, model,name):
        print("{} have {}M paramerters in total".format(name,sum(x.numel() for x in model.parameters())/1e6))


    def forward(self, batch_img,epoch = 0, share_model=None):
        """
        Args:
            batch_img (Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Rectified image with size :math:`(N, C, H_r, W_r)`.
        """
        # res = batch_img

        feat = self.down_conv(batch_img)
        for layers in self.down_layers:
            feat = layers(feat)
        feat = share_model(feat)
        feat = self.up_conv(feat)
        for up_layers in self.up_layers:
            feat = up_layers(feat)
        feat = rearrange(feat, 'b h m n -> b m n h')
        batch_rectified_img = F.grid_sample(
            batch_img,
            feat,
            padding_mode='border',
            align_corners=True)

        # return batch_rectified_img,p_feature
        return batch_rectified_img

@BACKBONES.register_module()
class TPSnet(BaseModule):
    """Rectification Network of RARE, namely TPS based STN in
    https://arxiv.org/pdf/1603.03915.pdf.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        img_size (tuple(int, int)): Size :math:`(H, W)` of the input image.
        rectified_img_size (tuple(int, int)): Size :math:`(H_r, W_r)` of
            the rectified image.
        num_img_channel (int): Number of channels of the input image.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 num_fiducial=20,
                 img_size=(32, 100),
                 rectified_img_size=(32, 100),
                 num_img_channel=1,
                 point_size=(4,16),
                 p_stride=1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_fiducial, int)
        assert num_fiducial > 0
        assert isinstance(img_size, tuple)
        assert isinstance(rectified_img_size, tuple)
        assert isinstance(num_img_channel, int)

        self.heads = 16
        self.num_fiducial = point_size[0]*point_size[1]
        self.img_size = img_size
        self.point_size = point_size
        self.rectified_img_size = rectified_img_size
        self.num_img_channel = num_img_channel
        # self.atten = Fuser(num_img_channel)
        # self.Unet = UNetwork(self.num_img_channel,self.point_size,p_stride)
        self.scale = self.num_img_channel ** -0.5
        self.GridGenerator_atten = GridGenerator_Atten(self.rectified_img_size,self.point_size)
        self.Unet = UNetwork(self.num_img_channel, self.point_size, p_stride)
        self.scale = self.num_img_channel ** -0.5
        self.p_linear = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32, 64 * 2),
        )
        self.p2_linear = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32, 64 * 2),
        )
        # self.cbam_pc = CBAM(self.heads,ratio=-4)
        # self.cbam_cc = CBAM(self.heads,ratio=-4)
        self.feat_linear = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32, 64 * 2),
        )

    def count_param(self, model,name):
        print("{} have {}M paramerters in total".format(name,sum(x.numel() for x in model.parameters())/1e6))

    def atten_score(self,a,b):
        attn = torch.einsum('b m c, b n c -> b m n', a, b)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        # attn = torch.sigmoid(attn)
        attn = torch.tanh(attn)
        # attn = F.softmax(attn, dim=2)
        return attn

    def get_score(self,point,feat,epoch):
        feat = rearrange(feat, 'b c h w -> b (h w) c')
        p1 = self.p_linear(point)
        p2 = self.p2_linear(point)
        f = self.feat_linear(feat)
        cc_score =self.atten_score(p1,p2)
        # cc_score = 0.0
        # pc_score = 0.0
        pc_score = self.atten_score(f,p1)
        if epoch <= 3:
            pc_score = torch.zeros_like(pc_score)
            pc_score = torch.zeros_like(pc_score)
        return cc_score,pc_score

    def forward(self, batch_img,epoch = 0, point=None):
        """
        Args:
            batch_img (Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Rectified image with size :math:`(N, C, H_r, W_r)`.
        """
        res = batch_img
        logits = self.Unet(
            batch_img)
        batch_C_prime = logits['point']
        batch_img = logits['feature']
        point_feat = logits['point_feat']

        # cc_score, pc_score = 0.0, 0.0
        cc_score,pc_score = self.get_score(point_feat,batch_img,epoch)
        build_P_prime = self.GridGenerator_atten.build_P_prime(
            batch_C_prime, cc_score,pc_score,batch_img.device
        )
        build_P_prime_reshape = build_P_prime.reshape([
            build_P_prime.size(0),  self.rectified_img_size[0],
            self.rectified_img_size[1], 2
        ])
        build_P_prime_reshape = rearrange(build_P_prime_reshape, 'b m n h -> b h m n')
        # batch_rectified_img = F.grid_sample(
        #     batch_img,
        #     build_P_prime_reshape,
        #     padding_mode='border',
        #     align_corners=True)

        # return batch_rectified_img,p_feature
        return build_P_prime_reshape

@BACKBONES.register_module()
class TPSnetv2(BaseModule):
    """Rectification Network of RARE, namely TPS based STN in
    https://arxiv.org/pdf/1603.03915.pdf.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        img_size (tuple(int, int)): Size :math:`(H, W)` of the input image.
        rectified_img_size (tuple(int, int)): Size :math:`(H_r, W_r)` of
            the rectified image.
        num_img_channel (int): Number of channels of the input image.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 num_fiducial=20,
                 img_size=(32, 100),
                 rectified_img_size=(32, 100),
                 num_img_channel=1,
                 point_size=(4,16),
                 p_stride=1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_fiducial, int)
        assert num_fiducial > 0
        assert isinstance(img_size, tuple)
        assert isinstance(rectified_img_size, tuple)
        assert isinstance(num_img_channel, int)

        self.heads = 16
        self.num_fiducial = point_size[0]*point_size[1]
        self.img_size = img_size
        self.point_size = point_size
        self.rectified_img_size = rectified_img_size
        self.num_img_channel = num_img_channel
        self.atten = Fuser(num_img_channel)
        self.get_point = PointNetwork(self.num_img_channel*2,self.point_size,p_stride)
        # self.Unet = UNetwork(self.num_img_channel,self.point_size,p_stride)
        self.scale = self.num_img_channel ** -0.5
        self.p_linear = nn.Sequential(
                            nn.Linear(num_img_channel*2,32),
                            nn.Linear(32, 64 * 2),
        )
        self.p2_linear = nn.Sequential(
                            nn.Linear(num_img_channel, 32),
                            nn.Linear(32, 64 * 2),
        )
        self.down_conv = ConvModule(
            32,
            32,
            kernel_size=3,
            stride=2,
            padding=1,)
        # self.cbam_pc = CBAM(self.heads,ratio=-4)
        # self.cbam_cc = CBAM(self.heads,ratio=-4)
        self.feat_linear = nn.Sequential(
            nn.Linear(num_img_channel*2, 32),
            nn.Linear(32, 64 * 2),
        )
        self.tps = LocalizationNetwork(20,3)
        self.GridGenerator_atten = GridGenerator_Atten(self.rectified_img_size,self.point_size)
        self.count_param(self.get_point, 'Total_PointNet')
        self.count_param(self.tps, 'Origin_Tps')

    def count_param(self, model,name):
        print("{} have {}M paramerters in total".format(name,sum(x.numel() for x in model.parameters())/1e6))

    def atten_score(self,a,b):
        attn = torch.einsum('b m c, b n c -> b m n', a, b)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        # attn = torch.sigmoid(attn)
        attn = torch.tanh(attn)
        # attn = F.softmax(attn, dim=2)
        return attn

    def get_score(self,point,feat,epoch):
        feat = rearrange(feat, 'b c h w -> b (h w) c')
        p1 = self.p_linear(point)
        # p2 = self.p2_linear(point)
        f = self.feat_linear(feat)
        # cc_score =self.atten_score(p1,p2)
        cc_score = 0.0
        pc_score = self.atten_score(f,p1)
        # if epoch <= 3:
        #     pc_score = torch.zeros_like(pc_score)
        return cc_score,pc_score

    def forward(self, batch_img,epoch = 0, outs=None, point=None):
        """
        Args:
            batch_img (Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Rectified image with size :math:`(N, C, H_r, W_r)`.
        """
        res = batch_img
        if outs != None:
            feat1 = self.down_conv(outs[0])
            batch_img = torch.cat( (feat1, outs[1],batch_img),dim=1)

        logits = self.get_point(
            batch_img)
        batch_C_prime = logits['point']
        batch_img = logits['feature']
        point_feat = logits['point_feat']

        cc_score,pc_score = self.get_score(point_feat,batch_img,epoch)
        build_P_prime = self.GridGenerator_atten.build_P_prime(
            batch_C_prime, cc_score,pc_score,batch_img.device
        )
        build_P_prime_reshape = build_P_prime.reshape([
            build_P_prime.size(0), self.rectified_img_size[0],
            self.rectified_img_size[1], 2
        ])

        batch_rectified_img = F.grid_sample(
            res,
            build_P_prime_reshape,
            padding_mode='border',
            align_corners=True)

        return batch_rectified_img


@BACKBONES.register_module()
class U_TPSnet(BaseModule):
    """Rectification Network of RARE, namely TPS based STN in
    https://arxiv.org/pdf/1603.03915.pdf.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        img_size (tuple(int, int)): Size :math:`(H, W)` of the input image.
        rectified_img_size (tuple(int, int)): Size :math:`(H_r, W_r)` of
            the rectified image.
        num_img_channel (int): Number of channels of the input image.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 num_fiducial=64,
                 img_size=(16, 64),
                 rectified_img_size=(16, 64),
                 num_img_channel=64,
                 point_size=(4,16),
                 p_stride=2,
                 visual_point=False,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_fiducial, int)
        assert num_fiducial > 0
        assert isinstance(img_size, tuple)
        assert isinstance(rectified_img_size, tuple)
        assert isinstance(num_img_channel, int)

        self.heads = 16
        pc_ratio = 1
        ic_ratio = 1

        self.visual_point = visual_point
        self.num_fiducial = point_size[0]*point_size[1]
        self.img_size = img_size
        self.point_size = point_size
        self.rectified_img_size = rectified_img_size
        self.num_img_channel = num_img_channel
        # self.atten = Fuser(num_img_channel)
        self.point_channel = num_img_channel * pc_ratio
        self.img_channel = num_img_channel * ic_ratio
        self.without_as =False

        self.Unet = UNetwork(self.num_img_channel,self.point_size,p_stride)
        self.scale = self.num_img_channel ** -0.5
        self.p_linear = nn.Sequential(
                            nn.Linear(self.point_channel,32),
                            nn.Linear(32, 64 * 2),
        )
        self.p2_linear = nn.Sequential(
                            nn.Linear(self.point_channel, 32),
                            nn.Linear(32, 64 * 2),
        )
        self.down_conv = ConvModule(
            32,
            32,
            kernel_size=3,
            stride=2,
            padding=1,)
        # self.cbam_pc = CBAM(self.heads,ratio=-4)
        # self.cbam_cc = CBAM(self.heads,ratio=-4)
        self.feat_linear = nn.Sequential(
            nn.Linear(self.img_channel, 32),
            nn.Linear(32, 64 * 2),
        )
        # self.cc_linear = nn.Sequential(
        #     # nn.Linear(16, 64),
        #     # nn.Linear(64, 1),
        #     nn.Linear(16, 1),
        # )
        # self.pc_linear = nn.Sequential(
        #     # nn.Linear(16, 64),
        #     # nn.Linear(64, 1),
        #     nn.Linear(16, 1),
        # )
        # self.head = 16
        # self.Unet = Unet(num_img_channel, num_img_channel, p_stride)
        # Unet(
        #     in_channels=num_img_channel,
        #     num_channels=num_img_channel if num_img_channel < 64 else 64,
        #     stride=p_stride
        # )
        # self.atten = AttentionNetwork(self.num_img_channel,self.point_size,p_stride)

        # self.LocalizationNetwork = LocalizationNetwork(self.num_fiducial,
        #                                                self.num_img_channel)
        # self.GridGenerator = GridGenerator(self.rectified_img_size,self.point_size)
        self.GridGenerator_atten = GridGenerator_Atten(self.rectified_img_size,self.point_size)
        # self.GridGenerator_plus = GridGenerator_Plus(
        #                             num_img_channel, self.rectified_img_size, self.point_size
        #                             )
        self.count_param(self.Unet, 'Total_PointNet')
        # self.load_ckpt(file_path="ckpt/ztl/reg/mmocr/Backbonev15_4_tps_base_10M_tf_2_unet_124_atten_score_PointNet_mulistage_layer2_training_end_to_end/latest.pth",device='cpu')

    def load_ckpt(self,file_path,device='cpu'):
        # self = revert_sync_batchnorm(self)
        load_checkpoint(self, file_path, map_location=device)

    # def freeze_network(self,):
    #     for layers in self:
    #         layers = getattr(self, layers)
    #         layers.eval()
    #         for name, parameter in layers.named_parameters():
    #             parameter.requires_grad = False
    #             # print("{}: {}".format(parameter, parameter.requries_grad))
    #     self.conv1.eval()
    #     for name, parameter in self.conv1.named_parameters():
    #         parameter.requires_grad = False
    #         # print("{}: {}".format(parameter, parameter.requries_grad))

    def count_param(self, model,name):
        print("{} have {}M paramerters in total".format(name,sum(x.numel() for x in model.parameters())/1e6))

    def atten_score(self,a,b):
        # q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        # k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        # v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        # a,b = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [a,b]]
        # attn = torch.einsum('b h m c, b h n c -> b h m n', a, b)
        # attn = rearrange(attn, 'b h m n -> b m n h')
        # attn = atten_linear(attn).squeeze(-1)
        attn = torch.einsum('b m c, b n c -> b m n', a, b)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        # attn = torch.sigmoid(attn)
        attn = torch.tanh(attn)
        # attn = F.softmax(attn, dim=2)
        return attn

    def get_score(self,point,feat,epoch):
        feat = rearrange(feat, 'b c h w -> b (h w) c')
        p1 = self.p_linear(point)
        f = self.feat_linear(feat)
        # cc_score =self.atten_score(p1,p2)
        cc_score = 0.0
        pc_score = self.atten_score(f,p1)
        if self.without_as == True:
        # if epoch <= 5 or self.without_as == True:
            pc_score = torch.zeros_like(pc_score)
        return cc_score,pc_score

    def forward(self, batch_img,epoch = 0, outs=None, point=None):
        """
        Args:
            batch_img (Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Rectified image with size :math:`(N, C, H_r, W_r)`.
        """
        if outs != None:
            feat1 = self.down_conv(outs[0])
            batch_img = torch.cat( (feat1, outs[1],batch_img),dim=1)

        res = batch_img
        logits = self.Unet(
            batch_img)
        batch_C_prime = logits['point']
        batch_img = logits['feature']
        point_feat = logits['point_feat']

        if self.visual_point == True:
            draw_point_map(einops.rearrange(batch_C_prime, 'b (h w) c -> b h w c', h=4, w=16))

        cc_score,pc_score = self.get_score(point_feat,batch_img,epoch)
        build_P_prime = self.GridGenerator_atten.build_P_prime(
            batch_C_prime, cc_score,pc_score,batch_img.device
        )
        build_P_prime_reshape = build_P_prime.reshape([
            build_P_prime.size(0), self.rectified_img_size[0],
            self.rectified_img_size[1], 2
        ])
        if self.visual_point == True:
            draw_point_map(build_P_prime_reshape)

        batch_rectified_img = F.grid_sample(
            batch_img,
            build_P_prime_reshape,
            padding_mode='border',
            align_corners=True)

        # batch_rectified_img = self.atten(batch_rectified_img, batch_img)
        # return batch_rectified_img,p_feature
        result = {
            'output': batch_rectified_img,
            'logits': None,
            'mp_img': build_P_prime_reshape
        }
        # return batch_rectified_img,build_P_prime_reshape
        return result

@BACKBONES.register_module()
class U_TPSnetv2(U_TPSnet):
    def __init__(self,
                 num_fiducial=64,
                 img_size=(16, 64),
                 rectified_img_size=(16, 64),
                 num_img_channel=64,
                 point_size=(4,16),
                 p_stride=2,
                 visual_point=False,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_fiducial, int)
        assert num_fiducial > 0
        assert isinstance(img_size, tuple)
        assert isinstance(rectified_img_size, tuple)
        assert isinstance(num_img_channel, int)

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
        self.without_as = False

        self.Unet = UNetwork(self.num_img_channel,self.point_size,p_stride)
        self.scale = self.num_img_channel ** -0.5
        self.p_linear = nn.Sequential(
                            nn.Linear(self.point_channel,32),
                            nn.Linear(32, 64 * 2),
        )
        self.p2_linear = nn.Sequential(
                            nn.Linear(self.point_channel, 32),
                            nn.Linear(32, 64 * 2),
        )
        self.down_conv = ConvModule(
            32,
            32,
            kernel_size=3,
            stride=2,
            padding=1,)

        self.feat_linear = nn.Sequential(
            nn.Linear(self.img_channel, 32),
            nn.Linear(32, 64 * 2),
        )

        self.cls = nn.Sequential(
            nn.Linear(self.img_channel, 256),
            nn.Linear(256, 37),
        )

        self.GridGenerator_atten = GridGenerator_Atten(self.rectified_img_size,self.point_size)
        # self.GridGenerator_plus = GridGenerator_Plus(
        #                             num_img_channel, self.rectified_img_size, self.point_size
        #                             )

        self.count_param(self.Unet, 'Total_PointNet')
        # self.load_ckpt(file_path="ckpt/ztl/reg/mmocr/Backbonev15_4_tps_base_10M_tf_2_unet_124_atten_score_PointNet_mulistage_layer2_training_end_to_end/latest.pth",device='cpu')

    # def get_score(self,point,feat,epoch):
    #     feat = rearrange(feat, 'b c h w -> b (h w) c')
    #     p1 = self.p_linear(point)
    #     f = self.feat_linear(feat)
    #     # cc_score =self.atten_score(p1,p2)
    #     cc_score = 0.0
    #     pc_score = self.atten_score(f,p1)
    #     if self.without_as == True:
    #         pc_score = torch.zeros_like(pc_score)
    #     return cc_score,pc_score

    def forward(self, batch_img,epoch = 0, outs=None, point=None):
        """
        Args:
            batch_img (Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Rectified image with size :math:`(N, C, H_r, W_r)`.
        """
        if outs != None:
            feat1 = self.down_conv(outs[0])
            batch_img = torch.cat((feat1, outs[1],batch_img),dim=1)

        # res = batch_img
        logits = self.Unet(
            batch_img)
        batch_C_prime = logits['point']
        batch_img = logits['feature']
        point_feat = logits['point_feat']

        if self.visual_point == True:
            draw_point_map(einops.rearrange(batch_C_prime, 'b (h w) c -> b h w c', h=4, w=16))

        cc_score,pc_score = self.get_score(point_feat,batch_img,epoch)
        build_P_prime = self.GridGenerator_atten.build_P_prime(
            batch_C_prime, cc_score,pc_score,batch_img.device
        )

        build_P_prime_reshape = build_P_prime.reshape([
            build_P_prime.size(0), self.rectified_img_size[0],
            self.rectified_img_size[1], 2
        ])
        if self.visual_point == True:
            draw_point_map(build_P_prime_reshape)

        batch_rectified_img = F.grid_sample(
            batch_img,
            build_P_prime_reshape,
            padding_mode='border',
            align_corners=True)

        logits = self.cls(point_feat)

        result = {
            'output': batch_rectified_img,
            'logits': logits,
            'mp_img': build_P_prime_reshape
        }
        return result


@BACKBONES.register_module()
class U_TPSnet_Warp(U_TPSnetv2):
    def __init__(self,
                 num_fiducial=64,
                 img_size=(16, 64),
                 rectified_img_size=(16, 64),
                 num_img_channel=64,
                 point_size=(4,16),
                 p_stride=2,
                 visual_point=False,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_fiducial, int)
        assert num_fiducial > 0
        assert isinstance(img_size, tuple)
        assert isinstance(rectified_img_size, tuple)
        assert isinstance(num_img_channel, int)

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
        self.without_as = False

        self.Unet = UNetwork(self.num_img_channel,self.point_size,p_stride, u_channel = 3)
        self.scale = self.num_img_channel ** -0.5
        self.p_linear = nn.Sequential(
                            nn.Linear(self.point_channel,32),
                            nn.Linear(32, 64 * 2),
        )

        # # ResNet31:
        # self.down0 = ConvModule(64,self.img_channel,kernel_size=3, stride=2,padding=1,)
        # self.down1 = ConvModule(128,self.img_channel, kernel_size=3,stride=2,padding=1,)
        # self.down2 = ConvModule(256, self.img_channel, kernel_size=1,stride=1,)

        #ResNet45:
        self.down0 = ConvModule(32, self.img_channel, kernel_size=3, stride=2, padding=1, )
        self.down1 = ConvModule(32, self.img_channel, kernel_size=1, stride=1 )
        self.down2 = ConvModule(64, self.img_channel, kernel_size=1, stride=1, )

        self.feat_linear = nn.Sequential(
            nn.Linear(self.img_channel, 32),
            nn.Linear(32, 64 * 2),
        )

        # self.cls = nn.Sequential(
        #     nn.Linear(self.img_channel, 256),
        #     nn.Linear(256, 37),
        # )

        self.GridGenerator_atten = GridGenerator_Atten(self.rectified_img_size,self.point_size)
        # self.GridGenerator_plus = GridGenerator_Plus(
        #                             num_img_channel, self.rectified_img_size, self.point_size
        #                             )

        self.count_param(self.Unet, 'Total_PointNet')
        # self.load_ckpt(file_path="ckpt/ztl/reg/mmocr/Backbonev15_4_tps_base_10M_tf_2_unet_124_atten_score_PointNet_mulistage_layer2_training_end_to_end/latest.pth",device='cpu')


    def forward(self, batch_img,epoch = 0, outs=None, point=None):
        """
        Args:
            batch_img (Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Rectified image with size :math:`(N, C, H_r, W_r)`.
        """
        feat_cat = batch_img
        if outs != None:
            feat0 = self.down0(outs[0])
            feat1 = self.down1(outs[1])
            feat2 = self.down2(batch_img)
            feat_cat = torch.cat((feat0, feat1, feat2),dim=1)

        # res = batch_img
        logits = self.Unet(
            feat_cat)
        batch_C_prime = logits['point']
        u_feat = logits['feature']
        point_feat = logits['point_feat']

        if self.visual_point == True:
            draw_point_map(einops.rearrange(batch_C_prime, 'b (h w) c -> b h w c', h=4, w=16))

        cc_score,pc_score = self.get_score(point_feat,u_feat,epoch)
        build_P_prime = self.GridGenerator_atten.build_P_prime(
            batch_C_prime, cc_score,pc_score,batch_img.device
        )

        build_P_prime_reshape = build_P_prime.reshape([
            build_P_prime.size(0), self.rectified_img_size[0],
            self.rectified_img_size[1], 2
        ])
        if self.visual_point == True:
            draw_point_map(build_P_prime_reshape)

        batch_rectified_img = F.grid_sample(
            batch_img,
            build_P_prime_reshape,
            padding_mode='border',
            align_corners=True)

        logits = self.cls(point_feat)

        result = {
            'output': batch_rectified_img,
            'logits': logits,
            'mp_img': build_P_prime_reshape
        }
        return result

@BACKBONES.register_module()
class U_TPSnet_v3(BaseModule):
    '''
    new multi-layers
    '''
    def __init__(self,
                 num_fiducial=64,
                 img_size=(16, 64),
                 rectified_img_size=(16, 64),
                 num_img_channel=64,
                 point_size=(4,16),
                 p_stride=2,
                 visual_point=False,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_fiducial, int)
        assert num_fiducial > 0
        assert isinstance(img_size, tuple)
        assert isinstance(rectified_img_size, tuple)
        assert isinstance(num_img_channel, int)

        self.heads = 16
        pc_ratio = 1
        ic_ratio = 1
        self.type = "ResNet45"

        self.visual_point = visual_point
        self.num_fiducial = point_size[0]*point_size[1]
        self.img_size = img_size
        self.point_size = point_size
        self.rectified_img_size = rectified_img_size
        self.num_img_channel = num_img_channel
        self.point_channel = num_img_channel * pc_ratio
        self.img_channel = num_img_channel * ic_ratio
        # self.without_as = False
        self.without_as = False
        if self.type == 'pren':
            u_channel = 0.5
        else:
            u_channel = 1
        self.Unet = UNetwork(self.num_img_channel,self.point_size,p_stride, u_channel = u_channel)
        self.scale = self.num_img_channel ** -0.5
        self.p_linear = nn.Sequential(
                            nn.Linear(self.point_channel,32),
                            nn.Linear(32, 64 * 2),
        )
        self.down_feat = ConvModule(self.img_channel*3, self.img_channel, kernel_size=1, stride=1,)
        # # ResNet31:
        if self.type=="ResNet31":
            self.down0 = ConvModule(64,self.img_channel,kernel_size=3, stride=2,padding=1,)
            self.down1 = ConvModule(128,self.img_channel, kernel_size=3,stride=2,padding=1,)
            self.down2 = ConvModule(256, self.img_channel, kernel_size=1,stride=1,)
            self.up_after = ConvModule(self.img_channel, 256, kernel_size=1, stride=1)

        #ResNet45:
        if self.type=='ResNet45':
            self.down0 = ConvModule(32, self.img_channel, kernel_size=3, stride=2, padding=1, )
            self.down1 = ConvModule(32, self.img_channel, kernel_size=1, stride=1 )
            self.down2 = ConvModule(64, self.img_channel, kernel_size=1, stride=1,)


        # svtr: 64,128,128
        if self.type == 'SVTR':
            self.down0 = ConvModule(32, self.img_channel, kernel_size=3, stride=2, padding=1,)
            self.down1 = ConvModule(64, self.img_channel, kernel_size=3, stride=2, padding=1,)
            self.down2 = ConvModule(64, self.img_channel, kernel_size=1, stride=1,)

        # [b, 24, 32, 128], [b, 24, 32, 128], [b, 32, 4, 16]
        if self.type == 'pren':
            self.down0 = ConvModule(24, self.img_channel, kernel_size=3, stride=2, padding=1,)
            self.down1 = ConvModule(24, self.img_channel, kernel_size=3, stride=2, padding=1,)
            self.down2 = ConvModule(32, self.img_channel, kernel_size=1, stride=1,)
            self.down_feat = ConvModule(self.img_channel * 3, self.img_channel//2, kernel_size=1, stride=1, )

        self.feat_linear = nn.Sequential(
            nn.Linear(self.img_channel, 32),
            nn.Linear(32, 64 * 2),
        )

        # self.cls = nn.Sequential(
        #     nn.Linear(self.img_channel, 256),
        #     nn.Linear(256, 37),
        # )

        self.GridGenerator_atten = GridGenerator_Atten(self.rectified_img_size,self.point_size)
        self.count_param(self.Unet, 'Total_PointNet')
        # self.load_ckpt(file_path="ckpt/ztl/reg/mmocr/Backbonev15_4_tps_base_10M_tf_2_unet_124_atten_score_PointNet_mulistage_layer2_training_end_to_end/latest.pth",device='cpu')

    def count_param(self, model, name):
        print("{} have {}M paramerters in total".format(name, sum(x.numel() for x in model.parameters()) / 1e6))

    def atten_score(self, a, b):
        attn = torch.einsum('b m c, b n c -> b m n', a, b)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        # attn = torch.sigmoid(attn)
        attn = torch.tanh(attn)
        # attn = F.softmax(attn, dim=2)
        return attn

    def get_score(self, point, feat, epoch):
        feat = rearrange(feat, 'b c h w -> b (h w) c')
        p1 = self.p_linear(point)
        f = self.feat_linear(feat)
        # cc_score =self.atten_score(p1,p2)
        cc_score = 0.0
        pc_score = self.atten_score(f, p1)
        if self.without_as == True:
            # if epoch <= 5 or self.without_as == True:
            pc_score = torch.zeros_like(pc_score)
        return cc_score, pc_score

    def forward(self, batch_img,epoch = 0, outs=None, point=None):
        """
        Args:
            batch_img (Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Rectified image with size :math:`(N, C, H_r, W_r)`.
        """
        # feat_cat = batch_img
        if outs != None:
            feat0 = self.down0(outs[0])
            feat1 = self.down1(outs[1])
            feat2 = self.down2(batch_img)
            feat_cat = torch.cat((feat0, feat1, feat2),dim=1)
            batch_img = self.down_feat(feat_cat)
        else:
            batch_img = self.down_feat(batch_img)

        # res = batch_img
        logits = self.Unet(
            batch_img)
        batch_C_prime = logits['point']
        u_feat = logits['feature']
        point_feat = logits['point_feat']

        if self.visual_point == True:
            draw_point_map(einops.rearrange(batch_C_prime, 'b (h w) c -> b h w c', h=4, w=16))

        cc_score,pc_score = self.get_score(point_feat,u_feat,epoch)
        build_P_prime = self.GridGenerator_atten.build_P_prime(
            batch_C_prime, cc_score,pc_score,batch_img.device
        )

        build_P_prime_reshape = build_P_prime.reshape([
            build_P_prime.size(0), self.rectified_img_size[0],
            self.rectified_img_size[1], 2
        ])
        if self.visual_point == True:
            draw_point_map(build_P_prime_reshape)

        batch_rectified_img = F.grid_sample(
            batch_img,
            build_P_prime_reshape,
            padding_mode='border',
            align_corners=True)

        # logits = self.cls(point_feat)
        logits = None

        # resnet31
        if self.type =='ResNet31':
            batch_rectified_img = self.up_after(batch_rectified_img)

        result = {
            'output': batch_rectified_img,
            'logits': logits,
            'mp_img': build_P_prime_reshape
        }
        return result

@BACKBONES.register_module()
class U_TPSnet_pos_index(U_TPSnet_v3):
    '''
    new multi-layers
    '''
    def __init__(self,
                 num_fiducial=64,
                 img_size=(32, 128),
                 rectified_img_size=(32, 128),
                 input_channel=64,
                 num_img_channel=64,
                 point_size=(4,16),
                 p_stride=2,
                 visual_point=False,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(img_size, tuple)
        assert isinstance(rectified_img_size, tuple)
        assert isinstance(num_img_channel, int)

        self.heads = 16
        pc_ratio = 1
        ic_ratio = 1
        self.type = "ResNet45"

        self.visual_point = visual_point
        self.num_fiducial = point_size[0]*point_size[1]
        self.img_size = img_size
        self.point_size = point_size
        self.rectified_img_size = rectified_img_size
        self.num_img_channel = num_img_channel
        self.point_channel = num_img_channel * pc_ratio
        self.img_channel = num_img_channel * ic_ratio
        # self.without_as = False
        self.without_as = False
        if self.type == 'pren':
            u_channel = 0.5
        else:
            u_channel = 1
        self.Unet = UNetwork_Warp(self.num_img_channel,self.point_size,p_stride, u_channel = u_channel)
        self.scale = self.num_img_channel ** -0.5
        self.p_linear = nn.Sequential(
                            nn.Linear(self.point_channel,32),
                            nn.Linear(32, 64 * 2),
        )
        self.down_feat = ConvModule(input_channel, self.img_channel, kernel_size=1, stride=1,)
        #ResNet45:
        if self.type=='ResNet45':
            self.down0 = ConvModule(32, self.img_channel, kernel_size=3, stride=2, padding=1, )
            self.down1 = ConvModule(32, self.img_channel, kernel_size=1, stride=1 )
            self.down2 = ConvModule(64, self.img_channel, kernel_size=1, stride=1,)

        self.feat_linear = nn.Sequential(
            nn.Linear(self.img_channel, 32),
            nn.Linear(32, 64 * 2),
        )

        self.GridGenerator_atten = GridGenerator_Atten(self.rectified_img_size,self.point_size)
        self.count_param(self.Unet, 'Total_PointNet')
        # self.load_ckpt(file_path="ckpt/ztl/reg/mmocr/Backbonev15_4_tps_base_10M_tf_2_unet_124_atten_score_PointNet_mulistage_layer2_training_end_to_end/latest.pth",device='cpu')

    def forward(self, batch_img,epoch = 0, outs=None, point=None):
        out = self.down_feat(batch_img)
        logits = self.Unet(out)
        batch_C_prime = logits['point']
        u_feat = logits['feature']
        point_feat = logits['point_feat']

        cc_score,pc_score = self.get_score(point_feat,u_feat,epoch)
        build_P_prime = self.GridGenerator_atten.build_P_prime(
            batch_C_prime, cc_score,pc_score,batch_img.device
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

        # logits = self.cls(point_feat)
        logits = None

        # resnet31
        if self.type =='ResNet31':
            batch_rectified_img = self.up_after(batch_rectified_img)

        result = {
            'output': batch_rectified_img,
            'logits': logits,
            'mp_img': build_P_prime_reshape
        }
        return result

class GridGenerator(nn.Module):
    """Grid Generator of RARE, which produces P_prime by multiplying T with P.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        rectified_img_size (tuple(int, int)):
            Size :math:`(H_r, W_r)` of the rectified image.
    """

    def __init__(self, rectified_img_size,point_size):
        """Generate P_hat and inv_delta_C for later."""
        super().__init__()
        self.eps = 1e-6
        self.point_size = point_size

        self.point_y = point_size[0]
        # if self.point_y == 1 :
        #     self.point_y = self.point_y * 2
        self.point_x = point_size[1]
        self.num_fiducial = self.point_y * self.point_x

        self.rectified_img_height = rectified_img_size[0]
        self.rectified_img_width = rectified_img_size[1]
        # self.num_fiducial = num_fiducial
        self.C = self._build_C(self.num_fiducial)  # num_fiducial x 2
        self.P = self._build_P(self.rectified_img_width,
                               self.rectified_img_height)
        # for multi-gpu, you need register buffer
        self.register_buffer(
            'inv_delta_C',
            torch.tensor(self._build_inv_delta_C(
                self.num_fiducial,
                self.C)).float())  # num_fiducial+3 x num_fiducial+3
        self.register_buffer('P_hat',
                             torch.tensor(
                                 self._build_P_hat(
                                     self.num_fiducial, self.C,
                                     self.P)).float())  # n x num_fiducial+3
        # for fine-tuning with different image width,
        # you may use below instead of self.register_buffer
        # self.inv_delta_C = torch.tensor(
        #     self._build_inv_delta_C(
        #         self.num_fiducial,
        #         self.C)).float().cuda()  # num_fiducial+3 x num_fiducial+3
        # self.P_hat = torch.tensor(
        #     self._build_P_hat(self.num_fiducial, self.C,
        #                       self.P)).float().cuda()  # n x num_fiducial+3

    def _build_C(self,):
        """Return coordinates of fiducial points in rectified_img; C.
        +++++
        -----
        +++++
        """
        ctrl_pts_x = np.linspace(-0.9, 0.9, num=int(self.point_x))
        ctrl_pts_y = np.linspace(-0.9, 0.9, num=int(self.point_y))  # X * Y * 2
        C = np.stack(np.meshgrid(ctrl_pts_x, ctrl_pts_y), axis=2).reshape([-1,2])

        # ctrl_pts_x = np.linspace(-1.0, 1.0, int(num_fiducial / 2))
        # ctrl_pts_y_top = -1 * np.ones(int(num_fiducial / 2))
        # ctrl_pts_y_bottom = np.ones(int(num_fiducial / 2))
        # ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        # ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        # C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # num_fiducial x 2

    def _build_inv_delta_C(self, num_fiducial, C):
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

    def _build_P(self, rectified_img_width, rectified_img_height):
        '''
        meshgrid for P in rectified_img
        '''
        rectified_img_grid_x = (
            np.arange(-rectified_img_width, rectified_img_width, 2) +
            1.0) / rectified_img_width  # self.rectified_img_width
        rectified_img_grid_y = (
            np.arange(-rectified_img_height, rectified_img_height, 2) +
            1.0) / rectified_img_height  # self.rectified_img_height
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
        rbf = np.multiply(np.square(rbf_norm),
                          np.log(rbf_norm + self.eps))  # n x num_fiducial
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x num_fiducial+3

    def build_P_prime(self, batch_C_prime, device='cuda'):
        """Generate Grid from batch_C_prime [batch_size x num_fiducial x 2]"""
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)),
            dim=1)  # batch_size x num_fiducial+3 x 2
        batch_T = torch.bmm(
            batch_inv_delta_C,
            batch_C_prime_with_zeros)  # batch_size x num_fiducial+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2


class GridGenerator_Plus(nn.Module):
    """Grid Generator of RARE, which produces P_prime by multiplying T with P.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        rectified_img_size (tuple(int, int)):
            Size :math:`(H_r, W_r)` of the rectified image.
    """

    def __init__(self, num_img_channel, rectified_img_size,point_size):
        """Generate P_hat and inv_delta_C for later."""
        super().__init__()
        self.eps = 1e-6
        self.point_size = point_size

        self.point_y = point_size[0]
        self.point_x = point_size[1]
        self.input_embed = nn.Linear(num_img_channel if num_img_channel < 64 else 64, 64)
        self.embedding = nn.Linear(2, 64)
        self.down = nn.Linear(64, 2)
        self.num_fiducial = self.point_y * self.point_x

        self.rectified_img_height = rectified_img_size[0]
        self.rectified_img_width = rectified_img_size[1]
        # self.num_fiducial = num_fiducial
        # self.C = self._build_C(self.num_fiducial)  # num_fiducial x 2
        self.P = self._build_P(self.rectified_img_width,
                               self.rectified_img_height)
        self.C = self._build_C_query()
        self.build_C = TFCommonDecoderLayer(d_model=64,
                                            d_inner=64,
                                            n_head=4,
                                            d_k=16,
                                            d_v=16,
                                            ifmask=False,
        )
        # # for multi-gpu, you need register buffer
        # self.register_buffer(
        #     'inv_delta_C',
        #     torch.tensor(self._build_inv_delta_C(
        #         self.num_fiducial,
        #         self.C)).float())  # num_fiducial+3 x num_fiducial+3
        # self.register_buffer('P_hat',
        #                      torch.tensor(
        #                          self._build_P_hat(
        #                              self.num_fiducial, self.C,
        #                              self.P)).float())  # n x num_fiducial+3

    # numpy to torch
    def _build_inv_delta_C(self,num_fiducial, C, device):
        """Return inv_delta_C which is needed to calculate T."""
        B, N, _ = C.size()
        hat_C = torch.zeros((B, num_fiducial, num_fiducial)).to(device)
        for i in range(0, num_fiducial):
            for j in range(i, num_fiducial):
                r = torch.linalg.norm(C[:, i, :] - C[:, j, :])  # sqrt(i-j)^2 [euclidean distance]
                hat_C[:, i, j] = r
                hat_C[:, j, i] = r
        # np.fill_diagonal(hat_C, 1)  #主对角线填充1
        # print("a----"+ str(hat_C))
        # hat_C = hat_C.fill_diagonal_(1.0)
        hat_C = torch.stack([hat_C[t].fill_diagonal_(1.0) for t in torch.arange(B)])
        # print("b----" + str(hat_C))
        hat_C = (hat_C) * torch.log(hat_C)
        # hat_C = (hat_C**2) * torch.log(hat_C)
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
        #     inv_delta_C = torch.linalg.inv(delta_C) #delta_C 取反
        return inv_delta_C  # num_fiducial+3 x num_fiducial+3

    def _build_P(self, rectified_img_width, rectified_img_height):
        '''
        meshgrid for P in rectified_img
        '''
        rectified_img_grid_x = (
            torch.arange(-rectified_img_width, rectified_img_width, 2) +
            1.0) / rectified_img_width  # self.rectified_img_width
        rectified_img_grid_y = (
            torch.arange(-rectified_img_height, rectified_img_height, 2) +
            1.0) / rectified_img_height  # self.rectified_img_height
        P = torch.stack(  # self.rectified_img_w x self.rectified_img_h x 2
            torch.meshgrid(rectified_img_grid_x, rectified_img_grid_y),
            dim=2)
        return P.reshape([
            -1, 2
        ])  # n (= self.rectified_img_width x self.rectified_img_height) x 2

    def _build_P_hat(self,num_fiducial, C, P, device):
        B, n, _ = P.size()  # n (= self.rectified_img_width x self.rectified_img_height)
        P_tile = torch.unsqueeze(P, dim=2)
        P_tile = P_tile.repeat(1, 1, num_fiducial, 1)  # n x 2 -> n x 1 x 2 -> n x num_fiducial x 2

        C_tile = torch.unsqueeze(C, dim=1)  # B * 1 x num_fiducial x 2
        P_diff = P_tile - C_tile  # B * n x num_fiducial x 2
        rbf_norm = torch.linalg.norm(
            P_diff, ord=2, dim=3, keepdim=False)  # B * n x num_fiducial
        rbf = torch.multiply(torch.square(rbf_norm),
                             torch.log(rbf_norm + self.eps))  # n x num_fiducial
        P_hat = torch.cat([torch.ones((B, n, 1)).to(device), P, rbf], dim=2)
        return P_hat  # n x num_fiducial+3

    def _build_C_query(self, ):
        """Return coordinates of fiducial points in rectified_img; C.
        +++++
        -----
        +++++
        """
        ctrl_pts_x = torch.linspace(-1, 1, steps=int(self.point_x))
        ctrl_pts_y = torch.linspace(-1, 1, steps=int(self.point_y))  # X * Y * 2
        C = torch.stack(torch.meshgrid(ctrl_pts_x, ctrl_pts_y), dim=2).reshape([-1,2])
        return C  # num_fiducial x 2


    def forward(self,batch_C_prime, C_feat, device='cuda'):
        batch_size = batch_C_prime.size(0)

        # C & inv_delta_C
        key_value = self.input_embed(C_feat)
        C = self.C.to(device).unsqueeze(0).repeat(batch_size, 1, 1) # B, num_fiducial, 2 -> B,N,E
        C = self.build_C(self.embedding(C), key_value, key_value)
        batch_inv_delta_C = self._build_inv_delta_C(self.num_fiducial, self.down(C),device)
        # batch_inv_delta_C = inv_delta_C.repeat(batch_size, 1, 1)

        # P & P_hat
        P = self.P.to(device).unsqueeze(0).repeat(batch_size, 1, 1)
        batch_P_hat = self._build_P_hat(self.num_fiducial, self.down(C), P,device)
        # batch_P_hat = P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)),
            dim=1)  # batch_size x num_fiducial+3 x 2

        batch_T = torch.bmm(
            batch_inv_delta_C,
            batch_C_prime_with_zeros)  # batch_size x num_fiducial+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2

        return batch_P_prime  # batch_size x n x 2

class GridGenerator_Atten(nn.Module):
    """Grid Generator of RARE, which produces P_prime by multiplying T with P.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        rectified_img_size (tuple(int, int)):
            Size :math:`(H_r, W_r)` of the rectified image.
    """

    def __init__(self, rectified_img_size,point_size):
        """Generate P_hat and inv_delta_C for later."""
        super().__init__()
        self.eps = 1e-6
        self.point_size = point_size

        self.point_y = point_size[0]
        # if self.point_y == 1 :
        #     self.point_y = self.point_y * 2
        self.point_x = point_size[1]
        self.num_fiducial = self.point_y * self.point_x

        self.rectified_img_height = rectified_img_size[0]
        self.rectified_img_width = rectified_img_size[1]
        # self.num_fiducial = num_fiducial
        self.C = self._build_C(self.num_fiducial)  # num_fiducial x 2
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
        # for fine-tuning with different image width,
        # you may use below instead of self.register_buffer
        # self.inv_delta_C = torch.tensor(
        #     self._build_inv_delta_C(
        #         self.num_fiducial,
        #         self.C)).float().cuda()  # num_fiducial+3 x num_fiducial+3
        # self.P_hat = torch.tensor(
        #     self._build_P_hat(self.num_fiducial, self.C,
        #                       self.P)).float().cuda()  # n x num_fiducial+3

    def _build_C(self,num_fiducial):
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

        # ctrl_pts_x = np.linspace(-1.0, 1.0, int(num_fiducial / 2))
        # ctrl_pts_y_top = -1 * np.ones(int(num_fiducial / 2))
        # ctrl_pts_y_bottom = np.ones(int(num_fiducial / 2))
        # ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        # ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        # C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
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
        # rectified_img_grid_x = (
        #     np.arange(-rectified_img_width, rectified_img_width, 2) +
        #     1.0) / rectified_img_width  # self.rectified_img_width
        # rectified_img_grid_y = (
        #     np.arange(-rectified_img_height, rectified_img_height, 2) +
        #     1.0) / rectified_img_height  # self.rectified_img_height

        # rectified_img_grid_x = (
        #     np.arange(0.5, rectified_img_width-0.5, 1)
        #     ) / rectified_img_width  # self.rectified_img_width
        # rectified_img_grid_y = (
        #     np.arange(0.5, rectified_img_height-0.5, 1)
        #     ) / rectified_img_height  # self.rectified_img_height

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
        thela = 0.5
        P_hat = P_hat * (pc_score * thela + 1)
        # P_hat = P_hat * (pc_score + 1)
        # P_hat = P_hat * (pc_score * 2 + self.eps)
        P_hat = torch.cat([torch.ones((B, n, 1)).to(device), P, P_hat], dim=2)

        return P_hat

    def build_P_prime(self, batch_C_prime, cc_score, pc_score,device='cuda'):
        """Generate Grid from batch_C_prime [batch_size x num_fiducial x 2]"""
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.hat_C.to(device).repeat(batch_size, 1, 1)
        # batch_hat_C = self.hat_C.repeat(batch_size, 1, 1)
        # batch_inv_delta_C = self.build_inv_delta_C(batch_hat_C,cc_score,device)

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



class UDAT_Net(nn.Module):

    def __init__(
            self,
            q_size, kv_size, stride, num_img_channel,
            n_heads, n_head_channels, n_groups,offset_range_factor, use_pe,
            attn_drop=0.0, proj_drop=0.0,dwc_pe='False',
            no_off='False', fixed_pe='False'
    ):

        super().__init__()
        # self.off_ratio = ratio
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        # self.Unet = UNetwork(num_img_channel, q_size, stride)
        self.Unet = Unet_wrap(num_img_channel, self.n_group_channels, stride)
        self.fuser = Fuser(num_img_channel)

        self.conv_offset = nn.Sequential(
            CBAM(self.n_group_channels),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
            nn.GELU(),
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        # ref[..., 1].div_(W_key).mul_(2).sub_(1)
        # ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref[..., 1].div_(W_key)
        ref[..., 0].div_(H_key)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        logits = self.Unet(
            x)
        # batch_C_prime = logits['point']
        feat = logits['feature']

        q_off = einops.rearrange(feat, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        # print("DAT_Point: {},{}".format(Hk,Wk))

        # 作为一个系数，这样可以归一化
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        # pos = (offset + reference).tanh()

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, Hk, Wk)
        x_sampled = self.fuser(x,x_sampled)

        # return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)
        return x_sampled


class DAttentionBaseline(nn.Module):

    def __init__(
            self,
            q_size, kv_size, stride,
            n_heads, n_head_channels, n_groups,offset_range_factor, use_pe,
            attn_drop=0.0, proj_drop=0.0,dwc_pe='False',
            no_off='False', fixed_pe='False'
    ):

        super().__init__()
        # self.off_ratio = ratio
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        # if self.q_h == 14 or self.q_w == 14 or self.q_h == 24 or self.q_w == 24:
        #     kk = 5
        # elif self.q_h == 7 or self.q_w == 7 or self.q_h == 12 or self.q_w == 12:
        #     kk = 3
        # elif self.q_h == 28 or self.q_w == 28 or self.q_h == 48 or self.q_w == 48:
        #     kk = 7
        # elif self.q_h == 56 or self.q_w == 56 or self.q_h == 96 or self.q_w == 96:
        #     kk = 9

        # self.conv_offset = nn.Sequential(
        #     nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk // 2, groups=self.n_group_channels),
        #     LayerNormProxy(self.n_group_channels),
        #     nn.GELU(),
        #     nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        # )
        ratio = 1
        if stride == 1:
            kk = 1
        else:
            kk = 3
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels//ratio, 3, 2, 2 // 2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels//ratio, self.n_group_channels, kk, stride, stride // 2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc,
                                           kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off)  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        # print("DAT_Point: {},{}".format(Hk,Wk))

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe:

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

                q_grid = self._get_ref_points(H, W, B, dtype, device)

                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    0.5)

                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)

                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        # return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)
        return y

class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

# class AttentionNetwork(nn.Module):
#     """Localization Network of RARE, which predicts C' (K x 2) from input
#     (img_width x img_height)
#
#     Args:
#         num_fiducial (int): Number of fiducial points of TPS-STN.
#         num_img_channel (int): Number of channels of the input image.
#     """
#
#     def __init__(self, num_img_channel,point_size,p_stride):
#         super().__init__()
#         # give stride & padding
#         # if p_stride == 1:
#         #     p_padding = 1
#         # else:
#         #     p_padding = 1
#         self.conv_feature = ConvModule(num_img_channel,64,3,1,1)
#         self.conv_point = ConvModule(64,64,1,stride=p_stride)
#
#         self.num_img_channel = num_img_channel
#
#         self.Atten = TransDecoderLayer(d_model=64,
#                  d_inner=num_img_channel if num_img_channel < 64 else 64,
#                  n_head=4,
#                  d_k=16,
#                  d_v=16,
#                  ifmask=False,)
#
#         self.point_x = point_size[1]
#         self.point_y = point_size[0]
#         self.num_fiducial = self.point_y * self.point_x
#         self.localization_fc1 = nn.Linear(64*self.num_fiducial, self.num_fiducial*16)
#         self.localization_fc2 = nn.Linear(16*self.num_fiducial, self.num_fiducial*2)
#
#         # Init fc2 in LocalizationNetwork
#         self.localization_fc2.weight.data.fill_(0)
#         ctrl_pts_x = np.linspace(-1.0, 1.0, num=int(self.point_x))
#         ctrl_pts_y = np.linspace(-1.0, 1.0, num=int(self.point_y)) #X * Y * 2
#         initial_bias = np.stack(np.meshgrid(ctrl_pts_x, ctrl_pts_y), axis=2)
#         # initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
#         self.localization_fc2.bias.data = torch.from_numpy(
#             initial_bias).float().view(-1)
#
#     def forward(self, batch_img,p_feature):
#         """
#         Args:
#             batch_img (Tensor): Batch input image of shape
#                 :math:`(N, C, H, W)`.
#
#         Returns:
#             Tensor: Predicted coordinates of fiducial points for input batch.
#             The shape is :math:`(N, F, 2)` where :math:`F` is ``num_fiducial``.
#         """
#         # B,T,C -> B,C,H,W
#
#         # p_feature = p_feature.view(B,C,self.point_y,self.point_x)
#         # stride(1,2)
#         p_feature = self.conv_point(p_feature)
#         B, C, H, W = p_feature.size()
#         # shrink channel
#         i_feature = self.conv_feature(batch_img)
#
#         p_feature = rearrange(p_feature, 'b c h w -> b (h w) c')
#         i_feature = rearrange(i_feature, 'b c h w -> b (h w) c')
#         point_coord = self.Atten(p_feature,i_feature,i_feature)
#
#         p_feature = point_coord.transpose(1, 2).contiguous().reshape(-1, C, H, W)
#         # p_feature= point_coord.view(B,C,H,W)
#         # batch_img = logits['feature']
#         # logits = self.conv(batch_img).view(batch_size, -1)
#         # x*y*2
#         batch_C_prime = self.localization_fc2(self.localization_fc1(
#             point_coord.view(B,-1))).view(B,self.num_fiducial, 2)
#         return {"point": batch_C_prime,"p_feature":p_feature}
#
# class UNetwork(nn.Module):
#     """Localization Network of RARE, which predicts C' (K x 2) from input
#     (img_width x img_height)
#
#     Args:
#         num_fiducial (int): Number of fiducial points of TPS-STN.
#         num_img_channel (int): Number of channels of the input image.
#     """
#
#     def __init__(self, num_img_channel,point_size,p_stride):
#         super().__init__()
#
#         self.num_img_channel = num_img_channel
#         self.point_x = point_size[1]
#         self.point_y = point_size[0]
#         # if point_size[0] == 1:
#         #     tuple_stride = (1,2)
#         #     self.point_y = point_size[0] * 2
#         # else:
#         #     tuple_stride = 2
#         #     self.point_y = point_size[0]
#         # if self.point_x // self.point_y == 2:
#         #     tuple_stride = (1, 2)
#         # else:
#         #     tuple_stride = 2
#         self.conv = Unet(
#                             in_channels=num_img_channel,
#                             num_channels= num_img_channel if num_img_channel<64 else 64,
#                             stride = p_stride
#         )
#         # self.point_y = point_size[0]
#         self.point_x = point_size[1]
#         self.num_fiducial = self.point_y * self.point_x
#
#         self.localization_fc1 = nn.Sequential(
#             nn.Linear(num_img_channel if num_img_channel<64 else 64, 512), nn.ReLU(True))
#         self.localization_fc2 = nn.Linear(512*self.num_fiducial, self.num_fiducial*2)
#
#         # Init fc2 in LocalizationNetwork
#         self.localization_fc2.weight.data.fill_(0)
#         ctrl_pts_x = np.linspace(-1.0, 1.0, num=int(self.point_x))
#         ctrl_pts_y = np.linspace(-1.0, 1.0, num=int(self.point_y))  # X * Y * 2
#         initial_bias = np.stack(np.meshgrid(ctrl_pts_x, ctrl_pts_y), axis=2)
#         # initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
#         self.localization_fc2.bias.data = torch.from_numpy(
#             initial_bias).float().view(-1)
#
#     def forward(self, batch_img):
#         """
#         Args:
#             batch_img (Tensor): Batch input image of shape
#                 :math:`(N, C, H, W)`.
#
#         Returns:
#             Tensor: Predicted coordinates of fiducial points for input batch.
#             The shape is :math:`(N, F, 2)` where :math:`F` is ``num_fiducial``.
#         """
#         batch_size = batch_img.size(0)
#         logits = self.conv(batch_img)
#         point = logits['point']
#         # B,C,H,W = point.size()
#         point = rearrange(point, 'b c h w -> b (h w) c')
#
#         # batch_img = logits['feature']
#         # logits = self.conv(batch_img).view(batch_size, -1)
#         batch_C_prime = self.localization_fc2(
#             self.localization_fc1(point).view(batch_size,-1)).view(batch_size,
#                                                   self.num_fiducial, 2)
#         return {"point": batch_C_prime,"feature": logits['feature']}
#
#
# class LocalizationNetwork(nn.Module):
#     """Localization Network of RARE, which predicts C' (K x 2) from input
#     (img_width x img_height)
#
#     Args:
#         num_fiducial (int): Number of fiducial points of TPS-STN.
#         num_img_channel (int): Number of channels of the input image.
#     """
#
#     def __init__(self, num_fiducial, num_img_channel):
#         super().__init__()
#         self.num_fiducial = num_fiducial
#         self.num_img_channel = num_img_channel
#         self.conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=self.num_img_channel,
#                 out_channels=64,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # batch_size x 64 x img_height/2 x img_width/2
#             nn.Conv2d(64, 128, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # batch_size x 128 x img_h/4 x img_w/4
#             nn.Conv2d(128, 256, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),  # batch_size x 256 x img_h/8 x img_w/8
#             nn.Conv2d(256, 512, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.AdaptiveAvgPool2d(1)  # batch_size x 512
#         )
#
#         self.localization_fc1 = nn.Sequential(
#             nn.Linear(512, 256), nn.ReLU(True))
#         self.localization_fc2 = nn.Linear(256, self.num_fiducial * 2)
#
#         # Init fc2 in LocalizationNetwork
#         self.localization_fc2.weight.data.fill_(0)
#
#
#         ctrl_pts_x = np.linspace(-1.0, 1.0, int(num_fiducial / 2))
#         ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(num_fiducial / 2))
#         ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(num_fiducial / 2))
#         ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
#         ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
#         initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
#         self.localization_fc2.bias.data = torch.from_numpy(
#             initial_bias).float().view(-1)
#
#     def forward(self, batch_img):
#         """
#         Args:
#             batch_img (Tensor): Batch input image of shape
#                 :math:`(N, C, H, W)`.
#
#         Returns:
#             Tensor: Predicted coordinates of fiducial points for input batch.
#             The shape is :math:`(N, F, 2)` where :math:`F` is ``num_fiducial``.
#         """
#         batch_size = batch_img.size(0)
#         features = self.conv(batch_img).view(batch_size, -1)
#         batch_C_prime = self.localization_fc2(
#             self.localization_fc1(features)).view(batch_size,
#                                                   self.num_fiducial, 2)
#         return batch_C_prime
# class Unet(nn.Module):
# # For mini-Unet
#     def __init__(self, in_channels=512,
#                  num_channels=64,
#                  attn_mode='nearest',
#                  stride = (1,2)):
#         super().__init__()
#         self.k_encoder = nn.Sequential(
#             self._encoder_layer(in_channels, num_channels, stride=1),
#             self._encoder_layer(num_channels, num_channels, stride=2),
#             self._encoder_layer(num_channels, num_channels, stride=2),
#             self._encoder_layer(num_channels, num_channels, stride=stride))
#
#         # self.trans = TFCommonDecoderLayer(d_model=64,
#         #          d_inner=64,
#         #          n_head=4,
#         #          d_k=16,
#         #          d_v=16,
#         #          ifmask=False,)
#         self.atten = CBAM(num_channels)
#
#         self.k_decoder = nn.Sequential(
#             self._decoder_layer(
#                 num_channels, num_channels, scale_factor=stride, mode=attn_mode),
#             self._decoder_layer(
#                 num_channels, num_channels, scale_factor=2, mode=attn_mode),
#             self._decoder_layer(
#                 num_channels, num_channels, scale_factor=2, mode=attn_mode),
#             self._decoder_layer(
#                 num_channels,
#                 in_channels,
#                 scale_factor=1,
#                 mode=attn_mode))
#     def _encoder_layer(self,
#                        in_channels,
#                        out_channels,
#                        kernel_size=3,
#                        stride=2,
#                        padding=1):
#         return ConvModule(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,)
#
#     def _decoder_layer(self,
#                        in_channels,
#                        out_channels,
#                        kernel_size=3,
#                        stride=1,
#                        padding=1,
#                        mode='nearest',
#                        scale_factor=None,
#                        size=None):
#         align_corners = None if mode == 'nearest' else True
#         return nn.Sequential(
#             nn.Upsample(
#                 size=size,
#                 scale_factor=scale_factor,
#                 mode=mode,
#                 align_corners=align_corners),
#             ConvModule(
#                 in_channels,
#                 out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding))
#
#     def forward(self,k):
#     # Apply mini U-Net on k
#     #     k = k.transpose(1,2)
#         features = []
#         for i in range(len(self.k_encoder)):
#             k = self.k_encoder[i](k)
#             features.append(k)
#         point = features[-1]
#         B,C,H,W = point.size()
#
#         point = self.atten(point)
#
#         # point = rearrange(point, 'b c h w -> b (h w) c')
#         # point = self.trans(point,point,point,mask=None, ifmask=False)
#         # point = point.transpose(1, 2).contiguous().reshape(-1, C, H, W)
#
#         for i in range(len(self.k_decoder) - 1):
#             k = self.k_decoder[i](k)
#             k = k + features[len(self.k_decoder) - 2 - i]
#         k = self.k_decoder[-1](k)
#         return {'feature':k, 'point':point}
#
# class Fuser(nn.Module):
# # For mini-Unet
#     def __init__(self, num_img_channel=512,
#                  ):
#         super().__init__()
#         self.w_att = ConvModule(2 * num_img_channel, num_img_channel,3, 1, 1)
#
#     def forward(self,l_feature, v_feature):
#         f = torch.cat((l_feature, v_feature), dim=1)
#         f_att = torch.sigmoid(self.w_att(f))
#         output = f_att * v_feature + (1 - f_att) * l_feature
#         return output
#
#
# class ChannelAttentionModule(nn.Module):
#     def __init__(self, channel, ratio=16):
#         super(ChannelAttentionModule, self).__init__()
#         # 使用自适应池化缩减map的大小，保持通道不变
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.shared_MLP = nn.Sequential(
#             nn.Conv2d(channel, channel // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avgout = self.shared_MLP(self.avg_pool(x))
#         maxout = self.shared_MLP(self.max_pool(x))
#         return self.sigmoid(avgout + maxout)
#
#
# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # map尺寸不变，缩减通道
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv2d(out))
#         return out
#
#
# class CBAM(nn.Module):
#     def __init__(self, channel):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttentionModule(channel)
#         self.spatial_attention = SpatialAttentionModule()
#
#     def forward(self, x):
#         out = self.channel_attention(x) * x
#         out = self.spatial_attention(out) * out
#         return out
