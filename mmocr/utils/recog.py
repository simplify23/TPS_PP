#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import cv2
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from PIL import Image

import einops
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import torch
from mmcv.image.misc import tensor2imgs
from mmcv.runner import load_checkpoint
from mmcv.utils.config import Config

from mmocr.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.models import build_detector
from mmocr.utils.box_util import stitch_boxes_into_lines
from mmocr.utils.fileio import list_from_file
from mmocr.utils.model import revert_sync_batchnorm


# Parse CLI arguments
from tools.data.textrecog.visual_feat import draw_point_map, draw_feature_map

def show_map(heatmap,save_dir = None,name = None,img=None,vis =True,):
    superimposed_img = heatmap
    superimposed_img = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.imshow(img,  alpha=1)
    # plt.show()
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去刻度
    plt.imshow(superimposed_img, cmap='jet', alpha=1)
    if save_dir != None:
        plt.savefig(save_dir, bbox_inches='tight', pad_inches=0)  # 注意两个参数
    if vis == True:
        plt.show()

def draw_feature_map(heatmap,save_dir = None,name = None,img=None,vis = True):
    # 这里的h,w指的是你想要把特征图resize成多大的尺寸
    # img = cv2.resize(img, (32, 128))
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # heatmap = heatmap*0.45+0.22
    heatmap = np.uint8(255 * heatmap)
    # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (128, 32))
    show_map(heatmap,save_dir,name,img,vis)

    # superimposed_img = heatmap
    # superimposed_img = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    #
    # # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # # plt.imshow(img,  alpha=1)
    # # plt.show()
    # plt.axis('off')  # 去坐标轴
    # plt.xticks([])  # 去刻度
    # plt.imshow(superimposed_img,cmap='jet',alpha=1)
    # if save_dir!=None:
    #     plt.savefig(save_dir, bbox_inches='tight', pad_inches=-0.1)  # 注意两个参数
    # else:
    #     plt.show()


def draw_img_map(img,save_dir = None,name = None,vis = True):
    # 这里的h,w指的是你想要把特征图resize成多大的尺寸
    # img = cv2.resize(img, (32, 128))
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img = np.uint8(255 * img)

    show_map(img, save_dir, name, img,vis)
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #
    # plt.axis('off')  # 去坐标轴
    # plt.xticks([])  # 去刻度
    # plt.imshow(img,alpha=1)
    # if save_dir!=None:
    #     plt.savefig(save_dir, bbox_inches='tight', pad_inches=-0.1)  # 注意两个参数
    # else:
    #     plt.show()
    # plt.imshow(superimposed_img,cmap='jet',alpha=1)
    # plt.show()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--img', type=str, help='Input image file or folder path.')
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Output file/folder name for visualization')
    parser.add_argument(
        '--det',
        type=str,
        default='PANet_IC15',
        help='Pretrained text detection algorithm')
    parser.add_argument(
        '--det-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected det model. It '
        'overrides the settings in det')
    parser.add_argument(
        '--det-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected det model. '
        'It overrides the settings in det')
    parser.add_argument(
        '--recog',
        type=str,
        default='SEG',
        help='Pretrained text recognition algorithm')
    parser.add_argument(
        '--recog-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected recog model. It'
        'overrides the settings in recog')
    parser.add_argument(
        '--recog-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected recog model. '
        'It overrides the settings in recog')
    parser.add_argument(
        '--kie',
        type=str,
        default='',
        help='Pretrained key information extraction algorithm')
    parser.add_argument(
        '--kie-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected kie model. It'
        'overrides the settings in kie')
    parser.add_argument(
        '--kie-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected kie model. '
        'It overrides the settings in kie')
    parser.add_argument(
        '--config-dir',
        type=str,
        default=os.path.join(str(Path.cwd()), 'configs/'),
        help='Path to the config directory where all the config files '
        'are located. Defaults to "configs/"')
    parser.add_argument(
        '--batch-mode',
        action='store_true',
        help='Whether use batch mode for inference')
    parser.add_argument(
        '--recog-batch-size',
        type=int,
        default=0,
        help='Batch size for text recognition')
    parser.add_argument(
        '--det-batch-size',
        type=int,
        default=0,
        help='Batch size for text detection')
    parser.add_argument(
        '--single-batch-size',
        type=int,
        default=0,
        help='Batch size for separate det/recog inference')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--export',
        type=str,
        default='',
        help='Folder where the results of each image are exported')
    parser.add_argument(
        '--export-format',
        type=str,
        default='json',
        help='Format of the exported result file(s)')
    parser.add_argument(
        '--details',
        action='store_true',
        help='Whether include the text boxes coordinates and confidence values'
    )
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Prints the recognised text')
    parser.add_argument(
        '--merge', action='store_true', help='Merge neighboring boxes')
    parser.add_argument(
        '--merge-xdist',
        type=float,
        default=20,
        help='The maximum x-axis distance to merge boxes')
    args = parser.parse_args()
    if args.det == 'None':
        args.det = None
    if args.recog == 'None':
        args.recog = None
    # Warnings
    if args.merge and not (args.det and args.recog):
        warnings.warn(
            'Box merging will not work if the script is not'
            ' running in detection + recognition mode.', UserWarning)
    if not os.path.samefile(args.config_dir, os.path.join(str(
            Path.cwd()))) and (args.det_config != ''
                               or args.recog_config != ''):
        warnings.warn(
            'config_dir will be overridden by det-config or recog-config.',
            UserWarning)
    return args


class MMOCR:

    def __init__(self,
                 det='PANet_IC15',
                 det_config='',
                 det_ckpt='',
                 recog='SEG',
                 recog_config='',
                 recog_ckpt='',
                 kie='',
                 kie_config='',
                 kie_ckpt='',
                 hook=True,
                 config_dir=os.path.join(str(Path.cwd()), 'configs/'),
                 device='cuda:0',
                 **kwargs):
        self.hook = hook
        textrecog_models = {
            'CRNN': {
                'config': 'crnn/crnn_academic_dataset.py',
                'ckpt': 'crnn/crnn_academic-a723a1c5.pth'
            },
            'SAR': {
                'config': 'sar/sar_r31_parallel_decoder_academic.py',
                'ckpt': 'sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth'
            },
            'SAR_CN': {
                'config':
                'sar/sar_r31_parallel_decoder_chinese.py',
                'ckpt':
                'sar/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214.pth'
            },
            'NRTR_1/16-1/8': {
                'config': 'nrtr/nrtr_r31_1by16_1by8_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by16_1by8_academic_20211124-f60cebf4.pth'
            },
            'NRTR_1/8-1/4': {
                'config': 'nrtr/nrtr_r31_1by8_1by4_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by8_1by4_academic_20211123-e1fdb322.pth'
            },
            'RobustScanner': {
                'config': 'robust_scanner/robustscanner_r31_academic.py',
                'ckpt': 'robustscanner/robustscanner_r31_academic-5f05874f.pth'
            },
            'SATRN': {
                'config': 'satrn/satrn_academic.py',
                'ckpt': 'satrn/satrn_academic_20211009-cb8b1580.pth'
            },
            'SATRN_sm': {
                'config': 'satrn/satrn_small.py',
                'ckpt': 'satrn/satrn_small_20211009-2cf13355.pth'
            },
            'ABINet': {
                'config': 'abinet/abinet_academic.py',
                'ckpt': 'abinet/abinet_academic-f718abf6.pth'
            },
            'SEG': {
                'config': 'seg/seg_r31_1by16_fpnocr_academic.py',
                'ckpt': 'seg/seg_r31_1by16_fpnocr_academic-72235b11.pth'
            },
            'CRNN_TPS': {
                'config': 'tps/crnn_tps_academic_dataset.py',
                'ckpt': 'tps/crnn_tps_academic_dataset_20210510-d221a905.pth'
            },
            'TPS++': {
                'config': 'tps/trans_tps++.py',
                'ckpt': 'ckpt/ztl/reg/mmocr/Backbone_v17_utps_ctc_loss_samha_without_3_ResNet50/latest.pth',
                # 'ckpt': 'ckpt/ztl/reg/mmocr/Backbonev16_v4_resnet50_re_again_with_utps/epoch_12.pth',
                'ckpt': 'ckpt/ztl/reg/mmocr/Backbone_v17_utps_ce_loss_ResNet50/latest.pth',
            },
            'TPS': {
                'config': 'tps/trans_tps.py',
                'ckpt': 'ckpt/ztl/reg/tps++/Baseline_TPS/latest.pth'
                # 'ckpt': 'ckpt/ztl/reg/mmocr/Backbonev14_1_abstudy_first_layer_tps_training_end_to_end'
            },
            'NRTR_TPS++': {
                'config': 'nrtr/nrtr_tps++.py',
                'ckpt': 'ckpt/ztl/reg/tps++/Baseline_ResNet50_NRTR_tps++/latest.pth'
            },
            'NRTR_TPS': {
                'config': 'nrtr/nrtr_tps.py',
                'ckpt': 'ckpt/ztl/reg/tps++/Baseline_ResNet50_NRTR_tps/epoch_5.pth'
            },
        }

        self.td = det
        self.tr = recog
        self.kie = kie
        self.device = device
        self.grid_kp_list = []
        self.grid_mp_list = []
        self.origin_mp_list = []
        self.img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Check if the det/recog model choice is valid
        if self.tr and self.tr not in textrecog_models:
            raise ValueError(self.tr,
                             'is not a supported text recognition algorithm')

        self.detect_model = None

        self.recog_model = None
        if self.tr:
            # Build recognition model
            if not recog_config:
                recog_config = os.path.join(
                    config_dir, 'textrecog/',
                    textrecog_models[self.tr]['config'])
            if not recog_ckpt:
                # recog_ckpt = 'https://download.openmmlab.com/mmocr/' + \
                #     'textrecog/' + textrecog_models[self.tr]['ckpt']
                recog_ckpt = textrecog_models[self.tr]['ckpt']

            self.recog_model = init_detector(
                recog_config, recog_ckpt, device=self.device)
            self.recog_model = revert_sync_batchnorm(self.recog_model)

        # Attribute check
        for model in list(filter(None, [self.recog_model, self.detect_model])):
            if hasattr(model, 'module'):
                model = model.module
        if self.hook == True:
            self.grid_keypoint()
            self.grid_map()

    def grid_mp_hook(self, module, input, output):
        if self.tr.find("TPS++") != -1:
            for feat in output['output']:
                # feat = einops.rearrange(feat, 'h w c -> (h w) c')
                # B,C,H,W
                heatmap = feat.detach().cpu().numpy()
                heatmap = np.mean(heatmap, axis=0)

                heatmap = np.maximum(heatmap, 0)
                heatmap /= np.max(heatmap)
                self.grid_mp_list.append(heatmap)
            for origin_feat in input[0]:
                # feat = einops.rearrange(feat, 'h w c -> (h w) c')
                # B,C,H,W
                origin_feat = origin_feat.detach().cpu().numpy()
                origin_feat = np.mean(origin_feat, axis=0)

                origin_heatmap = np.maximum(origin_feat, 0)
                origin_heatmap /= np.max(origin_heatmap)
                self.origin_mp_list.append(origin_heatmap)
        else:
            for inp in input[0]:
                self.origin_mp_list.append(inp)
            for outp in output:
                outp = einops.rearrange(outp, 'c h w -> h w c')
                feat = outp.detach().cpu().numpy()
                # self.img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                for i in range(3):
                    feat[:,:,i] = feat[:,:,i] * self.img_norm['std'][i] + self.img_norm['mean'][i]
                # feat = np.mean(feat, axis=0)
                # heatmap = np.maximum(feat, 0)
                # heatmap /= np.max(heatmap)
                self.grid_mp_list.append(feat)
        # self.grid_out.append(output)
        # draw_point_map(output[1])
        # print(output[1].size())

    def grid_kp_hook(self, module, input, output):
        if self.tr.find("TPS++") != -1:
            # kp = einops.rearrange(output['point'], 'b (h w) c -> b h w c', h=4, w=16)
            kp = output['point']
        elif self.tr.find("TPS") != -1:
            kp = output
        # for p in output:
        for p in kp:
            self.grid_kp_list.append(p.detach().cpu().numpy())

        #     self.grid_kp_list.append(p.detach().cpu().numpy())
        # draw_point_map(kp)
        # print(output['point'].size())



    def remove_hook(self):
        if self.grid_kp is not None:
            self.grid_kp.remove()
        if self.grid_map is not None:
            self.grid_map.remove()

    def grid_keypoint(self):
        if self.tr.find("TPS++") != -1:
            self.grid_kp = self.recog_model.tpsnet.Unet.register_forward_hook(self.grid_kp_hook)
        elif self.tr.find("TPS") != -1:
            self.grid_kp = self.recog_model.preprocessor.LocalizationNetwork.register_forward_hook(self.grid_kp_hook)

    def grid_map(self):
        # pass
        if self.tr.find("TPS++") != -1:
            self.grid_map = self.recog_model.tpsnet.register_forward_hook(self.grid_mp_hook)
        elif self.tr.find("TPS") != -1:
            self.grid_map = self.recog_model.preprocessor.register_forward_hook(self.grid_mp_hook)


    def readtext(self,
                 img,
                 output=None,
                 details=False,
                 export=None,
                 export_format='json',
                 batch_mode=False,
                 recog_batch_size=0,
                 det_batch_size=0,
                 single_batch_size=0,
                 imshow=False,
                 print_result=False,
                 merge=False,
                 merge_xdist=20,
                 **kwargs):
        args = locals().copy()
        [args.pop(x, None) for x in ['kwargs', 'self']]
        args = Namespace(**args)

        # Input and output arguments processing
        self._args_processing(args)
        self.args = args

        pp_result = None

        # Send args and models to the MMOCR model inference API
        # and call post-processing functions for the output
        for model in list(
                    filter(None, [self.recog_model, self.detect_model])):
                result = self.single_inference(model, args.arrays,
                                               args.batch_mode,
                                               args.single_batch_size)
                pp_result = self.image_pp(result, model)
        self.remove_hook()

        return pp_result

    def image_pp(self, result, model):
        i = 0
        show = True
        for arr,img, output, export, res, kpoint, featmp, originmp in zip(self.args.arrays, self.args.filenames, self.args.output,
                                            self.args.export, result, self.grid_kp_list, self.grid_mp_list, self.origin_mp_list):
            img_path = self.args.img+"/"+img+".jpg"
            # img = Image.open(img_path, mode='r')
            # crop_img = img.resize([128,32])
            img = cv2.imread(img_path)  # 读取图片
            img= cv2.resize(img, (128, 32))
            f = 10
            i = i+1
            img_cv = cv2.copyMakeBorder(img, f, f, f, f, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            for p in kpoint:
                if self.tr.find("TPS++") != -1:
                    x = int((p[0] + 0.9) * 128 / 1.8) + f
                    y = int((p[1] + 0.9) * 32 / 1.8) + f
                # x = int((p[0]) * 128)
                # y = int((p[1]) * 32 )
                elif self.tr.find("TPS") != -1:
                    x = int((p[0] + 1.0) * 128 / 2)+ f
                    y = int((p[1] + 1.0) * 32 / 2) + f
                # print(x, y)
                cv2.circle(img_cv, (x, y), 1, (0, 255, 120), 1)
            #     BGR

            show_map(img_cv, "cute_vis/{}_point_{}.jpg".format(i,self.tr),show)
            # img_kp = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            #
            # plt.imshow(img_kp, alpha=1)
            # plt.show()


            if self.tr.find("TPS++") != -1:
                draw_feature_map(originmp,"cute_vis/{}_mp_0_{}.jpg".format(i,self.tr),show)
                draw_feature_map(featmp,"cute_vis/{}_mp_1_{}.jpg".format(i,self.tr),show)
            elif self.tr.find("TPS") != -1:
                draw_img_map(featmp,"cute_vis/{}_mp_2_{}.jpg".format(i,self.tr),show)

            print("{}jpg finish!\n".format(i))

            # if export:
            #     mmcv.dump(res, export, indent=4)
            # if output or self.args.imshow:
            #     res_img = model.show_result(arr, res, out_file=output)
            #
            #     if self.args.imshow:
            #         mmcv.imshow(res_img, 'inference results')
            # if self.args.print_result:
            #     print(res, end='\n\n')
        return result

    # Post processing function for separate det/recog inference
    def single_pp(self, result, model):
        for arr, output, export, res in zip(self.args.arrays, self.args.output,
                                            self.args.export, result):
            if export:
                mmcv.dump(res, export, indent=4)
            if output or self.args.imshow:
                res_img = model.show_result(arr, res, out_file=output)

                if self.args.imshow:
                    mmcv.imshow(res_img, 'inference results')
            if self.args.print_result:
                print(res, end='\n\n')
        return result


    # Separate det/recog inference pipeline
    def single_inference(self, model, arrays, batch_mode, batch_size=0):
        result = []
        if batch_mode:
            if batch_size == 0:
                result = model_inference(model, arrays, batch_mode=True)
            else:
                n = batch_size
                arr_chunks = [
                    arrays[i:i + n] for i in range(0, len(arrays), n)
                ]
                for chunk in arr_chunks:
                    result.extend(
                        model_inference(model, chunk, batch_mode=True))
        else:
            for arr in arrays:
                result.append(model_inference(model, arr, batch_mode=False))
        return result

    # Arguments pre-processing function
    def _args_processing(self, args):
        # Check if the input is a list/tuple that
        # contains only np arrays or strings
        if isinstance(args.img, (list, tuple)):
            img_list = args.img
            if not all([isinstance(x, (np.ndarray, str)) for x in args.img]):
                raise AssertionError('Images must be strings or numpy arrays')

        # Create a list of the images
        if isinstance(args.img, str):
            img_path = Path(args.img)
            if img_path.is_dir():
                img_list = [str(x) for x in img_path.glob('*')]
            else:
                img_list = [str(img_path)]
        elif isinstance(args.img, np.ndarray):
            img_list = [args.img]

        # Read all image(s) in advance to reduce wasted time
        # re-reading the images for visualization output
        args.arrays = [mmcv.imread(x) for x in img_list]

        # Create a list of filenames (used for output images and result files)
        if isinstance(img_list[0], str):
            args.filenames = [str(Path(x).stem) for x in img_list]
        else:
            args.filenames = [str(x) for x in range(len(img_list))]

        # If given an output argument, create a list of output image filenames
        num_res = len(img_list)
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir():
                args.output = [
                    str(output_path / f'out_{x}.png') for x in args.filenames
                ]
            else:
                args.output = [str(args.output)]
                if args.batch_mode:
                    raise AssertionError('Output of multiple images inference'
                                         ' must be a directory')
        else:
            args.output = [None] * num_res

        # If given an export argument, create a list of
        # result filenames for each image
        if args.export:
            export_path = Path(args.export)
            args.export = [
                str(export_path / f'out_{x}.{args.export_format}')
                for x in args.filenames
            ]
        else:
            args.export = [None] * num_res

        return args


# Create an inference pipeline with parsed arguments
def main():
    args = parse_args()
    ocr = MMOCR(**vars(args))
    ocr.readtext(**vars(args))


if __name__ == '__main__':
    main()
