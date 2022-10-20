# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import time
from mmocr.models.builder import (DETECTORS, build_backbone, build_convertor,
                                  build_decoder, build_encoder, build_fuser,
                                  build_loss, build_preprocessor)
from .encode_decode_recognizer import EncodeDecodeRecognizer
from mmcv.runner import RUNNERS, EpochBasedRunner

@RUNNERS.register_module()
class RunnerWrapper(EpochBasedRunner):
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.model.module.epoch = self._epoch
        self.model.module.backbone.epoch = self._epoch
        # self.model.module.decoder.epoch = self._epoch
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter = self._iter + 1

        self.call_hook('after_train_epoch')
        self._epoch = self._epoch + 1


@DETECTORS.register_module()
class MixNet(EncodeDecodeRecognizer):
    """Implementation of `Read Like Humans: Autonomous, Bidirectional and
    Iterative LanguageModeling for Scene Text Recognition.

    <https://arxiv.org/pdf/2103.06495.pdf>`_
    """

    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 iter_size=1,
                 fuser=None,
                 loss=None,
                 en_label_convertor=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 max_seq_len=40,
                 pretrained=None,
                 init_cfg=None):
        super(EncodeDecodeRecognizer, self).__init__(init_cfg=init_cfg)
        self.num_classes= 37
        # self.epoch = 5
        # Label convertor (str2tensor, tensor2str)
        if en_label_convertor is not None:
            en_label_convertor.update(max_seq_len=max_seq_len)
            self.en_label_convertor = build_convertor(en_label_convertor)
        else:
            self.en_label_convertor = None

        assert label_convertor is not None
        label_convertor.update(max_seq_len=max_seq_len)
        self.label_convertor = build_convertor(label_convertor)
        self.padding_idx = self.label_convertor.padding_idx

        # Preprocessor module, e.g., TPS
        self.preprocessor = None
        if preprocessor is not None:
            self.preprocessor = build_preprocessor(preprocessor)

        # Backbone
        assert backbone is not None
        self.backbone = build_backbone(backbone)

        # Encoder module
        self.encoder = None
        if encoder is not None:
            encoder.update(num_classes=self.num_classes)
            self.encoder = build_encoder(encoder)

        # Decoder module
        self.decoder = None
        if decoder is not None:
            decoder.update(num_classes=self.num_classes)
            decoder.update(mask_id=self.label_convertor.padding_idx)
            decoder.update(end_id=self.label_convertor.end_idx)
            # decoder.update(padding_idx=self.label_convertor.padding_idx)
            decoder.update(max_seq_len=max_seq_len)

            self.decoder = build_decoder(decoder)

        # Loss
        assert loss is not None
        loss.update(padding_idx = self.padding_idx)
        loss.update(num_classes=self.num_classes)
        self.loss = build_loss(loss)

        # assert de_loss is not None
        # self.de_loss = build_loss(de_loss)
        if tpsnet != None:
            self.count_param(self.tpsnet,"TPSNet model")
        if preprocessor != None:
            self.count_param(self.preprocessor, "Preprocessor model")

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.max_seq_len = max_seq_len

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.iter_size = iter_size

        self.fuser = None
        if fuser is not None:
            self.fuser = build_fuser(fuser)

    def _get_location_mask(self, token_len=None, mask_fill = float('-inf'), dot_v =  1,device = None):
        """Generate location masks given input sequence length.

        Args:
            seq_len (int): The length of input sequence to transformer.
            device (torch.device or str, optional): The device on which the
                masks will be placed.

        Returns:
            Tensor: A mask tensor of shape (seq_len, seq_len) with -infs on
            diagonal and zeros elsewhere.
        """
        mask = torch.eye(token_len,device = device)
        other_mask = (1-mask).float().masked_fill(1-mask == 1, mask_fill)
        mask = mask.float().masked_fill(mask == 1, mask_fill)

        return  torch.stack((mask, other_mask), 0) * dot_v

    def _local_mask(self,targets_dict):
        pass

    def freeze_network(self):
        # self.encoder.eval()
        # for name, parameter in self.encoder.named_parameters():
        #     parameter.requires_grad = False
            # print("{}: {}".format(parameter,parameter.requries_grad))
        self.backbone.freeze_network()

    def forward_train(self, img, img_metas):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'filename', and may also contain
                'ori_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        # self.freeze_network()
        for img_meta in img_metas:
            valid_ratio = 1.0 * img_meta['resize_shape'][1] / img.size(-1)
            img_meta['valid_ratio'] = valid_ratio

        feat = self.extract_feat(img)

        gt_labels = [img_meta['text'] for img_meta in img_metas]

        targets_dict = self.label_convertor.str2tensor(gt_labels)
        if self.en_label_convertor is not None:
            targets_dict_en = self.en_label_convertor.str2tensor(gt_labels)
        else:
            targets_dict_en = None

        # text_logits = None
        # out_enc = None
        if self.encoder is not None:
            enc_logits, feat = self.encoder(feat)


        out_dec = None
        # feat = None
        # print(out_enc.shape)
        if self.decoder is not None:
            out_dec = self.decoder(
                feat,
                enc_logits,
                targets_dict,
                img_metas,
                train_mode=True)
        out_fuser = None
        if self.fuser is not None:
            out_fuser = self.fuser(feat[:,:self.max_seq_len,:], out_dec[-1]['dec_out'])
            # text_logits = out_fuser['logits']
            # out_fusers.append(out_fuser)

        outputs = dict(
            out_enc=enc_logits, out_dec=out_dec, out_fusers=out_fuser)
        # outputs = out_dec

        losses = self.loss(outputs, targets_dict, targets_dict_en, self.epoch,img_metas)

        return losses
        # index = (enc_logits.max(-1)[1]==36)


    def simple_test(self, img, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        for img_meta in img_metas:
            valid_ratio = 1.0 * img_meta['resize_shape'][1] / img.size(-1)
            img_meta['valid_ratio'] = valid_ratio

        if self.preprocessor is not None:
            img = self.preprocessor(img)
        feat = self.extract_feat(img)

        text_logits = None
        enc_logits = None
        if self.encoder is not None:
            enc_logits, feat = self.encoder(feat)
            # text_logits = out_enc

        out_dec = None

        if self.decoder is not None:
            out_dec = self.decoder(
                feat,
                enc_logits,
                img_metas=img_metas, train_mode=False)

        if self.fuser is not None:
            out_fuser = self.fuser(feat[:,:self.max_seq_len,:], out_dec[-1]['dec_out'])
            # text_logits = out_fuser['logits']
            # out_fusers.append(out_fuser)

        # outputs = dict(
        #     out_enc=enc_logits, out_dec=out_dec, out_fusers=out_fuser)

        # if len(out_dec) == 1:
        #     ret = out_dec[-1]['dec_class']
        # else:
        #     ret = out_dec[-1]['dec_class']
        # ret = out_dec[-1]['dec_class']
        # ret = out_fuser['logits']
        ret = enc_logits
        # ret = out_dec[-1]['dec_class']
        # if out_dec.get('out_dec', None) !=None:
        #     ret = out_dec['out_dec'] * 0.5 + out_dec['out_enc'] * 0.5
        # else:
        #     ret = out_dec['out_enc']
        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return ret['logits']

        label_indexes, label_scores = self.label_convertor.tensor2idx(
            ret, img_metas)
        label_strings = self.label_convertor.idx2str(label_indexes)

        # flatten batch results
        results = []
        for string, score in zip(label_strings, label_scores):
            results.append(dict(text=string, score=score))

        return results
