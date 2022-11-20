# Changelog

## v0.4.0 (15/12/2021)

### Highlights

1. We release a new text recognition model - [ABINet](https://arxiv.org/pdf/2103.06495.pdf) (CVPR 2021, Oral). With it dedicated model design and useful data augmentation transforms, ABINet can achieve the best performance on irregular text recognition tasks. [Check it out!](https://mmocr.readthedocs.io/en/latest/textrecog_models.html#read-like-humans-autonomous-bidirectional-and-iterative-language-modeling-for-scene-text-recognition)
2. We are also working hard to fulfill the requests from our community.
[OpenSet KIE](https://mmocr.readthedocs.io/en/latest/kie_models.html#wildreceiptopenset) is one of the achievement, which extends the application of SDMGR from text node classification to node-pair relation extraction. We also provide
a demo script to convert WildReceipt to open set domain, though it cannot
take the full advantage of OpenSet format. For more information, please read our
[tutorial](https://mmocr.readthedocs.io/en/latest/tutorials/kie_closeset_openset.html).
3. APIs of models can be exposed through TorchServe. [Docs](https://mmocr.readthedocs.io/en/latest/model_serving.html)

### Breaking Changes & Migration Guide

#### Postprocessor

Some refactoring processes are still going on. For all text detection models, we unified their `decode` implementations into a new module category, `POSTPROCESSOR`, which is responsible for decoding different raw outputs into boundary instances. In all text detection configs, the `text_repr_type` argument in `bbox_head` is deprecated and will be removed in the future release.

**Migration Guide**: Find a similar line from detection model's config:
```
text_repr_type=xxx,
```
And replace it with
```
postprocessor=dict(type='{MODEL_NAME}Postprocessor', text_repr_type=xxx)),
```
Take a snippet of PANet's config as an example. Before the change, its config for `bbox_head` looks like:
```
    bbox_head=dict(
        type='PANHead',
        text_repr_type='poly',
        in_channels=[128, 128, 128, 128],
        out_channels=6,
        loss=dict(type='PANLoss')),
```
Afterwards:
```
    bbox_head=dict(
    type='PANHead',
    in_channels=[128, 128, 128, 128],
    out_channels=6,
    loss=dict(type='PANLoss'),
    postprocessor=dict(type='PANPostprocessor', text_repr_type='poly')),
```
There are other postprocessors and each takes different arguments. Interested users can find their interfaces or implementations in `mmocr/models/textdet/postprocess` or through our [api docs](https://mmocr.readthedocs.io/en/latest/api.html#textdet-postprocess).

#### New Config Structure

We reorganized the `configs/` directory by extracting reusable sections into `configs/_base_`. Now the directory tree of `configs/_base_` is organized as follows:

```
_base_
├── det_datasets
├── det_models
├── det_pipelines
├── recog_datasets
├── recog_models
├── recog_pipelines
└── schedules
```

Most of model configs are making full use of base configs now, which makes the overall structural clearer and facilitates fair
comparison across models. Despite the seemingly significant hierarchical difference, **these changes would not break the backward compatibility** as the names of model configs remain the same.

### New Features
* Support openset kie by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/498
* Add converter for the Open Images v5 text annotations by Krylov et al. by @baudm in https://github.com/open-mmlab/mmocr/pull/497
* Support Chinese for kie show result by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/464
* Add TorchServe support for text detection and recognition by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/522
* Save filename in text detection test results by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/570
* Add codespell pre-commit hook and fix typos by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/520
* Avoid duplicate placeholder docs in CN by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/582
* Save results to json file for kie. by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/589
* Add SAR_CN to ocr.py by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/579
* mim extension for windows by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/641
* Support muitiple pipelines for different datasets by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/657
* ABINet Framework by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/651

### Refactoring
* Refactor textrecog config structure by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/617
* Refactor text detection config by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/626
* refactor transformer modules by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/618
* refactor textdet postprocess by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/640

### Docs
* C++ example section by @apiaccess21 in https://github.com/open-mmlab/mmocr/pull/593
* install.md Chinese section by @A465539338 in https://github.com/open-mmlab/mmocr/pull/364
* Add Chinese Translation of deployment.md. by @fatfishZhao in https://github.com/open-mmlab/mmocr/pull/506
* Fix a model link and add the metafile for SATRN by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/473
* Improve docs style by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/474
* Enhancement & sync Chinese docs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/492
* TorchServe docs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/539
* Update docs menu by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/564
* Docs for KIE CloseSet & OpenSet by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/573
* Fix broken links by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/576
* Docstring for text recognition models by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/562
* Add MMFlow & MIM by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/597
* Add MMFewShot by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/621
* Update model readme by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/604
* Add input size check to model_inference by @mpena-vina in https://github.com/open-mmlab/mmocr/pull/633
* Docstring for textdet models by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/561
* Add MMHuman3D in readme by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/644
* Use shared menu from theme instead by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/655
* Refactor docs structure by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/662
* Docs fix by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/664

### Enhancements
* Use bounding box around polygon instead of within polygon by @alexander-soare in https://github.com/open-mmlab/mmocr/pull/469
* Add CITATION.cff by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/476
* Add py3.9 CI by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/475
* update model-index.yml by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/484
* Use container in CI by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/502
* CircleCI Setup by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/611
* Remove unnecessary custom_import from train.py by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/603
* Change the upper version of mmcv to 1.5.0 by @zhouzaida in https://github.com/open-mmlab/mmocr/pull/628
* Update CircleCI by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/631
* Pass custom_hooks to MMCV by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/609
* Skip CI when some specific files were changed by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/642
* Add markdown linter in pre-commit hook by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/643
* Use shape from loaded image by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/652
* Cancel previous runs that are not completed by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/666

### Bug Fixes
* Modify algorithm "sar" weights path in metafile by @ShoupingShan in https://github.com/open-mmlab/mmocr/pull/581
* Fix Cuda CI by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/472
* Fix image export in test.py for KIE models by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/486
* Allow invalid polygons in intersection and union by default  by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/471
* Update checkpoints' links for SATRN by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/518
* Fix converting to onnx bug because of changing key from img_shape to resize_shape by @Harold-lkk in https://github.com/open-mmlab/mmocr/pull/523
* Fix PyTorch 1.6 incompatible checkpoints by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/540
* Fix paper field in metafiles by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/550
* Unify recognition task names in metafiles by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/548
* Fix py3.9 CI by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/563
* Always map location to cpu when loading checkpoint by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/567
* Fix wrong model builder in recog_test_imgs by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/574
* Improve dbnet r50 by fixing img std by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/578
* Fix resource warning: unclosed file by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/577
* Fix bug that same start_point for different texts in draw_texts_by_pil by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/587
* Keep original texts for kie by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/588
* Fix random seed by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/600
* Fix DBNet_r50 config by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/625
* Change SBC case to DBC case by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/632
* Fix kie demo by @innerlee in https://github.com/open-mmlab/mmocr/pull/610
* fix type check by @cuhk-hbsun in https://github.com/open-mmlab/mmocr/pull/650
* Remove depreciated image validator in totaltext converter by @gaotongxiao in https://github.com/open-mmlab/mmocr/pull/661
* Fix change locals() dict by @Fei-Wang in https://github.com/open-mmlab/mmocr/pull/663
* fix #614: textsnake targets by @HolyCrap96 in https://github.com/open-mmlab/mmocr/pull/660

## New Contributors
* @alexander-soare made their first contribution in https://github.com/open-mmlab/mmocr/pull/469
* @A465539338 made their first contribution in https://github.com/open-mmlab/mmocr/pull/364
* @fatfishZhao made their first contribution in https://github.com/open-mmlab/mmocr/pull/506
* @baudm made their first contribution in https://github.com/open-mmlab/mmocr/pull/497
* @ShoupingShan made their first contribution in https://github.com/open-mmlab/mmocr/pull/581
* @apiaccess21 made their first contribution in https://github.com/open-mmlab/mmocr/pull/593
* @zhouzaida made their first contribution in https://github.com/open-mmlab/mmocr/pull/628
* @mpena-vina made their first contribution in https://github.com/open-mmlab/mmocr/pull/633
* @Fei-Wang made their first contribution in https://github.com/open-mmlab/mmocr/pull/663

**Full Changelog**: https://github.com/open-mmlab/mmocr/compare/v0.3.0...0.4.0

## v0.3.0 (25/8/2021)

### Highlights
1. We add a new text recognition model -- SATRN! Its pretrained checkpoint achieves the best performance over other provided text recognition models. A lighter version of SATRN is also released which can obtain ~98% of the performance of the original model with only 45 MB in size. ([@2793145003](https://github.com/2793145003)) [#405](https://github.com/open-mmlab/mmocr/pull/405)
2. Improve the demo script, `ocr.py`, which supports applying end-to-end text detection, text recognition and key information extraction models on images with easy-to-use commands. Users can find its full documentation in the demo section. ([@samayala22](https://github.com/samayala22), [@manjrekarom](https://github.com/manjrekarom)) [#371](https://github.com/open-mmlab/mmocr/pull/371), [#386](https://github.com/open-mmlab/mmocr/pull/386), [#400](https://github.com/open-mmlab/mmocr/pull/400), [#374](https://github.com/open-mmlab/mmocr/pull/374), [#428](https://github.com/open-mmlab/mmocr/pull/428)
3. Our documentation is reorganized into a clearer structure. More useful contents are on the way! [#409](https://github.com/open-mmlab/mmocr/pull/409), [#454](https://github.com/open-mmlab/mmocr/pull/454)
4. The requirement of `Polygon3` is removed since this project is no longer maintained or distributed. We unified all its references to equivalent substitutions in `shapely` instead. [#448](https://github.com/open-mmlab/mmocr/pull/448)

### Breaking Changes & Migration Guide
1. Upgrade version requirement of MMDetection to 2.14.0 to avoid bugs [#382](https://github.com/open-mmlab/mmocr/pull/382)
2. MMOCR now has its own model and layer registries inherited from MMDetection's or MMCV's counterparts. ([#436](https://github.com/open-mmlab/mmocr/pull/436)) The modified hierarchical structure of the model registries are now organized as follows.

```text
mmcv.MODELS -> mmdet.BACKBONES -> BACKBONES
mmcv.MODELS -> mmdet.NECKS -> NECKS
mmcv.MODELS -> mmdet.ROI_EXTRACTORS -> ROI_EXTRACTORS
mmcv.MODELS -> mmdet.HEADS -> HEADS
mmcv.MODELS -> mmdet.LOSSES -> LOSSES
mmcv.MODELS -> mmdet.DETECTORS -> DETECTORS
mmcv.ACTIVATION_LAYERS -> ACTIVATION_LAYERS
mmcv.UPSAMPLE_LAYERS -> UPSAMPLE_LAYERS
```

To migrate your old implementation to our new backend, you need to change the import path of any registries and their corresponding builder functions (including `build_detectors`) from `mmdet.models.builder` to `mmocr.models.builder`. If you have referred to any model or layer of MMDetection or MMCV in your model config, you need to add `mmdet.` or `mmcv.` prefix to its name to inform the model builder of the right namespace to work on.

Interested users may check out [MMCV's tutorial on Registry](https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html) for in-depth explanations on its mechanism.


### New Features
- Automatically replace SyncBN with BN for inference [#420](https://github.com/open-mmlab/mmocr/pull/420), [#453](https://github.com/open-mmlab/mmocr/pull/453)
- Support batch inference for CRNN and SegOCR [#407](https://github.com/open-mmlab/mmocr/pull/407)
- Support exporting documentation in pdf or epub format [#406](https://github.com/open-mmlab/mmocr/pull/406)
- Support `persistent_workers` option in data loader [#459](https://github.com/open-mmlab/mmocr/pull/459)

### Bug Fixes
- Remove depreciated key in kie_test_imgs.py [#381](https://github.com/open-mmlab/mmocr/pull/381)
- Fix dimension mismatch in batch testing/inference of DBNet [#383](https://github.com/open-mmlab/mmocr/pull/383)
- Fix the problem of dice loss which stays at 1 with an empty target given [#408](https://github.com/open-mmlab/mmocr/pull/408)
- Fix a wrong link in ocr.py ([@naarkhoo](https://github.com/naarkhoo)) [#417](https://github.com/open-mmlab/mmocr/pull/417)
- Fix undesired assignment to "pretrained" in test.py [#418](https://github.com/open-mmlab/mmocr/pull/418)
- Fix a problem in polygon generation of DBNet [#421](https://github.com/open-mmlab/mmocr/pull/421), [#443](https://github.com/open-mmlab/mmocr/pull/443)
- Skip invalid annotations in totaltext_converter [#438](https://github.com/open-mmlab/mmocr/pull/438)
- Add zero division handler in poly utils, remove Polygon3 [#448](https://github.com/open-mmlab/mmocr/pull/448)

### Improvements
- Replace lanms-proper with lanms-neo to support installation on Windows (with special thanks to [@gen-ko](https://github.com/gen-ko) who has re-distributed this package!)
- Support MIM [#394](https://github.com/open-mmlab/mmocr/pull/394)
- Add tests for PyTorch 1.9 in CI [#401](https://github.com/open-mmlab/mmocr/pull/401)
- Enables fullscreen layout in readthedocs [#413](https://github.com/open-mmlab/mmocr/pull/413)
- General documentation enhancement [#395](https://github.com/open-mmlab/mmocr/pull/395)
- Update version checker [#427](https://github.com/open-mmlab/mmocr/pull/427)
- Add copyright info [#439](https://github.com/open-mmlab/mmocr/pull/439)
- Update citation information [#440](https://github.com/open-mmlab/mmocr/pull/440)

### Contributors

We thank [@2793145003](https://github.com/2793145003), [@samayala22](https://github.com/samayala22), [@manjrekarom](https://github.com/manjrekarom), [@naarkhoo](https://github.com/naarkhoo), [@gen-ko](https://github.com/gen-ko), [@duanjiaqi](https://github.com/duanjiaqi), [@gaotongxiao](https://github.com/gaotongxiao), [@cuhk-hbsun](https://github.com/cuhk-hbsun), [@innerlee](https://github.com/innerlee), [@wdsd641417025](https://github.com/wdsd641417025) for their contribution to this release!

## v0.2.1 (20/7/2021)

### Highlights
1. Upgrade to use MMCV-full **>= 1.3.8** and MMDetection **>= 2.13.0** for latest features
2. Add ONNX and TensorRT export tool, supporting the deployment of DBNet, PSENet, PANet and CRNN (experimental) [#278](https://github.com/open-mmlab/mmocr/pull/278), [#291](https://github.com/open-mmlab/mmocr/pull/291), [#300](https://github.com/open-mmlab/mmocr/pull/300), [#328](https://github.com/open-mmlab/mmocr/pull/328)
3. Unified parameter initialization method which uses init_cfg in config files [#365](https://github.com/open-mmlab/mmocr/pull/365)

### New Features
- Support TextOCR dataset [#293](https://github.com/open-mmlab/mmocr/pull/293)
- Support Total-Text dataset [#266](https://github.com/open-mmlab/mmocr/pull/266), [#273](https://github.com/open-mmlab/mmocr/pull/273), [#357](https://github.com/open-mmlab/mmocr/pull/357)
- Support grouping text detection box into lines [#290](https://github.com/open-mmlab/mmocr/pull/290), [#304](https://github.com/open-mmlab/mmocr/pull/304)
- Add benchmark_processing script that benchmarks data loading process [#261](https://github.com/open-mmlab/mmocr/pull/261)
- Add SynthText preprocessor for text recognition models [#351](https://github.com/open-mmlab/mmocr/pull/351), [#361](https://github.com/open-mmlab/mmocr/pull/361)
- Support batch inference during testing [#310](https://github.com/open-mmlab/mmocr/pull/310)
- Add user-friendly OCR inference script [#366](https://github.com/open-mmlab/mmocr/pull/366)

### Bug Fixes

- Fix improper class ignorance in SDMGR Loss [#221](https://github.com/open-mmlab/mmocr/pull/221)
- Fix potential numerical zero division error in DRRG [#224](https://github.com/open-mmlab/mmocr/pull/224)
- Fix installing requirements with pip and mim [#242](https://github.com/open-mmlab/mmocr/pull/242)
- Fix dynamic input error of DBNet [#269](https://github.com/open-mmlab/mmocr/pull/269)
- Fix space parsing error in LineStrParser [#285](https://github.com/open-mmlab/mmocr/pull/285)
- Fix textsnake decode error [#264](https://github.com/open-mmlab/mmocr/pull/264)
- Correct isort setup [#288](https://github.com/open-mmlab/mmocr/pull/288)
- Fix a bug in SDMGR config [#316](https://github.com/open-mmlab/mmocr/pull/316)
- Fix kie_test_img for KIE nonvisual [#319](https://github.com/open-mmlab/mmocr/pull/319)
- Fix metafiles [#342](https://github.com/open-mmlab/mmocr/pull/342)
- Fix different device problem in FCENet [#334](https://github.com/open-mmlab/mmocr/pull/334)
- Ignore improper tailing empty characters in annotation files [#358](https://github.com/open-mmlab/mmocr/pull/358)
- Docs fixes [#247](https://github.com/open-mmlab/mmocr/pull/247), [#255](https://github.com/open-mmlab/mmocr/pull/255), [#265](https://github.com/open-mmlab/mmocr/pull/265), [#267](https://github.com/open-mmlab/mmocr/pull/267), [#268](https://github.com/open-mmlab/mmocr/pull/268), [#270](https://github.com/open-mmlab/mmocr/pull/270), [#276](https://github.com/open-mmlab/mmocr/pull/276), [#287](https://github.com/open-mmlab/mmocr/pull/287), [#330](https://github.com/open-mmlab/mmocr/pull/330), [#355](https://github.com/open-mmlab/mmocr/pull/355), [#367](https://github.com/open-mmlab/mmocr/pull/367)
- Fix NRTR config [#356](https://github.com/open-mmlab/mmocr/pull/356), [#370](https://github.com/open-mmlab/mmocr/pull/370)

### Improvements
- Add backend for resizeocr [#244](https://github.com/open-mmlab/mmocr/pull/244)
- Skip image processing pipelines in SDMGR novisual [#260](https://github.com/open-mmlab/mmocr/pull/260)
- Speedup DBNet [#263](https://github.com/open-mmlab/mmocr/pull/263)
- Update mmcv installation method in workflow [#323](https://github.com/open-mmlab/mmocr/pull/323)
- Add part of Chinese documentations [#353](https://github.com/open-mmlab/mmocr/pull/353), [#362](https://github.com/open-mmlab/mmocr/pull/362)
- Add support for ConcatDataset with two workflows [#348](https://github.com/open-mmlab/mmocr/pull/348)
- Add list_from_file and list_to_file utils [#226](https://github.com/open-mmlab/mmocr/pull/226)
- Speed up sort_vertex [#239](https://github.com/open-mmlab/mmocr/pull/239)
- Support distributed evaluation of KIE [#234](https://github.com/open-mmlab/mmocr/pull/234)
- Add pretrained FCENet on IC15 [#258](https://github.com/open-mmlab/mmocr/pull/258)
- Support CPU for OCR demo [#227](https://github.com/open-mmlab/mmocr/pull/227)
- Avoid extra image pre-processing steps [#375](https://github.com/open-mmlab/mmocr/pull/375)


## v0.2.0 (18/5/2021)

### Highlights

1. Add the NER approach Bert-softmax (NAACL'2019)
2. Add the text detection method DRRG (CVPR'2020)
3. Add the text detection method FCENet (CVPR'2021)
4. Increase the ease of use via adding text detection and recognition end-to-end demo, and colab online demo.
5. Simplify the installation.

### New Features

- Add Bert-softmax for Ner task [#148](https://github.com/open-mmlab/mmocr/pull/148)
- Add DRRG [#189](https://github.com/open-mmlab/mmocr/pull/189)
- Add FCENet [#133](https://github.com/open-mmlab/mmocr/pull/133)
- Add end-to-end demo [#105](https://github.com/open-mmlab/mmocr/pull/105)
- Support batch inference [#86](https://github.com/open-mmlab/mmocr/pull/86) [#87](https://github.com/open-mmlab/mmocr/pull/87) [#178](https://github.com/open-mmlab/mmocr/pull/178)
- Add TPS preprocessor for text recognition [#117](https://github.com/open-mmlab/mmocr/pull/117) [#135](https://github.com/open-mmlab/mmocr/pull/135)
- Add demo documentation [#151](https://github.com/open-mmlab/mmocr/pull/151) [#166](https://github.com/open-mmlab/mmocr/pull/166) [#168](https://github.com/open-mmlab/mmocr/pull/168) [#170](https://github.com/open-mmlab/mmocr/pull/170) [#171](https://github.com/open-mmlab/mmocr/pull/171)
- Add checkpoint for Chinese recognition [#156](https://github.com/open-mmlab/mmocr/pull/156)
- Add metafile [#175](https://github.com/open-mmlab/mmocr/pull/175) [#176](https://github.com/open-mmlab/mmocr/pull/176) [#177](https://github.com/open-mmlab/mmocr/pull/177) [#182](https://github.com/open-mmlab/mmocr/pull/182) [#183](https://github.com/open-mmlab/mmocr/pull/183)
- Add support for numpy array inference [#74](https://github.com/open-mmlab/mmocr/pull/74)

### Bug Fixes

- Fix the duplicated point bug due to transform for textsnake [#130](https://github.com/open-mmlab/mmocr/pull/130)
- Fix CTC loss NaN [#159](https://github.com/open-mmlab/mmocr/pull/159)
- Fix error raised if result is empty in demo [#144](https://github.com/open-mmlab/mmocr/pull/141)
- Fix results missing if one image has a large number of boxes [#98](https://github.com/open-mmlab/mmocr/pull/98)
- Fix package missing in dockerfile [#109](https://github.com/open-mmlab/mmocr/pull/109)

### Improvements

- Simplify installation procedure via removing compiling [#188](https://github.com/open-mmlab/mmocr/pull/188)
- Speed up panet post processing so that it can detect dense texts [#188](https://github.com/open-mmlab/mmocr/pull/188)
- Add zh-CN README [#70](https://github.com/open-mmlab/mmocr/pull/70) [#95](https://github.com/open-mmlab/mmocr/pull/95)
- Support windows [#89](https://github.com/open-mmlab/mmocr/pull/89)
- Add Colab [#147](https://github.com/open-mmlab/mmocr/pull/147) [#199](https://github.com/open-mmlab/mmocr/pull/199)
- Add 1-step installation using conda environment [#193](https://github.com/open-mmlab/mmocr/pull/193) [#194](https://github.com/open-mmlab/mmocr/pull/194) [#195](https://github.com/open-mmlab/mmocr/pull/195)


## v0.1.0 (7/4/2021)

### Highlights

- MMOCR is released.

### Main Features

- Support text detection, text recognition and the corresponding downstream tasks such as key information extraction.
- For text detection, support both single-step (`PSENet`, `PANet`, `DBNet`, `TextSnake`) and two-step (`MaskRCNN`) methods.
- For text recognition, support CTC-loss based method `CRNN`; Encoder-decoder (with attention) based methods `SAR`, `Robustscanner`; Segmentation based method `SegOCR`; Transformer based method `NRTR`.
- For key information extraction, support GCN based method `SDMG-R`.
- Provide checkpoints and log files for all of the methods above.
