# TPS++: Attention-Enhanced Thin-Plate Spline for Scene Text Recognition

The official code of TPS_PP.

The main branch works with **PyTorch 1.6+**.

Documentation: https://mmocr.readthedocs.io/en/latest/.

## To Do List
* [x] CRNN + TPS_PP
* [x] NRTR + TPS_PP
* [ ] ABINet-LV + TPS


## Installation

Please refer to [Install Guide](https://github.com/simplify23/TPS_PP/blob/main/docs/en/install.md).

## Get Started

Please see [Getting Started](https://github.com/simplify23/TPS_PP/blob/main/docs/en/getting_started.md) for the basic usage of MMOCR 0.4.0.

## Datasets
The specific configuration of the dataset for training and testing can be found here [Dataset Document](https://github.com/simplify23/TPS_PP/blob/main/docs/en/datasets/recog.md)
```
testing 
├── mixture
│   ├── icdar_2013
│   ├── icdar_2015
│   ├── III5K
│   ├── ct80
│   ├── svt
│   ├── svtp

training
├── mixture
│   ├── Syn90k
│   │   ├── shuffle_labels.txt
│   │   ├── label.txt
│   │   ├── label.lmdb
│   │   ├── mnt
│   ├── SynthText
│   │   ├── alphanumeric_labels.txt
│   │   ├── shuffle_labels.txt
│   │   ├── instances_train.txt
│   │   ├── label.txt
│   │   ├── label.lmdb
│   │   ├── synthtext
```


## Pretrained Models

Get the pretrained models from [BaiduNetdisk(passwd:d6jd)](https://pan.baidu.com/s/1s0oNmd5jQJCvoH1efjfBdg), [GoogleDrive](https://drive.google.com/drive/folders/1PTPFjDdx2Ky0KsZdgn0p9x5fqyrdxKWF?usp=sharing). 
(We both offer training log and result.csv in same file.)

## Train
Please refer to the training configuration [Training Doc](https://github.com/simplify23/TPS_PP/blob/main/docs/en/training.md)

`CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config=configs/CDistNet_config.py`

## Eval
Please refer to the testing configuration [Training Doc](https://github.com/simplify23/TPS_PP/blob/main/docs/en/testing.md)

`CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CDistNet_config.py`


## Acknowledgement

This code is based on [MMOCR](https://github.com/open-mmlab/mmocr) MMOCR is an open-source project that is contributed by researchers and engineers from various colleges and companies. 

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation
```bash 
@article{Zheng2021CDistNetPM,
  title={CDistNet: Perceiving Multi-Domain Character Distance for Robust Text Recognition},
  author={Tianlun Zheng and Zhineng Chen and Shancheng Fang and Hongtao Xie and Yu-Gang Jiang},
  journal={ArXiv},
  year={2021},
  volume={abs/2111.11011}
}
```

