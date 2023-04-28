# TPS++: Attention-Enhanced Thin-Plate Spline for Scene Text Recognition
![1682662695807](https://user-images.githubusercontent.com/39580716/235069522-7e7c5013-9782-4b46-b757-0472df4c56b2.png)

The official code of TPS_PP (IJCAI 2023)

This code is based on MMOCR 0.4.0 ( [Documentation](https://mmocr.readthedocs.io/en/latest/) ) with **PyTorch 1.6+**.

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
│   ├── SynthText
```


## Pretrained Models

Get the pretrained models from [BaiduNetdisk(passwd:d6jd)](https://pan.baidu.com/s/1s0oNmd5jQJCvoH1efjfBdg), [GoogleDrive](https://drive.google.com/drive/folders/1PTPFjDdx2Ky0KsZdgn0p9x5fqyrdxKWF?usp=sharing). 
(We both offer training log and result.csv in same file.)

## Train
Please refer to the training configuration [Training Doc](https://github.com/simplify23/TPS_PP/blob/main/docs/en/training.md)
### CRNN+TPS++
Download [CRNN] in `mmocr_ijcai/crnn/crnn_latest.pth`
```
PORT=1234 ./tools/dist_train.sh configs/textrecog/crnn/crnn_tps++.py ./ckpt/ijcai_crnn_tps_pp 4 
          --seed=123456 --load-from=mmocr_ijcai/crnn/crnn_latest.pth
```
### NRTR+TPS++

Download [NRTR] in `mmocr_ijcai/nrtr/nrtr_latest.pth`

```
PORT=1234 ./tools/dist_train.sh configs/textrecog/nrtr/nrtr_tps++.py ./ckpt/ijcai_nrtr_tps_pp 4 
          --seed=123456 --load-from=mmocr_ijcai/nrtr/nrtr_latest.pth
```

## Testing
Please refer to the testing configuration [Testing Doc](https://github.com/simplify23/TPS_PP/blob/main/docs/en/testing.md)



## Acknowledgement

This code is based on [MMOCR](https://github.com/open-mmlab/mmocr)  

## License

This project is released under the [Apache 2.0 license](LICENSE).

