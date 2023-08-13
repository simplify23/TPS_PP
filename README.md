# TPS++: Attention-Enhanced Thin-Plate Spline for Scene Text Recognition
![1682662695807](https://user-images.githubusercontent.com/39580716/235069522-7e7c5013-9782-4b46-b757-0472df4c56b2.png)

The official code of TPS_PP (IJCAI 2023) [Paper Link](https://arxiv.org/abs/2305.05322)

TPS++, an attention-enhanced TPS transformation that incorporates the attention mechanism to text rectification for the first time.  TPS++ builds a more flexible content-aware rectifier, generating a natural text correction that is easier to read by the subsequent recognizer. This code is based on MMOCR 0.4.0 ( [Documentation](https://mmocr.readthedocs.io/en/latest/) ) with **PyTorch 1.6+**.

## Code List
* [x] NRTR + TPS_PP
* [ ] CRNN + TPS_PP
* [ ] ABINet-LV + TPS_PP


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

Get the pretrained models from [BaiduNetdisk(passwd:cd9r)](https://pan.baidu.com/s/1qdFBhC-6Ahb6EID5UGyMiQ?pwd=cd9r), [GoogleDrive](https://drive.google.com/drive/folders/13695D2VuzgI9ICFE9_Ed3ZKR9phm9KYG?usp=share_link). 
checkpoint model in `model/xxx/latest.pth`, pre-train model in `pre_train/xxx/latest.pth`

|        Methods       	           |        IIIT5K       	| SVT       	| IC13        	| IC15      	| SVTP      	| CUTE      	| AVG
|:------------------:              |:------------------:	|:---------:	|:------:   	|:---------:	|:---------:	|:---------:	|:---------:
|       NRTR + TPS_PP      	           |         96.3           |    94.6   	|    96.6   	|    85.7   	|    89.0       |    92.4       | 92.4
|       NRTR + TPS_PP *      | 	     95.6          |    95.1 	    |    97.2   	|    85.9   	|   89.8       |    90.3       | 92.3

First, the model needs to be pre-trained using without TPS_PP (pre-train), and then trained end-to-end with a network that incorporates TPS_PP (checkpoint). * denotes the performance of the implemented code. checkpoint model in `model/xxx/latest.pth`, pre-train model in `pre_train/xxx/latest.pth`.

## Train
Please refer to the training configuration [Training Doc](https://github.com/simplify23/TPS_PP/blob/main/docs/en/training.md)

### NRTR+TPS++

Setp 1 : Download [NRTR](https://pan.baidu.com/s/1qdFBhC-6Ahb6EID5UGyMiQ?pwd=cd9r) `pre_train/nrtr/latest.pth` in `mmocr_ijcai/nrtr/latest.pth`


```
#Step 2
PORT=1234 ./tools/dist_train.sh configs/textrecog/nrtr/nrtr_tps++.py ./ckpt/ijcai_nrtr_tps_pp 4 
          --seed=123456 --load-from=mmocr_ijcai/nrtr/nrtr_latest.pth
```

<!-- ### CRNN+TPS++
Step 1 : Download [CRNN](https://pan.baidu.com/s/1qdFBhC-6Ahb6EID5UGyMiQ?pwd=cd9r) `pre_train/crnn/latest.pth` in `mmocr_ijcai/crnn/latest.pth`

Step 2

```

PORT=1234 ./tools/dist_train.sh configs/textrecog/crnn/crnn_tps++.py ./ckpt/ijcai_crnn_tps_pp 4 
          --seed=123456 --load-from=mmocr_ijcai/crnn/latest.pth
``` -->

## Testing
Please refer to the testing configuration [Testing Doc](https://github.com/simplify23/TPS_PP/blob/main/docs/en/testing.md)



## Acknowledgement

This code is based on [MMOCR](https://github.com/open-mmlab/mmocr)  

## Citation
If you find our method useful for your reserach, please cite
```bash 
@article{zheng2023tps++,
  title={TPS++: Attention-Enhanced Thin-Plate Spline for Scene Text Recognition},
  author={Zheng, Tianlun and Chen, Zhineng and Bai, Jinfeng and Xie, Hongtao and Jiang, Yu-Gang},
  journal={IJCAI},
  year={2023}
}
 ```

## License

This project is released under the [Apache 2.0 license](LICENSE).

