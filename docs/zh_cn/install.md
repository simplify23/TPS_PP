# 安装

## 环境依赖

- Linux (Windows 暂未获得官方支持)
- Python 3.7
- PyTorch 1.6 或更高版本
- torchvision 0.7.0
- CUDA 10.1
- NCCL 2
- GCC 5.4.0 或更高版本
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation) >= 1.3.8
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/#installation) >= 2.14.0

我们已经测试了以下操作系统和软件版本:

- OS: Ubuntu 16.04
- CUDA: 10.1
- GCC(G++): 5.4.0
- MMCV 1.3.8
- MMDetection 2.14.0
- PyTorch 1.6.0
- torchvision 0.7.0

MMOCR 基于 PyTorch 和 MMDetection 项目实现。

## 详细安装步骤

a. 创建一个 Conda 虚拟环境并激活（open-mmlab 为自定义环境名）。

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. 按照 PyTorch 官网教程安装 PyTorch 和 torchvision ([参见官方链接](https://pytorch.org/)), 例如,

```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```
:::{note}
请确定 CUDA 编译版本和运行版本一致。你可以在 [PyTorch](https://pytorch.org/) 官网检查预编译 PyTorch 所支持的 CUDA 版本。
:::


c. 安装 [mmcv](https://github.com/open-mmlab/mmcv)，推荐以下方式进行安装。

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

请将上述 url 中 ``{cu_version}`` 和 ``{torch_version}``替换成你环境中对应的 CUDA 版本和 PyTorch 版本。例如，如果想要安装最新版基于 CUDA 11 和 PyTorch 1.7.0 的最新版 `mmcv-full`，请输入以下命令:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```
:::{note}
使用 mmocr 0.2.0 及更高版本需要安装 mmcv 1.3.4 或更高版本。

如果安装时进行了编译过程，请再次确认安装的 `mmcv-full` 版本与环境中 CUDA 和 PyTorch 的版本匹配。即使是 PyTorch 1.7.0 和 1.7.1，`mmcv-full` 的安装版本也是有区别的。

如有需要，可以在[此处](https://github.com/open-mmlab/mmcv#installation)检查 mmcv 与 CUDA 和 PyTorch 的版本对应关系。
:::

:::{warning}
如果你已经安装过 `mmcv`，你需要先运行 `pip uninstall mmcv` 删除 `mmcv`，再安装 `mmcv-full`。 如果环境中同时安装了 `mmcv` 和 `mmcv-full`, 将会出现报错 `ModuleNotFoundError`。
:::

d. 安装 [mmdet](https://github.com/open-mmlab/mmdetection), 我们推荐使用pip安装最新版 `mmdet`。
在 [此处](https://pypi.org/project/mmdet/) 可以查看 `mmdet` 版本信息.

```shell
pip install mmdet
```

或者，你也可以按照 [安装指南](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) 中的方法安装 `mmdet`。


e. 克隆 MMOCR 项目到本地.

```shell
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
```

f. 安装依赖软件环境并安装 MMOCR。

```shell
pip install -r requirements.txt
pip install -v -e . # or "python setup.py develop"
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## 完整安装命令

以下是 conda 方式安装 mmocr 的完整安装命令。

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# 安装最新的 PyTorch 预编译包
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch

# 安装最新的 mmcv-full
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# 安装 mmdet
pip install mmdet

# 安装 mmocr
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr

pip install -r requirements.txt
pip install -v -e .  # 或 "python setup.py develop"
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## 可选方式: Docker镜像

我们提供了一个 [Dockerfile](https://github.com/open-mmlab/mmocr/blob/master/docker/Dockerfile) 文件以建立 docker 镜像 。

```shell
# build an image with PyTorch 1.6, CUDA 10.1
docker build -t mmocr docker/
```

使用以下命令运行。

```shell
docker run --gpus all --shm-size=8g -it -v {实际数据目录}:/mmocr/data mmocr
```

## 数据集准备

我们推荐建立一个 symlink 路径映射，连接数据集路径到 `mmocr/data`。 详细数据集准备方法请阅读**数据集**章节。
如果你需要的文件夹路径不同，你可能需要在 configs 文件中修改对应的文件路径信息。

 `mmocr` 文件夹路径结构如下：
```
├── configs/
├── demo/
├── docker/
├── docs/
├── LICENSE
├── mmocr/
├── README.md
├── requirements/
├── requirements.txt
├── resources/
├── setup.cfg
├── setup.py
├── tests/
├── tools/
```
