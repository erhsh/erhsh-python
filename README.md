# Welcome to Use Erhsh Python

You can use **Erhsh Python**:

* Easy to use MindSpore with `ems`
* Easy to use PyTorch with `ept`
* Easy to use Common Command with `etools`
* ...


------


## Install & Distribute

Firstly, get the source code and to compile it yourself. Or install it by pip directly.

### 1. Download Source code

```shell script
git clone https://github.com/erhsh/erhsh-python.git
```

### 2. Build & Develop

```shell script
python setupt.py develop
```

### 3. Distribute

```shell script
python setupt.py bdist_wheel upload
```


### 4. Install by pip

```shell script
pip install erhsh-python -i https://pypi.org/simple --trusted-host pypi.org
```

## Usage

### Easy to use MindSpore: `ems`

About Easy use MindSpore, we provide easy way to:

> * Easy to get *`ckpt`* file information
> * Easy to generate *`hccl`* configuration file
> * Easy to get *`net`* demo, such as: `LeNet`, `GoogleNet`, `VGG` etc.
> * Easy to get *`dataset`* demo, such as: `GeneratorDataset`, `ImageFolderDataset`, `Cifar10Dataset` etc.
> * Easy to get *`train`* demo, such as: `train_demo.py`
> * Easy to get *`ops`* demo, such as: `Conv2D`, `TensorAdd` etc.
> * ...
>

#### 1. Easy to get *`ckpt`* file information

> **Example**

```shell script
ems ckpt -d resnet.ckpt
```

> **Output:**

```shell script
...
```

#### 2 Easy to generate *`hccl`* configuration file

> **Example**

```shell script
ems hcclv1 -sid=10.10.10.10 --visible_devices=0,1,2,3,4,5,6,7
```

> **Output:**

```shell script
...
```

#### 3 Easy to get *`net`* demo

#### 4 Easy to get *`dataset`* demo

#### 5 Easy to get *`train`* demo

#### 6 Easy to get *`ops`* demo


### Easy to use MindSpore: `ept`

### Easy to use MindSpore: `etools`

## Others
