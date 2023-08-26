# P-tuning for LAMA

This repo contains example code for LAMA experiments in P-tuning.
这是刘潇大佬的[P-tuning](https://github.com/THUDM/P-tuning)在LAMA数据集上的一个小demo，本人在读源码过程中发现其写的代码非常好，但由于大佬代码的工程性很强，阅读起来并不是很快，因此本人在最小粒度上增加了很多注释和一个基于bert-base-uncased的最小demo的展示。

## 数据

数据原github文件放在[[Zenodo]](https://zenodo.org/record/5578210/files/P-tune_LAMA.tar.gz?download=1)这里，下载会有点慢，下载完后，**放在LAMA目录下**解压就好。
## 环境问题
1. 最好使用gpu运行，不然很多问题，浮点数运算会报错。
2. 经过我测试，原作者说明使用torch==1.5.0版本，但我在实际测试过程中用了1.8.0版本报错，但在colab上使用2.0版本的torch是没有问题的，猜测可能某个属性在1.5版本有，1.8版本时成为bug没继承好，在2.0版本进行修复，好像叫什么**AttributeError: module "torch' has no attribute 'frombuffer'**
所以建议直接py3.10+torch2.0 反正最新配置就好



## 运行

在终端运行这个

```bash
python cli.py
```

## 代码结构

大部分代码都写在p_tuning文件夹下，cli.py文件指定了很多参数，大佬写的基本把论文里提供的实验全部封装了，因此运行很容易，但读起来还是比较费力气，所以我简单介绍一下其中的几个特性。同时也是自己遇到的问题。

对于P-tuning最疑惑的一个点在于增加了pseudo_token后，这个东西没有标签到底是如何进行迭代了，最初接触过的代码有[何小枝大佬](https://github.com/HarderThenHarder/transformers_tasks/tree/main/prompt_tasks/p-tuning)的，这位大佬代码写的也很好，但并没有使用双向LSTM进行初始化，所以看起来就只是使用了一些[unused]的token进行初始化，我当时甚至觉得是大佬弄错了，我一直纠结的一个问题在于这些pseudo_token在整个模型前进的过程中他们只是相当于被引入的一些多余词而已，如果没有label，那他们对[MASK]位置的预测其实没有任何作用。

真的纠结的时间很长了，因为源码工程性太高一直不愿意去读，但最近花了一天多时间认真看了后，发现原作者确实用了双向LSTM进行初始化，但其实问题的本质在于作者论文里提到的fine-tune+P-tuning 或者单独的P-tuning。本例子的演示的默认参数就只有对prompt_encoder那部分的参数可以动的一个展示，当然刘潇大佬的代码把所有的功能都实现了，而且写的真的很好。。。

所以其实本质上，这就是现在大模型时代下一直强调的PEFT技术的一种体现方式，虽然只是bert-base-uncased这种330M(0.3B)这种小东西的一种用法。

当然，学习过程中看的东西也太多了，苏剑林大佬（眼光非常具有前瞻性，2021年的博客都已经看到了PEFT技术的影子）也复现了，但bert4keras我真的不想淌keras的浑水，毕竟torch都学不明白。。。

越写越像Blog了，日后更新整理吧。写给自己的readme好了。当然如果你学习路线碰巧跟我吻合的话，估计会感同身受吧。。。。

## 下面是作者原始的readme


## Usage

run cli.py to start the experiment

```bash
python cli.py
```

## Data

We adopt the dataset created by [AutoPrompt](https://github.com/ucinlp/autoprompt), and created the shared vocab which removes 
some stopwords using the scripts from original [LAMA](https://github.com/facebookresearch/LAMA). The packed up data file is available. 
[[THU Cloud Drive]](https://cloud.tsinghua.edu.cn/f/21b9dcf05cc44adfad25/?dl=1) or [[Zenodo]](https://zenodo.org/record/5578210/files/P-tune_LAMA.tar.gz?download=1)

If you use our packed up data, please download it and unzip it in the *data/* folder in the root directory.

## About MegatronLM (11B)

The original model checkpoint is available in [FairSeq](https://github.com/pytorch/fairseq/tree/master/examples/megatron_11b), 
which applies the Megatron for model parallel and need at least 8 V100 GPUs. 

In our experiment, we freeze the parameters of MegatronLM (11B) and only train the continuous prompt, and thus merge the 
splited 8 model partitions into one and load into a 32G V100 GPU. We provide the merge function in ./megatron_11b/megatron_wrapper.py. 
If you want to use the model parallel feature, please refer to the implemention in FairSeq and Megatron.

You can create a *checkpoints/* folder in the root directory.