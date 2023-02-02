# Network Ablation Pytorch

This repository provides modules for ablating pretrained models in pytorch.

## Setup
On a Mac, create your environment by running the following commands in your terminal. You will need pytorch 2.0 and the pip packages necessary for pytorch transformer models.

```
conda install pytorch torchvision torchaudio -c pytorch-nightly
pip install tqdm boto3 requests regex sentencepiece sacremoses
conda develop src
```
Running the develop command in conda allows for the use of modules included in this package.
