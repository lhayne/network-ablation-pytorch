# Network Ablation Pytorch

![UMAP visualizations of activations in ViT and MLP-Mixer models.](/img/umap-visualization.png)

This repository provides modules for collecting, ablating, and visualizing activations in pretrained models in pytorch.
Researchers can use these modules for interpreting neural network models.
To this end, we provide two main classes.
First, the `ActivationModel` class is a wrapper for a `torch.nn.Module` and applies [forward hooks](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) for collecting activations during inference.
Second, the `MaskedModel` class applies forward hooks manipulating activations during inference.

## Setup
On a Mac, create your environment by running the following commands in your terminal. You will need pytorch 2.0 and the pip packages necessary for pytorch transformer models.

```
conda install pytorch torchvision torchaudio -c pytorch-nightly
pip install tqdm boto3 requests regex sentencepiece sacremoses
conda develop src
```
Running the develop command in conda allows for the use of modules included in this package.


## Getting started
The fastest way to get started is by using one of the experiment scripts provided in [`/experiments`](/experiments/).
For example, to save activations for a ViT-B16 model with default weights from each `EncoderBlock`, run the following command

```
python save_activations.py --model_name=vit_b_16 --model_weights=torchvision.models.ViT_B_16_Weights.DEFAULT --data_path=/path/to/images --layer_type=torchvision.models.vision_transformer.EncoderBlock
```

Make sure to specify a path to a directory that contains the images you want to feed to the model.
Activations will be saved to the same directory in pickle files, one for each image.

## Visualizing activations
You can additionally visualize activations using umap automatically by running the following command

```
python save_umap.py --model_name=vit_b_16 --data_path=/path/to/activations --experiment_name=my_experiment_name --layer_type=torchvision.models.vision_transformer.EncoderBlock
```

The UMAP coordinates will be saved in an experiment directory that you specify.