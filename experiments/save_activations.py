import torchvision
import torch
import glob
import os
import timm
import sys
sys.path.append('../src/')
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet

from masking.activation_model import ActivationModel

def main():
    """
    load model
    read images
    mask model
    save activations

    To run on alpine ami100 cluster:
        module purge
        module load rocm/5.2.3
        module load pytorch
        pip install timm
    """
    data_path = '/scratch/alpine/luha5813/ablation_data'
    torch.hub.set_dir(data_path)

    model_name = 'vit_b_16'
    model = torchvision.models.vit_b_16(weights='DEFAULT').to('cuda:0')
    model.eval()
    layers = [n for n,m in model.named_modules() if 
              isinstance(m,torchvision.models.vision_transformer.EncoderBlock)]

    image_list = glob.glob(os.path.join(data_path,'images/*'))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms(),
    ])

    masked_model = ActivationModel(model,layers)

    if not os.path.isdir(os.path.join(data_path,'activations',model_name)):
        os.mkdir(os.path.join(data_path,'activations',model_name))

    masked_model.save_activations(image_list,transforms,os.path.join(data_path,'activations',model_name))


    model_name = 'vit_l_16'
    model = torchvision.models.vit_l_16(weights='DEFAULT').to('cuda:0')
    model.eval()
    layers = [n for n,m in model.named_modules() if 
              isinstance(m,torchvision.models.vision_transformer.EncoderBlock)]

    image_list = glob.glob(os.path.join(data_path,'images/*'))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1.transforms(),
    ])

    masked_model = ActivationModel(model,layers)

    if not os.path.isdir(os.path.join(data_path,'activations',model_name)):
        os.mkdir(os.path.join(data_path,'activations',model_name))

    masked_model.save_activations(image_list,transforms,os.path.join(data_path,'activations',model_name))


    model_name = 'vit_medium_patch16_gap_256'
    model = timm.create_model('vit_medium_patch16_gap_256.sw_in12k_ft_in1k',pretrained=True).to('cuda:0')
    model.eval()
    layers = [n for n,m in model.named_modules() if 
              isinstance(m,timm.models.vision_transformer.Block)]

    image_list = glob.glob(os.path.join(data_path,'images/*'))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    masked_model = ActivationModel(model,layers)

    if not os.path.isdir(os.path.join(data_path,'activations',model_name)):
        os.mkdir(os.path.join(data_path,'activations',model_name))

    masked_model.save_activations(image_list,transforms,os.path.join(data_path,'activations',model_name))


    model_name = 'resnet50'
    model = torchvision.models.resnet50(weights='DEFAULT').to('cuda:0')
    model.eval()
    layers = [n for n,m in model.named_modules() if isinstance(m,torchvision.models.resnet.Bottleneck)]

    image_list = glob.glob(os.path.join(data_path,'images/*'))
    transforms = torchvision.models.ResNet50_Weights.DEFAULT.transforms()

    masked_model = ActivationModel(model,layers)

    if not os.path.isdir(os.path.join(data_path,'activations',model_name)):
        os.mkdir(os.path.join(data_path,'activations',model_name))

    masked_model.save_activations(image_list,transforms,os.path.join(data_path,'activations',model_name))


    model_name = 'resnet152'
    model = torchvision.models.resnet152(weights='DEFAULT').to('cuda:0')
    model.eval()
    layers = [n for n,m in model.named_modules() if isinstance(m,torchvision.models.resnet.Bottleneck)]

    image_list = glob.glob(os.path.join(data_path,'images/*'))
    transforms = torchvision.models.ResNet152_Weights.DEFAULT.transforms()

    masked_model = ActivationModel(model,layers)

    if not os.path.isdir(os.path.join(data_path,'activations',model_name)):
        os.mkdir(os.path.join(data_path,'activations',model_name))

    masked_model.save_activations(image_list,transforms,os.path.join(data_path,'activations',model_name))


    model_name = 'resnet50_robust'
    ds = ImageNet('/tmp')
    model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
             resume_path=os.path.join(data_path,'checkpoints','imagenet_l2_3_0.pt'))
    # model.load_state_dict(torch.load(os.path.join(data_path,'checkpoints','imagenet_l2_3_0.pt'))['model'])
    model.eval()
    layers = [n for n,m in model.named_modules() if isinstance(m,torchvision.models.resnet.Bottleneck)]

    image_list = glob.glob(os.path.join(data_path,'images/*'))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    masked_model = ActivationModel(model,layers)

    if not os.path.isdir(os.path.join(data_path,'activations',model_name)):
        os.mkdir(os.path.join(data_path,'activations',model_name))

    masked_model.save_activations(image_list,transforms,os.path.join(data_path,'activations',model_name))


    model_name = 'mixer_b16_224'
    model = timm.create_model('mixer_b16_224.goog_in21k_ft_in1k', pretrained=True).to('cuda:0')
    model.eval()
    layers = [n for n,m in model.named_modules() if isinstance(m,timm.models.mlp_mixer.MixerBlock)]

    image_list = glob.glob(os.path.join(data_path,'images/*'))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    masked_model = ActivationModel(model,layers)

    if not os.path.isdir(os.path.join(data_path,'activations',model_name)):
        os.mkdir(os.path.join(data_path,'activations',model_name))

    masked_model.save_activations(image_list,transforms,os.path.join(data_path,'activations',model_name))


    model_name = 'mixer_l16_224'
    model = timm.create_model('mixer_l16_224.goog_in21k_ft_in1k', pretrained=True).to('cuda:0')
    model.eval()
    layers = [n for n,m in model.named_modules() if isinstance(m,timm.models.mlp_mixer.MixerBlock)]

    image_list = glob.glob(os.path.join(data_path,'images/*'))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    masked_model = ActivationModel(model,layers)

    if not os.path.isdir(os.path.join(data_path,'activations',model_name)):
        os.mkdir(os.path.join(data_path,'activations',model_name))

    masked_model.save_activations(image_list,transforms,os.path.join(data_path,'activations',model_name))

if __name__=="__main__":
    main()