import torch
import glob
import os
import sys
sys.path.append('../src/')
import argparse
from re import split

from masking.activation_model import ActivationModel
from utils.model_utils import get_model, get_transforms

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

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--model_name', help='Model to use.')
    parser.add_argument('--model_weights',default='None', help='Model weights to use.')
    parser.add_argument('--data_path', help='Path to store data.')
    parser.add_argument('--device', help='Device on which to run model.')
    parser.add_argument('--layer_type', help='The type of layer from which to get activations.')
    args = parser.parse_args()

    torch.hub.set_dir(args.data_path)

    model = get_model(args.model_name,('DEFAULT' if 'DEFAULT' in args.model_weights else args.model_weights))
    model.to(args.device)
    model.eval()
    layer_module = '.'.join(split('\.',args.layer_type)[:-1])
    layer_type = split('\.',args.layer_type)[-1]
    layers = [n for n,m in model.named_modules() if 
              isinstance(m,getattr(sys.modules[layer_module], layer_type))]

    image_list = glob.glob(os.path.join(args.data_path,'images/*'))
    transforms = get_transforms(model,args.model_weights)

    masked_model = ActivationModel(model,layers)

    if not os.path.isdir(os.path.join(args.data_path,'activations',args.model_name)):
        os.mkdir(os.path.join(args.data_path,'activations',args.model_name))

    masked_model.save_activations(image_list,transforms,os.path.join(args.data_path,'activations',args.model_name))


if __name__=="__main__":
    main()