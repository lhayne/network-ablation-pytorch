import torch
import torchvision
import timm
import numpy as np
import glob
import os
from scipy.io import loadmat
from PIL import Image
import pickle
import gc
from re import split

import sys
sys.path.append('../src/')

from masking.masked_model import MaskedModel
from utils.model_utils import get_model, get_transforms, get_labels, get_wid_labels, top5accuracy


def main():
    """
    load model
    read images
    load labels
    load clusters
    mask model
    save losses
    save probe accuracies

    To run on alpine ami100 cluster:
        module purge
        module load rocm/5.2.3
        module load pytorch
        pip install timm
    """

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--model_name', help='Model to use.')
    parser.add_argument('--model_weights', help='Model weights to use.')
    parser.add_argument('--data_path', help='Path to store data.')
    parser.add_argument('--experiment_name', help='Path to store data.')
    parser.add_argument('--device', help='Device on which to run model.')
    parser.add_argument('--layer_type', help='The type of layer from which to get activations.')
    args = parser.parse_args()

    torch.hub.set_dir(args.data_path)

    model = get_model(args.model_name,args.model_weights)
    model.to(args.device)
    model.eval()
    layer_module = '.'.join(split('\.',args.layer_type)[:-1])
    layer_type = split('\.',args.layer_type)[-1]
    layers = [n for n,m in model.named_modules() if 
              isinstance(m,getattr(sys.modules[layer_module], layer_type))]

    # load images
    image_list = glob.glob(os.path.join(args.data_path,'images/*'))
    transforms = get_transforms(args.model_weights)

    # load labels
    labels = get_labels(args.data_path,image_list)

    # transform images
    images = []
    for filename in image_list:
        im = Image.open(filename).convert('RGB')
        images.append(transforms(im))
    images = torch.stack(images).to(args.device)

    # load clusters
    clusters = pickle.load(open(os.path.join(args.data_path,args.experiment_name,args.model_name,'clusters.pkl'),'rb'))
    
    # load one activation to get output shape
    activation = pickle.load(open(os.path.join(args.data_path,'activations',args.model_name,'ILSVRC2012_val_00000011.pkl'),'rb'))

    losses = pd.DataFrame([],columns=['model','layer','cluster_idx','label','loss'])

    for layer in layers:
        for cluster_idx in np.sort(np.unique(clusters)):
            # construct mask
            mask = np.where(clusters==cluster_idx,1.,0.)

            # vit models do not mask classification token
            if args.model_name in ['vit_b_16','vit_l_16']:
                mask = np.row_stack((np.ones(activation[layer].shape[-1],dtype=np.float32),mask.reshape(activation[layer].shape[1:])))
            else:
                mask = mask.reshape(activation[layer].shape[1:]) # reshape to same shape as non-batch dimensions
            mask = torch.to_numpy(mask,dtype=torch.float32)
            layer_masks = {layer:mask}

            # mask model
            masked_model = MaskedModel(model,layer_masks)

            # run model
            outs = []
            for im in images:
                with torch.no_grad():
                    if args.model_name == 'resnet50_robust':
                        out = model(im.unsqueeze(0).to(args.device), with_image=False)
                    else:
                        out = model(im.unsqueeze(0).to(args.device))
                    outs.append(out)
            out = torch.row_stack(outs)
            print(out.shape)

            # calculate per class losses and save
            losses[cluster_idx] = {}
            for label in np.sort(np.unique(labels)):
                loss = torch.nn.CrossEntropyLoss()
                loss = loss(out[labels==label], torch.from_numpy(labels[labels==label]).to(args.device))
                losses.loc[len(losses)] = [model_name,layer,cluster_idx,label,loss.cpu().numpy()]
                
                losses.to_csv(os.path.join(args.data_path,args.experiment_name,args.model_name,'ablation_losses.csv'))


if __name__=="__main__":
    main()