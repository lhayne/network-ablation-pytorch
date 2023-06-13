import torchvision
import timm
import numpy as np
import glob
import os
import torch
from scipy.io import loadmat
from PIL import Image
import pickle
import gc
from re import split

import sys
sys.path.append('../src/')

from masking.masked_model import MaskedModel
from utils.model_utils import get_model, get_transforms, 
    get_labels, get_wid_labels, top5accuracy, to_categorical, rls, acc


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

    del model
    gc.collect()

    # load images
    image_list = glob.glob(os.path.join(args.data_path,'images/*'))
    transforms = get_transforms(args.model_weights)

    # load labels
    labels = get_labels(args.data_path,image_list)

    # Load activations
    activation_matrix = {l:[] for l in layers}
    for file in np.sort(glob.glob(os.path.join(path,'*'))):
        print(file)
        activations = pickle.load(open(file,'rb'))
        for layer in layers:
            if args.model_name in ['vit_b_16','vit_l_16']:
                activation_matrix[layer].append(activations[layer][:,1:,:]) # Remove classification token
            else:
                activation_matrix[layer].append(activations[layer])

    print('\n','Transform')

    # Transform
    for layer in layers:
        activation_matrix[layer] = np.stack(activation_matrix[layer]).reshape(2500,-1)
        print(layer,activation_matrix[layer].shape)

    # load clusters
    clusters = pickle.load(open(os.path.join(args.data_path,args.experiment_name,args.model_name,'clusters.pkl'),'rb'))

    losses = pd.DataFrame([],columns=['model','layer','cluster_idx','label','loss'])

    for layer in layers:
        for cluster_idx in np.sort(np.unique(clusters)):
            
            activations = activation_matrix[layer][:,np.argwhere(clusters==cluster_idx)]

            # calculate per class losses and save
            losses[cluster_idx] = {}
            for label in np.sort(np.unique(labels)):
                y = np.asarray([1 if i==label else 0 for i in labels])
                training_indices = np.random.choice(np.arange(2500),500,replace=False)
                testing_indices = np.setdiff1d(np.arange(2500),training_indices)
                y_one_hot = to_categorical(y,2)
                w = rls(torch.from_numpy(activations[training_indices]).float().to(args.device),
                        torch.from_numpy(y_one_hot[training_indices]).float().to(args.device),
                        penalty=10)
                regularized_decoding_accuracy = acc(
                    torch.from_numpy(intact_activations[testing_indices]).float().to(args.device),
                    torch.from_numpy(y_one_hot[testing_indices]).float().to(args.device),
                    w).cpu().numpy()

                losses.loc[len(losses)] = [args.model_name,layer,cluster_idx,label,regularized_decoding_accuracy]
                
                losses.to_csv(os.path.join(args.data_path,args.experiment_name,args.model_name,'linear_probes.csv'))


if __name__=="__main__":
    main()