import glob
import os
import timm
import numpy as np
import pickle
import torch
import torchvision
from sklearn.decomposition import PCA
from umap import UMAP
from hdbscan import HDBSCAN
from umap.umap_ import nearest_neighbors
import robustness
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet
import gc
from re import split

import sys
sys.path.append('../src/')

from masking.activation_model import ActivationModel
from utils.model_utils import get_model

def main():
    """
    To run on alpine ami100 cluster:
        module purge
        module load rocm/5.2.3
        module load pytorch
        pip install timm
        pip install umap-learn
        pip install hdbscan

    Or to run on alpine cpu cluster:
        module purge
        module load anaconda
        conda activate ablation
    """
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--model_name', help='Model to use.')
    parser.add_argument('--data_path', help='Path to store data.')
    parser.add_argument('--experiment_name', help='Name of experiment to add to path.')
    parser.add_argument('--layer_type', help='The type of layer from which to get activations.')
    args = parser.parse_args()

    path = os.path.join(args.data_path,'activations',args.model_name)
    model = get_model(args.model_name,args.model_weights)
    model.eval()
    layer_module = '.'.join(split('\.',args.layer_type)[:-1])
    layer_type = split('\.',args.layer_type)[-1]
    layers = [n for n,m in model.named_modules() if 
              isinstance(m,getattr(sys.modules[layer_module], layer_type))]
    del model
    gc.collect()

    activation_matrix = pickle.load(open(os.path.join(args.data_path,args.experiment_name,args.model_name,'umap.pkl'),'rb'))

    # HDBSCAN
    clusters = {}
    for layer in layers:
        n_neurons = activation_matrix[layer].shape[1]
        clusters[layer] = HDBSCAN().fit_predict(activation_matrix[layer].T)
        print(layer,len(clusters[layer]),np.unique(clusters[layer]))

    if not os.path.isdir(os.path.join(args.data_path,args.experiment_name,args.model_name)):
        os.mkdir(os.path.join(args.data_path,args.experiment_name,args.model_name))

    pickle.dump(clusters,open(os.path.join(args.data_path,args.experiment_name,args.model_name,'clusters.pkl'),'wb'))


if __name__=="__main__":
    main()