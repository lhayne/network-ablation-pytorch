import os
import numpy as np
import pickle
from hdbscan import HDBSCAN
import gc
from re import split
import argparse
from sklearn.cluster import KMeans

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
    parser.add_argument('--method', default='hdbscan',help='Clustering method to use.')
    parser.add_argument('--num_groups', type=float, default=100, help='Minimum size of cluster passed to HDBSCAN.')
    args = parser.parse_args()

    model = get_model(args.model_name,model_weights=None)
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
        if len(activation_matrix[layer]) == 0:
            clusters[layer] = []
        else:
            n_neurons = activation_matrix[layer].shape[1]
            if args.method == 'hdbscan':
                clusters[layer] = HDBSCAN(min_cluster_size=int(n_neurons/(5*args.num_groups)), min_samples=1).fit_predict(activation_matrix[layer].T)
            elif args.method == 'kmeans':
                clusters[layer] = KMeans(n_clusters=int(args.num_groups)).fit_predict(activation_matrix[layer].T)
            else:
                raise ValueError('No clustering method',args.method)
            print(layer,'neurons',len(clusters[layer]),'clusters',len(np.unique(clusters[layer])))

    if not os.path.isdir(os.path.join(args.data_path,args.experiment_name,args.model_name)):
        os.mkdir(os.path.join(args.data_path,args.experiment_name,args.model_name))

    pickle.dump(clusters,open(os.path.join(args.data_path,args.experiment_name,args.model_name,'clusters.pkl'),'wb'))


if __name__=="__main__":
    main()