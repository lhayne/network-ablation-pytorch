import glob
import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
from umap import UMAP
import gc
import argparse
from re import split
import warnings
import sys
sys.path.append('../src/')
from collections import Counter

from utils.model_utils import get_model, to_categorical
from utils.data_utils import get_wid_labels, get_classes, get_class_wids, ImageWids, get_labels


def main():
    """
    load activations
    apply umap
    cluster neurons
    save clusters

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
    parser.add_argument('--model_weights', default='', help='The type of layer from which to get activations.')
    parser.add_argument('--class_start', default=0, type=int, help='The first class to include.')
    parser.add_argument('--class_end', default=50, type=int, help='The last class to include.')
    args = parser.parse_args()

    path = os.path.join(args.data_path,'activations',args.model_name)
    model = get_model(args.model_name,args.model_weights,args.data_path)
    model.eval()
    layer_module = '.'.join(split('\.',args.layer_type)[:-1])
    layer_type = split('\.',args.layer_type)[-1]
    layers = [n for n,m in model.named_modules() if 
              isinstance(m,getattr(sys.modules[layer_module], layer_type))]
    del model
    gc.collect()

    image_wids = ImageWids(os.path.join(args.data_path,'wid_labels.pkl'))
    image_list = np.sort(glob.glob(os.path.join(args.data_path,'images/*')))

    # filter images by class
    image_list = [im for im in image_list if image_wids[im] in get_class_wids()[args.class_start:args.class_end]]

    # load labels
    labels = get_labels(args.data_path,image_list)
    label_to_index = dict(zip(np.sort(np.unique(labels)),np.arange(10)))
    y = [label_to_index[l] for l in labels]
    y_one_hot = to_categorical(y,len(label_to_index),binary=True)
    print('y',y_one_hot)

    # Load activations
    activation_matrix = {l:[] for l in layers}
    for layer in layers:
        # Load activations
        for file in np.sort(glob.glob(os.path.join(path,'*'))):
            wid = image_wids[file]
            if wid in get_class_wids()[args.class_start:args.class_end]:
                # print(file)

                activations = pickle.load(open(file,'rb'))
                
                if args.model_name in ['vit_b_16','vit_l_16']:
                    activation_matrix[layer].append(activations[layer][:,1:,:]) # Remove classification token
                else:
                    activation_matrix[layer].append(activations[layer])

        print('\n','Transform')

        # Transform
        activation_matrix[layer] = np.stack(activation_matrix[layer]).reshape(len(activation_matrix[layer]),-1)
        print(layer,activation_matrix[layer].shape,np.min(activation_matrix[layer]),np.max(activation_matrix[layer]))

        print('\n','Selective Magnitude')

        # Selective magnitude
        activation_matrix[layer] = (y_one_hot.T @ activation_matrix[layer]) / np.sum(y_one_hot,axis=0).reshape(-1,1)
        print(layer,activation_matrix[layer].shape,np.min(activation_matrix[layer]),np.max(activation_matrix[layer]),np.sum(y_one_hot,axis=0))

    if not os.path.isdir(os.path.join(args.data_path,args.experiment_name,args.model_name)):
        os.mkdir(os.path.join(args.data_path,args.experiment_name,args.model_name))

    pickle.dump(activation_matrix,open(os.path.join(args.data_path,args.experiment_name,args.model_name,'selective_magnitude.pkl'),'wb'))

if __name__=="__main__":
    main()