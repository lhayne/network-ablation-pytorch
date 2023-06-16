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

from utils.model_utils import get_model
from utils.data_utils import get_wid_labels, get_classes, get_class_wids, ImageWids


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
    parser.add_argument('--model_weights', default=None, help='The type of layer from which to get activations.')
    parser.add_argument('--num_classes', default=50, type=int, help='Number of classes to visualize.')
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

    # Load activations
    activation_matrix = {l:[] for l in layers}
    skip = False
    for layer in layers:
        # Load activations
        for file in np.sort(glob.glob(os.path.join(path,'*'))):
            wid = image_wids[file]
            if wid in get_class_wids()[:args.num_classes]:
                print(file)
                activations = pickle.load(open(file,'rb'))
                
                if args.model_name in ['vit_b_16','vit_l_16']:
                    activation_matrix[layer].append(activations[layer][:,1:,:]) # Remove classification token
                else:
                    activation_matrix[layer].append(activations[layer])

                if activation_matrix[layer][0].size > 300000:
                    warnings.warn('Skipping layer '+layer+' too many neurons: '+str(activation_matrix[layer][0].size))
                    activation_matrix[layer] = []
                    skip = True
                    break

        if skip:
            skip = False
            continue

        print('\n','Transform')

        # Transform
        activation_matrix[layer] = np.stack(activation_matrix[layer]).reshape(len(activation_matrix[layer]),-1)
        print(layer,activation_matrix[layer].shape)

        print('\n','PCA')

        # PCA
        activation_matrix[layer] = PCA(n_components=50).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

        print('\n','UMAP')

        # UMAP
        activation_matrix[layer] = UMAP(low_memory=False,verbose=True).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    if not os.path.isdir(os.path.join(args.data_path,args.experiment_name,args.model_name)):
        os.mkdir(os.path.join(args.data_path,args.experiment_name,args.model_name))

    pickle.dump(activation_matrix,open(os.path.join(args.data_path,args.experiment_name,args.model_name,'umap.pkl'),'wb'))

if __name__=="__main__":
    main()