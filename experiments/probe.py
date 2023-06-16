import numpy as np
import glob
import os
import torch
import pickle
import gc
from re import split
import argparse
import pandas as pd
import sys
sys.path.append('../src/')

from utils.model_utils import get_model, to_categorical, rls, acc
from utils.data_utils import ImageWids, get_labels, get_class_wids

def main():
    """
    load model
    read images
    load labels
    load clusters
    mask model
    save accuracies
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
    parser.add_argument('--num_classes', default=50, type=int, help='The number of image classes to include.')
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
    print(len(image_list))

    # filter images by class
    image_wids = ImageWids(os.path.join(args.data_path,'wid_labels.pkl'))
    image_list = [im for im in image_list if image_wids[im] in get_class_wids()[:args.num_classes]]
    print(len(image_list))

    # load labels
    labels = get_labels(args.data_path,image_list)

    # load clusters
    clusters = pickle.load(open(os.path.join(args.data_path,args.experiment_name,args.model_name,'clusters.pkl'),'rb'))

    accuracies = pd.DataFrame([],columns=['model','layer','cluster_idx','label','accuracy'])

    # Load activations
    for layer in layers:
        if len(clusters[layer]) == 0:
            continue

        activation_matrix = []
        for file in np.sort(glob.glob(os.path.join(args.data_path,'activations',args.model_name,'*'))):
            wid = image_wids[file]
            if wid in get_class_wids()[:args.num_classes]:
                print(file)
                activations = pickle.load(open(file,'rb'))
                if args.model_name in ['vit_b_16','vit_l_16']:
                    activation_matrix.append(activations[layer][:,1:,:]) # Remove classification token
                else:
                    activation_matrix.append(activations[layer])

        activation_matrix = np.stack(activation_matrix).reshape(len(activation_matrix),-1)
        print(layer,activation_matrix.shape)

        for cluster_idx in np.sort(np.unique(clusters[layer])):
            if cluster_idx != -1 and len(clusters[layer][clusters[layer]==cluster_idx]) < 10000: 
                cluster_activations = activation_matrix[:,np.argwhere(clusters[layer]==cluster_idx).squeeze()]

                # calculate per class accuracies and save
                for label in np.sort(np.unique(labels)):
                    y = np.asarray([1 if l==label else 0 for l in labels])
                    training_indices = np.random.choice(np.arange(len(labels)),int(len(labels)*0.8),replace=False)
                    testing_indices = np.setdiff1d(np.arange(len(labels)),training_indices)
                    y_one_hot = to_categorical(y,2)
                    print(torch.from_numpy(cluster_activations[training_indices]).shape,torch.from_numpy(y_one_hot[training_indices]).shape)
                    w = rls(torch.from_numpy(cluster_activations[training_indices]).float().to(args.device),
                            torch.from_numpy(y_one_hot[training_indices]).float().to(args.device),
                            penalty=10)
                    regularized_decoding_accuracy = acc(
                        torch.from_numpy(cluster_activations[testing_indices]).float().to(args.device),
                        torch.from_numpy(y_one_hot[testing_indices]).float().to(args.device),
                        w).cpu().numpy()

                    accuracies.loc[len(accuracies)] = [args.model_name,layer,cluster_idx,label,regularized_decoding_accuracy]
                    
                accuracies.to_csv(os.path.join(args.data_path,args.experiment_name,args.model_name,'linear_probes.csv'))


if __name__=="__main__":
    main()