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

import sys
sys.path.append('../src/')

from masking.activation_model import ActivationModel

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
    data_path = '/scratch/alpine/luha5813/ablation_data'
    torch.hub.set_dir(data_path)

    model_name = 'vit_b_16'
    path = os.path.join(data_path,'activations',model_name)

    model = torchvision.models.vit_b_16()
    layers = [n for n,m in model.named_modules() if 
              isinstance(m,torchvision.models.vision_transformer.EncoderBlock)]
    del model

    # Load activations
    activation_matrix = {l:[] for l in layers}
    for file in np.sort(glob.glob(os.path.join(path,'*'))):
        print(file)
        activations = pickle.load(open(file,'rb'))
        for layer in layers:
            activation_matrix[layer].append(activations[layer][:,1:,:])

    print('\n','Transform')

    # Transform
    for layer in layers:
        activation_matrix[layer] = np.stack(activation_matrix[layer]).reshape(2500,-1)
        print(layer,activation_matrix[layer].shape)

    print('\n','PCA')

    # PCA
    for layer in layers:
        activation_matrix[layer] = PCA(n_components=50).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    print('\n','UMAP')

    # UMAP
    for layer in layers:
        activation_matrix[layer] = UMAP(low_memory=False,verbose=True).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(activation_matrix,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'umap.pkl'),'wb'))

    print('\n','HDBSCAN')

    # HDBSCAN
    clusters = {}
    for layer in layers:
        n_neurons = activation_matrix[layer].shape[1]
        clusters[layer] = HDBSCAN(min_cluster_size=int(n_neurons/1000)).fit_predict(activation_matrix[layer].T)
        print(layer,len(clusters[layer]),np.unique(clusters[layer]))

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(clusters,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'clusters.pkl'),'wb'))

    
    model_name = 'vit_l_16'
    path = os.path.join(data_path,'activations',model_name)

    model = torchvision.models.vit_l_16()
    layers = [n for n,m in model.named_modules() if 
              isinstance(m,torchvision.models.vision_transformer.EncoderBlock)]
    del model

    # Load activations
    activation_matrix = {l:[] for l in layers}
    for file in np.sort(glob.glob(os.path.join(path,'*'))):
        print(file)
        activations = pickle.load(open(file,'rb'))
        for layer in layers:
            activation_matrix[layer].append(activations[layer][:,1:,:])

    print('\n','Transform')

    # Transform
    for layer in layers:
        activation_matrix[layer] = np.stack(activation_matrix[layer]).reshape(2500,-1)
        print(layer,activation_matrix[layer].shape)

    print('\n','PCA')

    # PCA
    for layer in layers:
        activation_matrix[layer] = PCA(n_components=50).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    print('\n','UMAP')

    # UMAP
    for layer in layers:
        activation_matrix[layer] = UMAP(low_memory=False,verbose=True).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(activation_matrix,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'umap.pkl'),'wb'))

    print('\n','HDBSCAN')

    # HDBSCAN
    clusters = {}
    for layer in layers:
        n_neurons = activation_matrix[layer].shape[1]
        clusters[layer] = HDBSCAN(min_cluster_size=int(n_neurons/1000)).fit_predict(activation_matrix[layer].T)
        print(layer,len(clusters[layer]),np.unique(clusters[layer]))

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(clusters,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'clusters.pkl'),'wb'))


    model_name = 'vit_medium_patch16_gap_256'
    path = os.path.join(data_path,'activations',model_name)

    model = timm.create_model('vit_medium_patch16_gap_256.sw_in12k_ft_in1k',pretrained=True).to('cuda:0')
    layers = [n for n,m in model.named_modules() if 
              isinstance(m,torchvision.models.vision_transformer.EncoderBlock)]
    del model

    # Load activations
    activation_matrix = {l:[] for l in layers}
    for file in np.sort(glob.glob(os.path.join(path,'*'))):
        print(file)
        activations = pickle.load(open(file,'rb'))
        for layer in layers:
            activation_matrix[layer].append(activations[layer])

    print('\n','Transform')

    # Transform
    for layer in layers:
        activation_matrix[layer] = np.stack(activation_matrix[layer]).reshape(2500,-1)
        print(layer,activation_matrix[layer].shape)

    print('\n','PCA')

    # PCA
    for layer in layers:
        activation_matrix[layer] = PCA(n_components=50).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    print('\n','UMAP')

    # UMAP
    for layer in layers:
        activation_matrix[layer] = UMAP(low_memory=False,verbose=True).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(activation_matrix,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'umap.pkl'),'wb'))

    print('\n','HDBSCAN')

    # HDBSCAN
    clusters = {}
    for layer in layers:
        n_neurons = activation_matrix[layer].shape[1]
        clusters[layer] = HDBSCAN(min_cluster_size=int(n_neurons/1000)).fit_predict(activation_matrix[layer].T)
        print(layer,len(clusters[layer]),np.unique(clusters[layer]))

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(clusters,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'clusters.pkl'),'wb'))


    model_name = 'resnet50'
    path = os.path.join(data_path,'activations',model_name)

    model = torchvision.models.resnet50(weights='DEFAULT').to('cuda:0')
    layers = [n for n,m in model.named_modules() if isinstance(m,torchvision.models.resnet.Bottleneck)]
    del model

    # Load activations
    activation_matrix = {l:[] for l in layers}
    for file in np.sort(glob.glob(os.path.join(path,'*'))):
        print(file)
        activations = pickle.load(open(file,'rb'))
        for layer in layers:
            activation_matrix[layer].append(activations[layer])

    print('\n','Transform')

    # Transform
    for layer in layers:
        activation_matrix[layer] = np.stack(activation_matrix[layer]).reshape(2500,-1)
        print(layer,activation_matrix[layer].shape)

    print('\n','PCA')

    # PCA
    for layer in layers:
        activation_matrix[layer] = PCA(n_components=50).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    print('\n','UMAP')

    # UMAP
    for layer in layers:
        activation_matrix[layer] = UMAP(low_memory=False,verbose=True).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(activation_matrix,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'umap.pkl'),'wb'))

    print('\n','HDBSCAN')

    # HDBSCAN
    clusters = {}
    for layer in layers:
        n_neurons = activation_matrix[layer].shape[1]
        clusters[layer] = HDBSCAN(min_cluster_size=int(n_neurons/1000)).fit_predict(activation_matrix[layer].T)
        print(layer,len(clusters[layer]),np.unique(clusters[layer]))

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(clusters,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'clusters.pkl'),'wb'))


    model_name = 'resnet152'
    path = os.path.join(data_path,'activations',model_name)

    model = torchvision.models.resnet152(weights='DEFAULT').to('cuda:0')
    layers = [n for n,m in model.named_modules() if isinstance(m,torchvision.models.resnet.Bottleneck)]
    del model

    # Load activations
    activation_matrix = {l:[] for l in layers}
    for file in np.sort(glob.glob(os.path.join(path,'*'))):
        print(file)
        activations = pickle.load(open(file,'rb'))
        for layer in layers:
            activation_matrix[layer].append(activations[layer])

    print('\n','Transform')

    # Transform
    for layer in layers:
        activation_matrix[layer] = np.stack(activation_matrix[layer]).reshape(2500,-1)
        print(layer,activation_matrix[layer].shape)

    print('\n','PCA')

    # PCA
    for layer in layers:
        activation_matrix[layer] = PCA(n_components=50).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    print('\n','UMAP')

    # UMAP
    for layer in layers:
        activation_matrix[layer] = UMAP(low_memory=False,verbose=True).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(activation_matrix,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'umap.pkl'),'wb'))

    print('\n','HDBSCAN')

    # HDBSCAN
    clusters = {}
    for layer in layers:
        n_neurons = activation_matrix[layer].shape[1]
        clusters[layer] = HDBSCAN(min_cluster_size=int(n_neurons/1000)).fit_predict(activation_matrix[layer].T)
        print(layer,len(clusters[layer]),np.unique(clusters[layer]))

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(clusters,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'clusters.pkl'),'wb'))


    # model_name = 'resnet50_robust'
    # path = os.path.join(data_path,'activations',model_name)

    # model = torchvision.models.resnet50(weights='DEFAULT').to('cuda:0')
    # layers = [n for n,m in model.named_modules() if isinstance(m,torchvision.models.resnet.Bottleneck)]
    # del model

    # # Load activations
    # activation_matrix = {l:[] for l in layers}
    # for file in np.sort(glob.glob(os.path.join(path,'*'))):
    #     print(file)
    #     activations = pickle.load(open(file,'rb'))
    #     for layer in layers:
    #         activation_matrix[layer].append(activations[layer])

    # print('\n','Transform')

    # # Transform
    # for layer in layers:
    #     activation_matrix[layer] = np.stack(activation_matrix[layer]).reshape(2500,-1)
    #     print(layer,activation_matrix[layer].shape)

    # print('\n','PCA')

    # # PCA
    # for layer in layers:
    #     activation_matrix[layer] = PCA(n_components=50).fit_transform(activation_matrix[layer].T).T
    #     print(layer,activation_matrix[layer].shape)

    # print('\n','UMAP')

    # # UMAP
    # for layer in layers:
    #     activation_matrix[layer] = UMAP(low_memory=False,verbose=True).fit_transform(activation_matrix[layer].T).T
    #     print(layer,activation_matrix[layer].shape)

    # if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
    #     os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    # pickle.dump(activation_matrix,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'umap.pkl'),'wb'))

    # print('\n','HDBSCAN')

    # # HDBSCAN
    # clusters = {}
    # for layer in layers:
    #     n_neurons = activation_matrix[layer].shape[1]
    #     clusters[layer] = HDBSCAN(min_cluster_size=int(n_neurons/1000)).fit_predict(activation_matrix[layer].T)
    #     print(layer,len(clusters[layer]),np.unique(clusters[layer]))

    # if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
    #     os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    # pickle.dump(clusters,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'clusters.pkl'),'wb'))


    model_name = 'mixer_b16_224'
    path = os.path.join(data_path,'activations',model_name)

    model = timm.create_model('mixer_b16_224.goog_in21k_ft_in1k', pretrained=True).to('cuda:0')
    layers = [n for n,m in model.named_modules() if isinstance(m,torchvision.models.resnet.Bottleneck)]
    del model

    # Load activations
    activation_matrix = {l:[] for l in layers}
    for file in np.sort(glob.glob(os.path.join(path,'*'))):
        print(file)
        activations = pickle.load(open(file,'rb'))
        for layer in layers:
            activation_matrix[layer].append(activations[layer])

    print('\n','Transform')

    # Transform
    for layer in layers:
        activation_matrix[layer] = np.stack(activation_matrix[layer]).reshape(2500,-1)
        print(layer,activation_matrix[layer].shape)

    print('\n','PCA')

    # PCA
    for layer in layers:
        activation_matrix[layer] = PCA(n_components=50).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    print('\n','UMAP')

    # UMAP
    for layer in layers:
        activation_matrix[layer] = UMAP(low_memory=False,verbose=True).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(activation_matrix,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'umap.pkl'),'wb'))

    print('\n','HDBSCAN')

    # HDBSCAN
    clusters = {}
    for layer in layers:
        n_neurons = activation_matrix[layer].shape[1]
        clusters[layer] = HDBSCAN(min_cluster_size=int(n_neurons/1000)).fit_predict(activation_matrix[layer].T)
        print(layer,len(clusters[layer]),np.unique(clusters[layer]))

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(clusters,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'clusters.pkl'),'wb'))


    model_name = 'mixer_l16_224'
    path = os.path.join(data_path,'activations',model_name)

    model = timm.create_model('mixer_l16_224.goog_in21k_ft_in1k', pretrained=True).to('cuda:0')
    layers = [n for n,m in model.named_modules() if isinstance(m,torchvision.models.resnet.Bottleneck)]
    del model

    # Load activations
    activation_matrix = {l:[] for l in layers}
    for file in np.sort(glob.glob(os.path.join(path,'*'))):
        print(file)
        activations = pickle.load(open(file,'rb'))
        for layer in layers:
            activation_matrix[layer].append(activations[layer])

    print('\n','Transform')

    # Transform
    for layer in layers:
        activation_matrix[layer] = np.stack(activation_matrix[layer]).reshape(2500,-1)
        print(layer,activation_matrix[layer].shape)

    print('\n','PCA')

    # PCA
    for layer in layers:
        activation_matrix[layer] = PCA(n_components=50).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    print('\n','UMAP')

    # UMAP
    for layer in layers:
        activation_matrix[layer] = UMAP(low_memory=False,verbose=True).fit_transform(activation_matrix[layer].T).T
        print(layer,activation_matrix[layer].shape)

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(activation_matrix,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'umap.pkl'),'wb'))

    print('\n','HDBSCAN')

    # HDBSCAN
    clusters = {}
    for layer in layers:
        n_neurons = activation_matrix[layer].shape[1]
        clusters[layer] = HDBSCAN(min_cluster_size=int(n_neurons/1000)).fit_predict(activation_matrix[layer].T)
        print(layer,len(clusters[layer]),np.unique(clusters[layer]))

    if not os.path.isdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name)):
        os.mkdir(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name))

    pickle.dump(clusters,open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'clusters.pkl'),'wb'))


if __name__=="__main__":
    main()