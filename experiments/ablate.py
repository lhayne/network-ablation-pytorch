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

import sys
sys.path.append('../src/')

from masking.masked_model import MaskedModel


def get_wid_labels(data_path,image_list):
    #Load the details of all the 1000 classes and the function to convert the synset id to words
    meta_clsloc_file = os.path.join(data_path,'meta_clsloc.mat')
    synsets = loadmat(meta_clsloc_file)['synsets'][0]
    synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]],key=lambda v: v[1])

    #Code snippet to load the ground truth labels to measure the performance
    truth = {}
    with open(os.path.join(data_path,'ILSVRC2014_clsloc_validation_ground_truth.txt')) as f:
        line_num = 1
        for line in f.readlines():
            ind_ = int(line)
            temp  = None
            for i in synsets_imagenet_sorted:
                if i[0] == ind_:
                    temp = i
            #print ind_,temp
            if temp != None:
                truth[line_num] = temp
            else:
                print('##########', ind_)
                pass
            line_num += 1

    # Make list of wids
    true_valid_wids = []
    for i in image_list:
        temp1 = i.split('/')[-1]
        temp = temp1.split('.')[0].split('_')[-1]
        true_valid_wids.append(truth[int(temp)][1])
    true_valid_wids = np.asarray(true_valid_wids)

    return true_valid_wids


def get_labels(data_path,image_list):
    #Load the details of all the 1000 classes and the function to convert the synset id to words
    meta_clsloc_file = os.path.join(data_path,'meta_clsloc.mat')
    synsets = loadmat(meta_clsloc_file)['synsets'][0]
    synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]],key=lambda v: v[1])

    #Code snippet to load the ground truth labels to measure the performance
    truth = {}
    with open(os.path.join(data_path,'ILSVRC2014_clsloc_validation_ground_truth.txt')) as f:
        line_num = 1
        for line in f.readlines():
            ind_ = int(line)
            temp  = None
            for idx,i in enumerate(synsets_imagenet_sorted):
                if i[0] == ind_:
                    temp = idx
            #print ind_,temp
            if temp != None:
                truth[line_num] = temp
            else:
                print('##########', ind_)
                pass
            line_num += 1

    # Make list of wids
    true_valid_wids = []
    for i in image_list:
        temp1 = i.split('/')[-1]
        temp = temp1.split('.')[0].split('_')[-1]
        true_valid_wids.append(truth[int(temp)])
    true_valid_wids = np.asarray(true_valid_wids)

    return true_valid_wids


def pprint_output(out, n_max_synsets=10):
    meta_clsloc_file = data_path+'meta_clsloc.mat'
    synsets = loadmat(meta_clsloc_file)['synsets'][0]
    synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]],key=lambda v: v[1])
    corr = {}
    for j in range(1000):
        corr[synsets_imagenet_sorted[j][0]] = j

    corr_inv = {}
    for j in range(1, 1001):
        corr_inv[corr[j]] = j

    wids = []
    best_ids = out.argsort()[::-1][:n_max_synsets]
    for u in best_ids:
        wids.append(str(synsets[corr_inv[u] - 1][1][0]))
    return wids


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

    data_path = '/scratch/alpine/luha5813/ablation_data'
    torch.hub.set_dir(data_path)

    # load model
    model_name = 'vit_b_16'
    model = torchvision.models.vit_b_16(weights='DEFAULT').to('cuda:0')
    layers = [n for n,m in model.named_modules() if 
              isinstance(m,torchvision.models.vision_transformer.EncoderBlock)]

    # load images
    image_list = np.sort(glob.glob(os.path.join(data_path,'images/*')))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms(),
    ])

    # load labels
    labels = get_labels(data_path,image_list)

    # transform images
    images = []
    for filename in image_list:
        im = Image.open(filename).convert('RGB')
        images.append(transforms(im))
    images = torch.stack(images).to('cuda:0')

    # load clusters
    clusters = pickle.load(open(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'clusters.pkl'),'rb'))
    
    # load one activation to get output shape
    activation = pickle.load(open(os.path.join(data_path,'activations',model_name,'ILSVRC2012_val_00000011.pkl'),'rb'))

    losses = pd.DataFrame([],columns=['model','layer','cluster_idx','label','loss'])

    for layer in layers:
        for cluster_idx in np.sort(np.unique(clusters)):
            # construct mask
            mask = np.where(clusters==cluster_idx,1,0)
            mask = mask.reshape(activaiton[layer].shape[1:]) # reshape to same shape as non-batch dimensions
            layer_masks = {layer:mask}

            # mask model
            masked_model = MaskedModel(model,layer_masks)

            # run model
            outs = []
            for im in images:
                with torch.no_grad():
                    out = model(im.unsqueeze(0).to('cuda:0'))
                    outs.append(out)
            out = torch.row_stack(outs)
            print(out.shape)

            # calculate per class losses and save
            losses[cluster_idx] = {}
            for label in np.sort(np.unique(labels)):
                loss = torch.nn.CrossEntropyLoss()
                loss = loss(out[labels==label], torch.from_numpy(labels[labels==label]).to('cuda:0'))
                losses.loc[len(losses)] = [model_name,layer,cluster_idx,label,loss.cpu().numpy()]
                
                losses.to_csv(os.path.join(data_path,'pca_100_umap_default_HDBSCAN_1000',model_name,'ablation_losses.pkl'))

    del model
    del images
    gc.collect()


if __name__=="__main__":
    main()