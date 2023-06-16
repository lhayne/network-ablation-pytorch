import torch
import numpy as np
import glob
import os
from scipy.io import loadmat
from PIL import Image
import pickle
import gc
import argparse
from re import split
import sys
sys.path.append('../src/')

from utils.model_utils import get_model, get_transforms 
from utils.data_utils import get_labels, get_wid_labels, top5accuracy


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
    parser.add_argument('--model_weights', default=None, help='Model weights to use.')
    parser.add_argument('--data_path', help='Path to store data.')
    parser.add_argument('--device', help='Device on which to run model.')
    args = parser.parse_args()

    meta_clsloc_file = os.path.join(args.data_path,'meta_clsloc.mat')
    synsets = loadmat(meta_clsloc_file)['synsets'][0]
    synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]],key=lambda v: v[1])
    corr = {}
    for j in range(1000):
        corr[synsets_imagenet_sorted[j][0]] = j

    corr_inv = {}
    for j in range(1, 1001):
        corr_inv[corr[j]] = j

    def pprint_output(out, n_max_synsets=10):
        wids = []
        best_ids = out.argsort()[::-1][:n_max_synsets]
        for u in best_ids:
            wids.append(str(synsets[corr_inv[u] - 1][1][0]))
        return wids

    torch.hub.set_dir(args.data_path)

    model = get_model(args.model_name,('DEFAULT' if 'DEFAULT' in args.model_weights else args.model_weights))
    model.to(args.device)
    model.eval()

    # load images
    image_list = glob.glob(os.path.join(args.data_path,'images/*'))
    transforms = get_transforms(model,args.model_weights)

    # load labels
    labels = get_labels(args.data_path,image_list)
    true_valid_wids = get_wid_labels(args.data_path,image_list)

    # transform images
    images = []
    for filename in image_list:
        im = Image.open(filename).convert('RGB')
        images.append(transforms(im))
    images = torch.stack(images)

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

    # calculate accuracy
    predicted_valid_wids = []
    for i in range(len(image_list)):
        predicted_valid_wids.append(pprint_output(out.cpu().numpy()[i],1000))
    predicted_valid_wids = np.asarray(predicted_valid_wids)
    count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)
    print(args.model_name,count,error)

    # calculate per class losses and save
    losses = {}
    for label in np.sort(np.unique(labels)):
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(out[labels==label], torch.from_numpy(labels[labels==label]).to(args.device))
        losses[label] = loss.cpu().numpy()
        print(args.model_name,losses[label])
    pickle.dump(losses,open(os.path.join(args.data_path,'baseline_losses',args.model_name+'.pkl'),'wb'))


if __name__=="__main__":
    main()