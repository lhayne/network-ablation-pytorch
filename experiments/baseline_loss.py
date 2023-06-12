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


def top5accuracy(true, predicted):
    """
    Function to predict the top 5 accuracy
    """
    assert len(true) == len(predicted)
    result = []
    flag  = 0
    for i in range(len(true)):
        flag  = 0
        temp = true[i]
        for j in predicted[i][0:5]:
            if j == temp:
                flag = 1
                break
        if flag == 1:
            result.append(1)
        else:
            result.append(0)
    counter = 0.
    for i in result:
        if i == 1:
            counter += 1.
    error = 1.0 - counter/float(len(result))
    #print len(np.where(np.asarray(result) == 1)[0])
    return len(np.where(np.asarray(result) == 1)[0]), error


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

    meta_clsloc_file = os.path.join(data_path,'meta_clsloc.mat')
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

    # load model
    model_name = 'vit_b_16'
    model = torchvision.models.vit_b_16(weights='DEFAULT').to('cuda:0')
    model.eval()

    # load images
    image_list = np.sort(glob.glob(os.path.join(data_path,'images/*')))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms(),
    ])

    # load labels
    labels = get_labels(data_path,image_list)
    true_valid_wids = get_wid_labels(data_path,image_list)

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
            out = model(im.unsqueeze(0).to('cuda:0'))
            outs.append(out)
    out = torch.row_stack(outs)
    print(out.shape)

    # calculate accuracy
    predicted_valid_wids = []
    for i in range(len(image_list)):
        predicted_valid_wids.append(pprint_output(out.cpu().numpy()[i],1000))
    predicted_valid_wids = np.asarray(predicted_valid_wids)
    count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)
    print(model_name,count,error)

    # calculate per class losses and save
    losses = {}
    for label in np.sort(np.unique(labels)):
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(out[labels==label], torch.from_numpy(labels[labels==label]).to('cuda:0'))
        losses[label] = loss.cpu().numpy()
        print(model_name,losses[label])
    pickle.dump(losses,open(os.path.join(data_path,'baseline_losses',model_name+'.pkl'),'wb'))

    del model
    del images
    gc.collect()

    # load model
    model_name = 'vit_l_16'
    model = torchvision.models.vit_l_16(weights='DEFAULT').to('cuda:0')
    model.eval()

    # load images
    image_list = np.sort(glob.glob(os.path.join(data_path,'images/*')))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1.transforms(),
    ])

    # load labels
    labels = get_labels(data_path,image_list)
    true_valid_wids = get_wid_labels(data_path,image_list)

    # transform images
    images = []
    for filename in image_list:
        im = Image.open(filename).convert('RGB')
        images.append(transforms(im))
    images = torch.stack(images).to('cuda:0')

    # run model
    outs = []
    for im in images:
        with torch.no_grad():
            out = model(im.unsqueeze(0).to('cuda:0'))
            outs.append(out)
    out = torch.row_stack(outs)
    print(out.shape)

    # calculate accuracy
    predicted_valid_wids = []
    for i in range(len(image_list)):
        predicted_valid_wids.append(pprint_output(out.cpu().numpy()[i],1000))
    predicted_valid_wids = np.asarray(predicted_valid_wids)
    count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)
    print(model_name,count,error)

    # calculate per class losses and save
    losses = {}
    for label in np.sort(np.unique(labels)):
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(out[labels==label], torch.from_numpy(labels[labels==label]).to('cuda:0'))
        losses[label] = loss.cpu().numpy()
        print(model_name,losses[label])
    pickle.dump(losses,open(os.path.join(data_path,'baseline_losses',model_name+'.pkl'),'wb'))

    del model
    del images
    gc.collect()

    # load model
    model_name = 'vit_medium_patch16_gap_256'
    model = timm.create_model('vit_medium_patch16_gap_256.sw_in12k_ft_in1k',pretrained=True).to('cuda:0')
    model.eval()

    # load images
    image_list = np.sort(glob.glob(os.path.join(data_path,'images/*')))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load labels
    labels = get_labels(data_path,image_list)
    true_valid_wids = get_wid_labels(data_path,image_list)

    # transform images
    images = []
    for filename in image_list:
        im = Image.open(filename).convert('RGB')
        images.append(transforms(im))
    images = torch.stack(images).to('cuda:0')

    # run model
    outs = []
    for im in images:
        with torch.no_grad():
            out = model(im.unsqueeze(0).to('cuda:0'))
            outs.append(out)
    out = torch.row_stack(outs)
    print(out.shape)

    # calculate accuracy
    predicted_valid_wids = []
    for i in range(len(image_list)):
        predicted_valid_wids.append(pprint_output(out.cpu().numpy()[i],1000))
    predicted_valid_wids = np.asarray(predicted_valid_wids)
    count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)
    print(model_name,count,error)

    # calculate per class losses and save
    losses = {}
    for label in np.sort(np.unique(labels)):
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(out[labels==label], torch.from_numpy(labels[labels==label]).to('cuda:0'))
        losses[label] = loss.cpu().numpy()
        print(model_name,losses[label])
    pickle.dump(losses,open(os.path.join(data_path,'baseline_losses',model_name+'.pkl'),'wb'))

    del model
    del images
    gc.collect()

    # load model
    model_name = 'resnet50'
    model = torchvision.models.resnet50(weights='DEFAULT').to('cuda:0')
    model.eval()

    # load images
    image_list = glob.glob(os.path.join(data_path,'images/*'))
    transforms = torchvision.models.ResNet50_Weights.DEFAULT.transforms()

    # load labels
    labels = get_labels(data_path,image_list)
    true_valid_wids = get_wid_labels(data_path,image_list)

    # transform images
    images = []
    for filename in image_list:
        im = Image.open(filename).convert('RGB')
        images.append(transforms(im))
    images = torch.stack(images).to('cuda:0')

    # run model
    outs = []
    for im in images:
        with torch.no_grad():
            out = model(im.unsqueeze(0).to('cuda:0'))
            outs.append(out)
    out = torch.row_stack(outs)
    print(out.shape)

    # calculate accuracy
    predicted_valid_wids = []
    for i in range(len(image_list)):
        predicted_valid_wids.append(pprint_output(out.cpu().numpy()[i],1000))
    predicted_valid_wids = np.asarray(predicted_valid_wids)
    count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)
    print(model_name,count,error)

    # calculate per class losses and save
    losses = {}
    for label in np.sort(np.unique(labels)):
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(out[labels==label], torch.from_numpy(labels[labels==label]).to('cuda:0'))
        losses[label] = loss.cpu().numpy()
        print(model_name,losses[label])
    pickle.dump(losses,open(os.path.join(data_path,'baseline_losses',model_name+'.pkl'),'wb'))

    del model
    del images
    gc.collect()

    # load model
    model_name = 'resnet152'
    model = torchvision.models.resnet152(weights='DEFAULT').to('cuda:0')
    model.eval()

    # load images
    image_list = glob.glob(os.path.join(data_path,'images/*'))
    transforms = torchvision.models.ResNet152_Weights.DEFAULT.transforms()

    # load labels
    labels = get_labels(data_path,image_list)
    true_valid_wids = get_wid_labels(data_path,image_list)

    # transform images
    images = []
    for filename in image_list:
        im = Image.open(filename).convert('RGB')
        images.append(transforms(im))
    images = torch.stack(images).to('cuda:0')

    # run model
    outs = []
    for im in images:
        with torch.no_grad():
            out = model(im.unsqueeze(0).to('cuda:0'))
            outs.append(out)
    out = torch.row_stack(outs)
    print(out.shape)

    # calculate accuracy
    predicted_valid_wids = []
    for i in range(len(image_list)):
        predicted_valid_wids.append(pprint_output(out.cpu().numpy()[i],1000))
    predicted_valid_wids = np.asarray(predicted_valid_wids)
    count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)
    print(model_name,count,error)

    # calculate per class losses and save
    losses = {}
    for label in np.sort(np.unique(labels)):
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(out[labels==label], torch.from_numpy(labels[labels==label]).to('cuda:0'))
        losses[label] = loss.cpu().numpy()
        print(model_name,losses[label])
    pickle.dump(losses,open(os.path.join(data_path,'baseline_losses',model_name+'.pkl'),'wb'))

    del model
    del images
    gc.collect()

    # load model
    model_name = 'mixer_b16_224'
    model = timm.create_model('mixer_b16_224.goog_in21k_ft_in1k', pretrained=True).to('cuda:0')
    model.eval()

    # load images
    image_list = glob.glob(os.path.join(data_path,'images/*'))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load labels
    labels = get_labels(data_path,image_list)
    true_valid_wids = get_wid_labels(data_path,image_list)

    # transform images
    images = []
    for filename in image_list:
        im = Image.open(filename).convert('RGB')
        images.append(transforms(im))
    images = torch.stack(images).to('cuda:0')

    # run model
    outs = []
    for im in images:
        with torch.no_grad():
            out = model(im.unsqueeze(0).to('cuda:0'))
            outs.append(out)
    out = torch.row_stack(outs)
    print(out.shape)

    # calculate accuracy
    predicted_valid_wids = []
    for i in range(len(image_list)):
        predicted_valid_wids.append(pprint_output(out.cpu().numpy()[i],1000))
    predicted_valid_wids = np.asarray(predicted_valid_wids)
    count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)
    print(model_name,count,error)

    # calculate per class losses and save
    losses = {}
    for label in np.sort(np.unique(labels)):
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(out[labels==label], torch.from_numpy(labels[labels==label]).to('cuda:0'))
        losses[label] = loss.cpu().numpy()
        print(model_name,losses[label])
    pickle.dump(losses,open(os.path.join(data_path,'baseline_losses',model_name+'.pkl'),'wb'))

    del model
    del images
    gc.collect()

    # load model
    model_name = 'mixer_l16_224'
    model = timm.create_model('mixer_l16_224.goog_in21k_ft_in1k', pretrained=True).to('cuda:0')
    model.eval()

    # load images
    image_list = glob.glob(os.path.join(data_path,'images/*'))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load labels
    labels = get_labels(data_path,image_list)
    true_valid_wids = get_wid_labels(data_path,image_list)

    # transform images
    images = []
    for filename in image_list:
        im = Image.open(filename).convert('RGB')
        images.append(transforms(im))
    images = torch.stack(images).to('cuda:0')

    # run model
    outs = []
    for im in images:
        with torch.no_grad():
            out = model(im.unsqueeze(0).to('cuda:0'))
            outs.append(out)
    out = torch.row_stack(outs)
    print(out.shape)

    # calculate accuracy
    predicted_valid_wids = []
    for i in range(len(image_list)):
        predicted_valid_wids.append(pprint_output(out.cpu().numpy()[i],1000))
    predicted_valid_wids = np.asarray(predicted_valid_wids)
    count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)
    print(model_name,count,error)

    # calculate per class losses and save
    losses = {}
    for label in np.sort(np.unique(labels)):
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(out[labels==label], torch.from_numpy(labels[labels==label]).to('cuda:0'))
        losses[label] = loss.cpu().numpy()
        print(model_name,losses[label])
    pickle.dump(losses,open(os.path.join(data_path,'baseline_losses',model_name+'.pkl'),'wb'))


if __name__=="__main__":
    main()