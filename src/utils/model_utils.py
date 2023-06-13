import torchvision
import numpy as np
import torch
import timm
import robustness
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet


def get_model(model_name,model_weights=None):
    if model_name=='resnet50_robust':
        ds = ImageNet('/tmp')
        model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
                resume_path=os.path.join(data_path,'checkpoints',model_weights))
    else:
        try:
            model = torchvision.models.get_model(model_name,weights=model_weights)
        except:
            try:
                model = timm.create_model(model_name,weights=model_weights)
            except:
                raise ValueError(model_name, "not found in torchvision.models or timm.")
    
    return model


def get_transforms(model,model_weights=None):
    try:
        transform = torchvision.models.get_weights(model_weights).transforms()
    except:
        try:
            data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
            transform = timm.data.create_transform(**data_cfg)
        except:
            raise ValueError(model_weights, "not found in torchvision.models or timm.")
    return transform


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


def rls(X,Y,penalty=0):
    return (torch.linalg.inv(
                X.T @ X + penalty * X.shape[0] * torch.eye(X.shape[1],dtype=X.dtype,device=X.device)) 
            @ X.T @ Y)


def acc(X,Y,W):
    predictions = torch.argmax(X @ W, 1)
    labels = torch.argmax(Y, 1)
    return torch.count_nonzero(predictions==labels)/len(predictions)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return (2*np.eye(num_classes)-1)[y]