import os
import torchvision
import numpy as np
import torch
import timm
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet
from scipy.io import loadmat


def get_model(model_name,model_weights=None,data_path='/'):
    if model_name=='resnet50_robust':
        ds = ImageNet('/tmp')
        model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
                resume_path=os.path.join(data_path,'checkpoints',model_weights))
    else:
        try:
            model = torchvision.models.get_model(model_name,weights=model_weights)
        except:
            try:
                model = timm.create_model(model_name,pretrained=True)
            except:
                raise ValueError(model_name, "not found in torchvision.models or timm.")
    
    return model


def get_transforms(model,model_weights=None):
    try:
        transform = torchvision.models.get_weight(model_weights).transforms()
    except:
        try:
            data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
            transform = timm.data.create_transform(**data_cfg)
        except:
            raise ValueError(model_weights, "not found in torchvision.models or timm.")
    return transform


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