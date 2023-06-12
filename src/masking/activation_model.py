import torch 
import os
from PIL import Image
from re import split
import copy
import pickle

from masking.base import HookedModel


class ActivationModel(HookedModel):
    def __init__(self,model,layers):
        super(ActivationModel,self).__init__(model)

        self.layers = layers
        
        for layer in layers:
            hook = GetActivationsHook(layer)
            self.apply_hook(layer,hook)

    def get_activations(self):
        return dict(zip(self.hooks.keys(),
                        [self.hooks[k].get_activations() for k in self.hooks.keys()]))

    def save_activations(self, image_list, transforms, output_location):
        """
        Mask all layers
        For every image
            Evaluate model
            Save activations
        """
        for filename in image_list:

            if os.path.isfile(os.path.join(output_location,split('\.|\/',filename)[-2]+'.pkl')):
                continue

            print('saving ',split('\.|\/',filename)[-2])

            im = Image.open(filename).convert('RGB')
            im = transforms(im).unsqueeze(0).to('cuda:0')

            with torch.no_grad():
                self(im)
            
            activations = self.get_activations()

            pickle.dump(activations,open(os.path.join(output_location,split('\.|\/',filename)[-2]+'.pkl'),'wb'))


class GetActivationsHook:
    """
    Hook for retrieving activations from output of layer.
    """
    def __init__(self,name):
        self.name = name
        self.activations = []

    def __call__(self,model, input, output):
        self.activations = output.clone().cpu().detach().numpy()
    
    def get_activations(self):
        return self.activations
