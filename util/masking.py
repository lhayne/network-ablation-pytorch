import torch

class HookedModel(torch.nn.Module):
    """
    Constructs a model for applying forward hooks.

    Interface:
        1. Choose model
        2. Choose layer
        3. Choose mask
        4. Apply mask
        5. Evaluate model
        6. Remove mask
    """
    def __init__(self,model):
        super(HookedModel,self).__init__()
        self.model = model
        self.hooks = []
    
    def forward(self,*args,**kwargs):
        return self.model(*args,**kwargs)

    def apply_hook(self,layer_name,hook):
        self.hooks.append(
            get_module(self.model,layer_name).register_forward_hook(hook)
        )

    def remove_hooks(self):
        for _ in range(len(self.hooks)):
            hook = self.hooks.pop()
            hook.remove()


class OutputMaskHook:
    """
    Hook for applying elementwise mask to output of layer.
    """
    def __init__(self,mask):
        self.mask = mask

    def __call__(self,model, input, output):
        output = torch.mul(output, self.mask) # Elementwise multiplication
        return output


class GetActivationsHook:
    """
    Hook for retrieving activations from output of layer.
    """
    def __init__(self,name):
        self.name = name
        self.activations = []

    def __call__(self,model, input, output):
        self.activations = output.detach()
    
    def get_activations(self):
        return self.activations

def get_module(model, name):
    """
    Finds the named module within the given model.

    Courtesy of https://github.com/kmeng01/rome/blob/main/util/nethook.py#L355
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


