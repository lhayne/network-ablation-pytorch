import torch


def get_module(model, name):
    """
    Finds the named module within the given model.

    Courtesy of https://github.com/kmeng01/rome/blob/main/util/nethook.py#L355
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


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
        self.hooks = {}
        self.hook_handles = {}
    
    def forward(self,*args,**kwargs):
        return self.model(*args,**kwargs)

    def apply_hook(self,layer_name,hook):
        self.hooks[layer_name] = hook
        self.hook_handles[layer_name] = get_module(self.model,layer_name).register_forward_hook(hook)

    def remove_hooks(self):
        for k in self.hooks.keys():
            self.hook_handles[k].remove()
        
        self.hook_handles = {}
        self.hooks = {}

    def remove_hook(self,layer_name):
        self.hook_handles[layer_name].remove()
        del self.hook_handles[layer_name]
        del self.hooks[layer_name]