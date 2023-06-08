import torch 

from masking.base import HookedModel


class MaskedModel(HookedModel):
    def __init__(self,model,layer_masks):
        super(MaskedModel,self).__init__(model)

        self.layer_masks = layer_masks

        for layer,mask in layer_masks.items():
            mask_hook = OutputMaskHook(mask)
            self.apply_hook(layer,mask_hook)


class OutputMaskHook:
    """
    Hook for applying elementwise mask to output of layer.
    """
    def __init__(self,mask):
        self.mask = mask

    def __call__(self,model, input, output):
        output = torch.mul(output, self.mask) # Elementwise multiplication
        return output


class OutputReplaceHook:
    """
    Hook for applying elementwise mask to output of layer.
    Mask has numerical entries for elements to replace, NaNs where elements can pass through.
    """
    def __init__(self,mask):
        self.mask = mask

    def __call__(self,model, input, output):
        # Repeat along batch dimension
        if output.shape != self.mask.shape:
            batch_size = output.shape[0]
            mask = self.mask.unsqueeze(0).repeat([batch_size] + [1 for _ in range(len(self.mask.shape))])
            output = torch.where(torch.isnan(mask), output, mask) 
        else:
            output = torch.where(torch.isnan(self.mask), output, self.mask)
        
        return output
