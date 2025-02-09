import torch

@torch.no_grad()
def add_feature_on_text(sae, feature_idx, steering_feature, module, input, output):
    ## input shape 
    if input[0].size(-1) == 768:
        return (output[0] + steering_feature[:,:768].unsqueeze(0)),
    else:
        return (output[0] + steering_feature[:,768:].unsqueeze(0)),

@torch.no_grad()
def add_feature_on_text_prompt(sae, steering_feature, module, input, output):
    if input[0].size(-1) == 768:
        return (output[0] + steering_feature[:,:768].unsqueeze(0)),
    else:
        return (output[0] + steering_feature[:,768:].unsqueeze(0)),

@torch.no_grad()
def add_feature_on_text_prompt_flux(sae, steering_feature, module, input, output):

    return (output[0] + steering_feature.unsqueeze(0)), output[1]

@torch.no_grad()
def minus_feature_on_text_prompt(sae, steering_feature, module, input, output):
    if input[0].size(-1) == 768:
        return (output[0] - steering_feature[:,:768].unsqueeze(0)),
    else:
        return (output[0] - steering_feature[:,768:].unsqueeze(0)),

@torch.no_grad()
def do_nothing(sae, steering_feature, module, input, output):
    return (output[0]),

