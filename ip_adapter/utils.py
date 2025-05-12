import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

attn_maps = {}
def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            module.register_forward_hook(hook_fn(name))

    return unet

def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0)
    attn_map = attn_map.permute(2,0,1)
    # temp_size = None

    # for i in range(0,5):
    #     scale = 2 ** i
    #     if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
    #         temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
    #         break

    # assert temp_size is not None, "temp_size cannot is None"

    # attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode='bilinear',
        align_corners=False
    )[0]

    attn_map = torch.softmax(attn_map, dim=0)
    return attn_map
def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):

    idx = 0 if instance_or_negative else 1
    net_attn_maps_0 = []
    net_attn_maps_1 = []
    net_attn_maps_bg = []
    for name, attn_map_combine in attn_maps.items():
        attn_map_chunk = attn_map_combine.chunk(3, dim=0)
        attn_map_0 = attn_map_chunk[0]
        attn_map_1 = attn_map_chunk[1]
        attn_map_bg = attn_map_chunk[2]

        attn_map_0 = attn_map_0.cpu() if detach else attn_map_0
        attn_map_0 = torch.chunk(attn_map_0, batch_size)[idx].squeeze()
        attn_map_0 = upscale(attn_map_0, image_size) 
        net_attn_maps_0.append(attn_map_0) 

        attn_map_1 = attn_map_1.cpu() if detach else attn_map_1
        attn_map_1 = torch.chunk(attn_map_1, batch_size)[idx].squeeze()
        attn_map_1 = upscale(attn_map_1, image_size) 
        net_attn_maps_1.append(attn_map_1) 

        attn_map_bg = attn_map_bg.cpu() if detach else attn_map_bg
        attn_map_bg = torch.chunk(attn_map_bg, batch_size)[idx].squeeze()
        attn_map_bg = upscale(attn_map_bg, image_size) 
        net_attn_maps_bg.append(attn_map_bg) 

    net_attn_maps_0 = torch.mean(torch.stack(net_attn_maps_0,dim=0),dim=0)
    net_attn_maps_1 = torch.mean(torch.stack(net_attn_maps_1,dim=0),dim=0)
    net_attn_maps_bg = torch.mean(torch.stack(net_attn_maps_bg,dim=0),dim=0)

    return net_attn_maps_0, net_attn_maps_1, net_attn_maps_bg

def attnmaps2images(net_attn_maps):

    #total_attn_scores = 0
    images = []

    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        #total_attn_scores += attn_map.mean().item()

        normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        #print("norm: ", normalized_attn_map.shape)
        image = Image.fromarray(normalized_attn_map)

        #image = fix_save_attn_map(attn_map)
        images.append(image)

    #print(total_attn_scores)
    return images
def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator