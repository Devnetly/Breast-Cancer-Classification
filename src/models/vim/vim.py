import sys
import os
import timm
import dotenv
import torch
sys.path.append('../..')
from timm.models.registry import register_model

try:
    from Vim.vim.models_mamba import *
except:
    print("Error importing models_mamba")

env = dotenv.find_dotenv()
VIM_WEIGHTS_FOLDER = dotenv.get_key(env,"VIM_WEIGHTS_FOLDER")

@register_model
def vim_tiny(num_classes: int,pretrained : bool = True,**args):
    
    model = timm.create_model(
        model_name="vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2",
        pretrained=False,
        num_classes=num_classes,
    )

    if pretrained:
        state_dict_path = os.path.join(VIM_WEIGHTS_FOLDER,"vim_tiny","vim_t_midclstok_76p1acc.pth")
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict,strict=False)

    return model