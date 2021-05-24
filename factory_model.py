import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers.helpers import to_2tuple
from config import ModelsAvailable
import timm
import torch

import types
from models.layers import _projection_mlp,_prediction_mlp
from models.trasnfg import CONFIGS as vit_fg_config




#esto iria en la parte de create_model pero por ahora aquí se queda

                
                
def create_model(model_chosen:ModelsAvailable,
                img_size:int,
                num_classes:int,
                pretrained:bool,
                is_loss_similarity_necessary:bool=False,
                    ):
            
            prefix_name=model_chosen.name[0:3]
            if prefix_name==ModelsAvailable.vit_large_patch16_224_in21k.name[0:3]:
                
                extras=dict(
                img_size=img_size
                )
                model_name=model_chosen.value.split("-")[0]
                model=timm.create_model(model_name,pretrained=pretrained,num_classes=num_classes,**extras)
                
                overlap=model_chosen.name.split("_")[-1]
                if overlap=="overlap":
                                model.patch_embed=PatchEmbed(
                                        img_size=img_size,
                                        exist_overlap=True,
                                        slide_step=12)
                                model.pos_embed = nn.Parameter(torch.zeros(1,
                                                                            model.patch_embed.num_patches + 1,
                                                                            768))

                model.pre_classifier=model.forward_features
                model.classifier=model.head
                if is_loss_similarity_necessary:
                    num_ftrs: int = 768
                    proj_hidden_dim: int = 768
                    pred_hidden_dim: int = 512
                    out_dim: int = 768
                    num_mlp_layers: int = 3
                    create_layers_to_similitud_loss(model,num_ftrs,
                                                    proj_hidden_dim,pred_hidden_dim,out_dim,
                                                    num_mlp_layers)
                
            elif prefix_name==ModelsAvailable.resnet50.name[0:3]:
                
                model=timm.create_model(model_chosen.value,pretrained=pretrained,num_classes=num_classes)
                model.pre_classifier=types.MethodType(resnet_forward_features,model)
                model.classifier=model.fc
                if is_loss_similarity_necessary:
                    num_ftrs: int = 2048
                    proj_hidden_dim: int = 2048
                    pred_hidden_dim: int = 512
                    out_dim: int = 2048
                    num_mlp_layers: int = 3
                    create_layers_to_similitud_loss(model,num_ftrs,
                                                    proj_hidden_dim,pred_hidden_dim,out_dim,
                                                    num_mlp_layers)    
            
            return model

def create_layers_to_similitud_loss(model,
                                    num_ftrs: int = 2048,
                                    proj_hidden_dim: int = 2048,
                                    pred_hidden_dim: int = 512,
                                    out_dim: int = 2048,
                                    num_mlp_layers: int = 3,
                ):


    model.projection_mlp = \
        _projection_mlp(num_ftrs, proj_hidden_dim, out_dim, num_mlp_layers)

    model.prediction_mlp = \
        _prediction_mlp(out_dim, pred_hidden_dim, out_dim)
    return model

def resnet_forward_features(self,x):
    x=self.forward_features(x)
    x = self.global_pool(x)
    if self.drop_rate:
        x = F.dropout(x, p=float(self.drop_rate), training=self.training)
    return x

def resnet_classifier(self,x):
    
    x = self.fc(x)
    return x
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None,exist_overlap:bool=False,slide_step=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        if exist_overlap:
                self.num_patches = ((img_size[0] - patch_size[0]) // slide_step + 1) * ((img_size[1] - patch_size[1]) // slide_step + 1)
                self.proj = nn.Conv2d(in_channels=in_chans,
                                        out_channels=embed_dim,
                                        kernel_size=patch_size,
                                        stride=(slide_step, slide_step))
        else:
                
                self.patch_size = patch_size
                self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
                self.num_patches = self.grid_size[0] * self.grid_size[1]

                self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
                                      
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
    