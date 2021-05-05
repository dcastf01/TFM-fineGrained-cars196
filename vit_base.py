import torch.nn as nn
import timm

import torch

class HierarchicalTransformers(nn.Module):
        def __init__(self,class_level0,class_level1,class_level2):
                super(HierarchicalTransformers,self).__init__()
                self.vit1=ViTBase16(num_classes=class_level0)
                self.vit2=ViTBase16(num_classes=class_level1)
                self.vit3=ViTBase16(num_classes=class_level2)
                ##test
                # self.vit1(torch.rand(2,3,224,224),
                #           torch.rand(2,1,768))
                
        def forward(self, x):
                y0,x0=self.vit1(x)
                y00,x00=self.vit2(x,x0)
                y000,x000=self.vit3(x,x00)
                
                return y0,y00,y000
                
                
                
class ViTBase16(nn.Module):

        
        def __init__(self,num_classes=20):
                super(ViTBase16,self).__init__()
                model_name="vit_base_patch16_224_in21k"
                self.model=timm.create_model(model_name,pretrained=True)
                self.model.head = nn.Linear(self.model.head.in_features, num_classes)
                self.model.forward=self.forward
                self.model.forward_features=self.forward_features


        def forward_features(self,x,extra_tensor=None):
                B = x.shape[0]
                x = self.model.patch_embed(x)
                cls_tokens = self.model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                x = torch.cat((cls_tokens, x), dim=1)

                x = x + self.model.pos_embed
                if extra_tensor is not None:
                        cls_tokens_from_previuos_net=torch.unsqueeze(extra_tensor,1)
                        x=torch.cat((cls_tokens_from_previuos_net,x),dim=1)
                x = self.model.pos_drop(x)

                for blk in self.model.blocks:
                        x = blk(x)

                x = self.model.norm(x)[:, 0]
                x = self.model.pre_logits(x)
                return x

        def forward(self, x,extra_tensor=None):
                x = self.forward_features(x,extra_tensor)
                y = self.model.head(x)
                return y,x
    
# HierarchicalTransformers()