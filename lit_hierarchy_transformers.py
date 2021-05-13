

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from lit_system import LitSystem
import logging
import timm
class LitHierarchyTransformers(LitSystem):
    def __init__(self,
                 class_level:dict,
                 optim,
                 lr,
                img_size,
                  ):
        
        super().__init__(lr,optim)
        
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        
        self.HierarchicalTransformers=HierarchicalTransformers(class_level,img_size) #Make
        
                
        self.criterion=F.cross_entropy

        # a=self.HierarchicalTransformers(torch.rand(2,3,224,224))
        # print(a)

    def forward(self,x):
        y0,y00=self.HierarchicalTransformers(x)
        
        return y0,y00

    def training_step(self,batch,batch_idx):
        x,targets=batch
        target0,target00,target000=targets
        y0,y00,y000=self.HierarchicalTransformers(x)
        loss0=self.criterion(y0,target0)
        loss00=self.criterion(y00,target00)
        loss000=self.criterion(y000,target000)

        loss_total=loss0+loss00+loss000
        
        preds0_probability=y0.softmax(dim=1)
        preds00_probability=y00.softmax(dim=1)
        preds000_probability=y000.softmax(dim=1)
        
        metric_value0=self.train_metrics_base0(preds0_probability,target0)
        metric_value00=self.train_metrics_base00(preds00_probability,target00)
        metric_value000=self.train_metrics_base000(preds000_probability,target000)
        
        data_dict={"loss0":loss0,"loss00":loss00,"loss000":loss000,
                   "loss_total":loss_total,
                   **metric_value0,**metric_value00,**metric_value000}
        
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        return loss_total
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x,targets=batch
        target0,target00,target000=targets
        y0,y00,y000=self.HierarchicalTransformers(x)
        loss0=self.criterion(y0,target0)
        loss00=self.criterion(y00,target00)
        loss000=self.criterion(y000,target000)

        loss_total=loss0+loss00+loss000
        
        preds0_probability=y0.softmax(dim=1)
        preds00_probability=y00.softmax(dim=1)
        preds000_probability=y000.softmax(dim=1)
        metric_value0=self.valid_metrics_base0(preds0_probability,target0)
        metric_value00=self.valid_metrics_base00(preds00_probability,target00)
        metric_value000=self.valid_metrics_base000(preds000_probability,target000)
        data_dict={"val_loss0":loss0,"val_loss00":loss00, "val_loss000":loss000,
                   "val_loss_total":loss_total,
                   **metric_value0,**metric_value00,**metric_value000}
        
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
      
class HierarchicalTransformers(nn.Module):
        def __init__(self,
                     class_level:dict,
                     img_size,
                     ):
                #ver como hacen lo del block y hacerlo de esa forma
                class_level0=class_level["level0"]
                class_level00=class_level["level00"]
                class_level000=class_level["level000"]
                super(HierarchicalTransformers,self).__init__()
                self.vit1=ViTBase16(num_classes=class_level0,img_size=img_size)
                self.vit2=ViTBase16(num_classes=class_level00,img_size=img_size)
                self.vit3=ViTBase16(num_classes=class_level000,img_size=img_size)
                ##test
                # self.vit1(torch.rand(2,3,224,224),
                #           torch.rand(2,1,768))
                
        def forward(self, x):
                y0,x0=self.vit1(x)
                y00,x00=self.vit2(x,x0)
                y000,x000=self.vit3(x,x00)
                
                return y0,y00,y000
      
class ViTBase16(nn.Module):

        
        def __init__(self,num_classes,img_size):
                super(ViTBase16,self).__init__()
                model_name="vit_base_patch16_224_in21k"
                extras=dict(
                    img_size=img_size
                )   
                self.model=timm.create_model(model_name,pretrained=True,**extras)
                self.model.head = nn.Linear(self.model.head.in_features, num_classes)
                self.model.forward=self.forward
                self.model.forward_features=self.forward_features


        def forward_features(self,x,extra_tensor=None):
                B = x.shape[0]
                x = self.model.patch_embed(x)
                
                if extra_tensor is not None:
                        cls_tokens_from_previuos_net=torch.unsqueeze(extra_tensor,1)
                        x=torch.cat((cls_tokens_from_previuos_net,x),dim=1)
                else:
                        cls_tokens = self.model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                        x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.model.pos_embed
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