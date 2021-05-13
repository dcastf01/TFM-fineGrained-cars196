

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
        
        super().__init__(class_level,lr,optim)
        
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
        predictions:dict=self.HierarchicalTransformers(x)
        loss={}
        metrics={}
        
        for (level, y_pred),y_true in zip(predictions.items(),targets):
                loss[level]=self.criterion(y_pred,y_true)
                preds_probability=y_pred.softmax(dim=1)
                metrics[level]=self.train_metrics_base[level](preds_probability,y_true)
                
        loss_total=sum(loss.values())
        data_dict={
                "loss_total":loss_total,
                **loss,
                **dict(ele for sub in metrics.values() for ele in sub.items()) #remove top level from dictionary
                }

        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        return loss_total
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x,targets=batch
        predictions:dict=self.HierarchicalTransformers(x)
        loss={}
        metrics={}
        
        for (level, y_pred),y_true in zip(predictions.items(),targets):
                loss["_val_loss"+level[5:]]=self.criterion(y_pred,y_true)
                preds_probability=y_pred.softmax(dim=1)
                metrics[level]=self.valid_metrics_base[level](preds_probability,y_true)
                
        loss_total=sum(loss.values())
        data_dict={
                "_val_loss_total":loss_total,
                **loss,
                **dict(ele for sub in metrics.values() for ele in sub.items()) #remove top level from dictionary
                }

        self.insert_each_metric_value_into_dict(data_dict,prefix="")
      
class HierarchicalTransformers(nn.Module):
        def __init__(self,
                     class_level:dict,
                     img_size,
                     ):

                super(HierarchicalTransformers,self).__init__()
                self.model=ViTBase16(class_level=class_level,img_size=img_size)
                
        def forward(self, x):
                return self.model(x)
   
class ViTBase16(nn.Module):

        
        def __init__(self,class_level:dict,img_size:int):
                
                super(ViTBase16,self).__init__()
                self.num_layers_out=len(class_level)
                
                model_name="vit_base_patch16_224_in21k"
                extras=dict(
                    img_size=img_size
                )   
                self.model=timm.create_model(model_name,pretrained=True,**extras)
                
                self.model.heads=nn.ModuleDict(  {
                        level:nn.Linear(self.model.head.in_features,num_classes)
                        
                        for level,num_classes in class_level.items()
                })
                self.model.head=nn.Identity()
                # self.model.head = nn.Linear(self.model.head.in_features, num_classes)
                
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
                y={}
                # prelevel=None
                for level,head in self.model.heads.items():
                        y[level]=head(x)
                #sería un for ya que es dependiente de la jerarquia
                # y0,x0=self.head0(x,"level0")
                # y00,x00=self.head00(x,x0)
                # y000,x000=self.head000(x,x00)
                return y