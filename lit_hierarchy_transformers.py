import logging
from typing import Optional

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import ModelsAvailable
from lit_system import LitSystem
from losses import ContrastiveLossFG
from factory_model import PatchEmbed

class LitHierarchyTransformers(LitSystem):
        def __init__(self,
                        model_name:ModelsAvailable,
                        class_level:dict,
                        optim,
                        lr,
                        img_size,
                        pretrained,
                        epoch:Optional[int]=None,
                        steps_per_epoch:Optional[int]=None,
                        ):
                
                super().__init__(class_level,
                         lr,
                         optim,
                         epoch=epoch,
                         steps_per_epoch=steps_per_epoch)
                
                self.HierarchicalTransformers=HierarchicalTransformers(model_name,class_level,img_size,
                                                                       pretrained=pretrained) 
                        
                self.criterion=nn.CrossEntropyLoss()
                self.contrastive_loss=ContrastiveLossFG()
                self.projection=nn.Linear(self.HierarchicalTransformers.out_features,128)
                self.weights={
                        'level0':0.7,
                        'level00': 0.2,
                        'level000':0.1
                }

        def forward(self,x):
                embbeding,predictions=self.HierarchicalTransformers(x)
                
                return embbeding,predictions

        
        def training_step(self,batch,batch_idx):
                x,targets=batch
                pre_embbeding,predictions=self.HierarchicalTransformers(x)
                embbeding=torch.squeeze(self.projection(pre_embbeding),dim=1)
                loss_total=self.train_val_steps(predictions,embbeding,targets,self.train_metrics_base)

                return loss_total
        
        def validation_step(self, batch, batch_idx):
                '''used for logging metrics'''
                x,targets=batch
                pre_embbeding,predictions=self.HierarchicalTransformers(x)
                embbeding=torch.squeeze(self.projection(pre_embbeding),dim=1)
                loss_total=self.train_val_steps(predictions,embbeding,targets,self.valid_metrics_base)
        
        def train_val_steps(self,predictions,embbeding,targets,split_metrics,split):
                #falta el split
                loss_crossentropy={}
                loss_contrastive={}
                metrics={}
                for (level, y_pred),y_true in zip(predictions.items(),targets):
                        
                        loss_crossentropy[f"{split}_loss_crosentropy"+level[5:]]=\
                                self.apply_loss_correction_hierarchical(loss=self.criterion(y_pred,y_true),
                                                                        level=level
                                )
                        
                        loss_contrastive[f"{split}_loss_contrastive"+level[5:]]=\
                                self.apply_loss_correction_hierarchical(loss=self.contrastive_loss(embbeding,y_true),
                                                                        level=level
                                )
                                
                        preds_probability=y_pred.softmax(dim=1)
                        metrics[level]=split_metrics[level](preds_probability,y_true)

                total_loss_crossentropy=sum(loss_crossentropy.values())
                total_loss_contrastive=sum(loss_contrastive.values())
                loss_total=total_loss_crossentropy+total_loss_contrastive
                data_dict={
                        f"{split}_loss_crossentropy":total_loss_crossentropy,
                        f"{split}_loss_contrastive":total_loss_contrastive,
                        f"{split}_loss_total":loss_total,            
                        **loss_crossentropy,
                        **loss_contrastive,
                        **dict(ele for sub in metrics.values() for ele in sub.items()) #remove top level from dictionary
                        }

                self.insert_each_metric_value_into_dict(data_dict,prefix="")
                return loss_total
        
        def apply_loss_correction_hierarchical(self,loss,level):
                
                return loss*self.weights[level]
                
class HierarchicalTransformers(nn.Module):
        def __init__(self,
                     model_chosen:ModelsAvailable,
                     class_level:dict,
                     img_size:int,
                     overlap:bool=None,
                     pretrained:bool=True
                     
                     ):

                super(HierarchicalTransformers,self).__init__()
                # self.encoder=ViTBase16(class_level=class_level,img_size=img_size)
                
                model_name=model_chosen.value
                overlap=True
                if model_chosen.name[0:3]==ModelsAvailable.vit_large_patch16_224_in21k.name[0:3]:
                        extras=dict(
                        img_size=img_size,
                        # embed_layer=
                        
                        )   
                        self.encoder=timm.create_model(model_name,pretrained=True,**extras)
                        
                        if overlap is not None:
                                self.encoder.patch_embed=PatchEmbed(
                                        img_size=img_size,
                                        exist_overlap=True,
                                        slide_step=12)

                                self.encoder.pos_embed = nn.Parameter(torch.zeros(1,
                                                                                  self.encoder.patch_embed.num_patches + 1,
                                                                                  768))

                        self.heads=nn.ModuleDict(  {
                                level:nn.Linear(self.encoder.head.in_features,num_classes)
                                for level,num_classes in class_level.items()
                        })
                        self.out_features=self.encoder.head.in_features
                        self.encoder.head=nn.Identity()
                else:
                        self.encoder=timm.create_model(model_name,pretrained=pretrained)
                        self.heads=nn.ModuleDict(  {
                                level:nn.Linear(self.encoder.fc.in_features,num_classes)
                                for level,num_classes in class_level.items()
                        })
                        self.out_features=self.encoder.fc.in_features
                        self.encoder.fc=nn.Identity()
            
     
                # self.encoder.forward_features=self.forward_features
                
        def forward(self, x):
                x=self.encoder(x)
                y={}
                # prelevel=None
                for level,head in self.heads.items():
                        y[level]=head(x)
                return x,y





