
import logging
from typing import Optional

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import ModelsAvailable
from lit_system import LitSystem
from factory_model import create_model
from lightly.loss import SymNegCosineSimilarityLoss

class LitGeneralModellevel0(LitSystem):
    def __init__(self,
                 model_name:ModelsAvailable,
                 criterions:dict,
                 class_level:dict,
                 optim:str,
                 lr:float,
                 img_size:int,
                 pretrained:bool,
                 epoch:Optional[int]=None,
                 steps_per_epoch:Optional[int]=None, #len(train_loader)
                  ):
        
        fine_class={"level0":class_level["level0"]}
        super().__init__(fine_class,
                         lr,
                         optim,
                         epoch=epoch,
                         steps_per_epoch=steps_per_epoch)
        num_classes=class_level["level0"]
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        
                
        self.criterions=criterions
        if "similarity" in self.criterions.keys():
            self.is_similitud_loss=True
            
        else:
            self.is_similitud_loss=False
        # nn.CrossEntropyLoss()
        self.model=create_model(model_name,
                                img_size,
                                num_classes,
                                pretrained,
                                self.is_similitud_loss                        
                                )
    def forward(self,x):
        y0=self.model(x)
        
        return y0
    
    def train_val_steps(self,images,targets,split_metric,split):

        if isinstance(images,list):
            f=[]
            target=[]
            loss={}
            for x in images:
                tensorstargets_split_in_list=torch.chunk(targets, 3, 1)
                target0=torch.squeeze(tensorstargets_split_in_list[0],dim=1)
                loss,embbeding,y0=self.step_loss(x,target0,split)   
                f.append(embbeding)
                target.append(target0)
                #revisar lighly
                self.calculate_metrics(y0,target0,split_metric)
            if self.is_similitud_loss:
                f0,f1=f
                f0 = f0.flatten(start_dim=1)
                z0 = self.model.projection_mlp(f0)
                p0 = self.model.prediction_mlp(z0)
                out0=(z0,p0)
                f1 = f1.flatten(start_dim=1)
                z1 = self.model.projection_mlp(f1)
                p1 = self.model.prediction_mlp(z1)
                out1=(z1,p1)
                loss[f"{split}_similarity"]=self.criterions["similarity"](out0,out1)
        
        else:
            target0=targets[0]
            loss,embbeding,y0=self.step_loss(images,target0,split)
            self.calculate_metrics(y0,target0,split_metric)
            
        loss_total=self.calculate_loss_total(loss,split)
        
        return loss_total
    
    def step_loss(self,x,targets,split):   
        embbeding=self.model.pre_classifier(x)

        y0=self.model.classifier(embbeding)
        loss={}
        for key,criterion in self.criterions.items():
            if not isinstance(criterion,SymNegCosineSimilarityLoss):
                loss[f"{split}_{key}"]=criterion(embbeding,y0,targets)
        
        return loss,embbeding,y0
    
 
    def training_step(self,batch,batch_idx):
        
        images,targets,_=batch
        
        loss_total=self.train_val_steps(images,targets,self.train_metrics_base,"train")

        return loss_total
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        images,targets,_=batch

        loss_total=self.train_val_steps(images,targets,self.valid_metrics_base,"val")

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        images,targets,_=batch

        loss_total=self.train_val_steps(images,targets,self.valid_metrics_base,"test")