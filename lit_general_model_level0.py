
import logging
from typing import Optional

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import TripletMarginLoss

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
        if any([ isinstance(criterion,SymNegCosineSimilarityLoss) for criterion in self.criterions.values()]):
        # if "similarity" in self.criterions.keys():
            self.have_similitud_loss=True
            
        else:
            self.have_similitud_loss=False
            
        if any([ isinstance(criterion,TripletMarginLoss) for criterion in self.criterions.values()]):
            self.have_triplet_loss=True
        else:
            self.have_triplet_loss=False

        is_projection_embbeding_necessary=any([self.have_similitud_loss,self.have_triplet_loss])

        # if isinstance(criterion,TripletMarginLoss)
        # nn.CrossEntropyLoss()
        self.model=create_model(model_name,
                                img_size,
                                num_classes,
                                pretrained,
                                is_projection_embbeding_necessary
                                )
    def forward(self,x):
        y0=self.model(x)
        
        return y0
    
    def train_val_steps(self,images,targets,split_metric,split):
        loss={}
        if isinstance(images,list):
            embbedings=[]
            for x in images:
                embbeding=self.get_embbeding(x)
                embbedings.append(embbeding)
            
            tensorstargets_split_in_list=torch.chunk(targets, 3, 1)
            target0=torch.squeeze(tensorstargets_split_in_list[0],dim=1)
            
            loss,y0=self.step_loss(embbedings[0],target0,split,loss)   
            
            # target.append(target0)
            #revisar lighly
            self.calculate_metrics(y0,target0,split_metric)
            if self.have_similitud_loss:
                self.train_similitud_loss(embbedings,loss,split)
            if self.have_triplet_loss:
                self.train_triplet_loss(embbedings,loss,split)
                
        else:
            target0=targets[0]
            embbeding=self.get_embbeding(images)
            loss,y0=self.step_loss(embbeding,target0,split,loss)
            self.calculate_metrics(y0,target0,split_metric)
            
        loss_total=self.calculate_loss_total(loss,split)
        
        return loss_total
    def get_embbeding(self,x):
        embbeding=self.model.pre_classifier(x)
        return embbeding
    def step_loss(self,embbeding,targets,split,loss):   
        y0=self.model.classifier(embbeding)
    
        for key,criterion in self.criterions.items():
            if isinstance(criterion,SymNegCosineSimilarityLoss) or isinstance(criterion,TripletMarginLoss):
                pass
            else:
                loss[f"{split}_{key}"]=criterion(embbeding,y0,targets)
        
        return loss,y0
    
 
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
        
    
    def train_similitud_loss(self,f,loss,split="train"):
        f0,f1=f[:1]
        f0 = f0.flatten(start_dim=1)
        z0 = self.model.projection_mlp(f0)
        p0 = self.model.prediction_mlp(z0)
        out0=(z0,p0)
        f1 = f1.flatten(start_dim=1)
        z1 = self.model.projection_mlp(f1)
        p1 = self.model.prediction_mlp(z1)
        out1=(z1,p1)
        loss[f"{split}_similarity"]=self.criterions["similarity"](out0,out1)
        return loss
    
    def train_triplet_loss(self,f,loss,split="train"):
        a,p,n=f
        a=self.model.projection_mlp(a)
        a=self.model.prediction_mlp(a)
        p=self.model.projection_mlp(p)
        n=self.model.projection_mlp(n)
        loss[f"{split}_triplet_loss"]=self.criterions["triplet_loss"](a,p,n)
        return loss