
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
from layers import _projection_mlp,_prediction_mlp
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
        self.model=create_model(model_name,img_size,num_classes,pretrained)
                
        self.criterions=criterions
        if "similarity" in self.criterions.keys():
            self.is_similitud_loss=True
            num_ftrs: int = 2048
            proj_hidden_dim: int = 2048
            pred_hidden_dim: int = 512
            out_dim: int = 2048
            num_mlp_layers: int = 3
            #esto iria en la parte de create_model pero por ahora aquí se queda
            self.projection_mlp = \
                _projection_mlp(num_ftrs, proj_hidden_dim, out_dim, num_mlp_layers)

            self.prediction_mlp = \
                _prediction_mlp(out_dim, pred_hidden_dim, out_dim)
        else:
            self.is_similitud_loss=False
        # nn.CrossEntropyLoss()

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
                z0 = self.projection_mlp(f0)
                p0 = self.prediction_mlp(z0)
                out0=(z0,p0)
                f1 = f1.flatten(start_dim=1)
                z1 = self.projection_mlp(f1)
                p1 = self.prediction_mlp(z1)
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
    
    def calculate_loss_total(self,loss:dict,split:str):
        loss_total=sum(loss.values())
        data_dict={
                    f"{split}_loss_total":loss_total,
                    **loss,
                    }
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        return loss_total
    
    def calculate_metrics(self,y0,target,split_metric,):

        preds0_probability=y0.softmax(dim=1)
            
        try:
            data_dict=split_metric["level0"](preds0_probability,target)
            self.insert_each_metric_value_into_dict(data_dict,prefix="")
    
        except Exception as e:
            print(e)
            sum_by_batch=torch.sum(preds0_probability,dim=1)
            logging.error("la suma de las predicciones da distintos de 1, procedemos a imprimir el primer elemento")
            print(sum_by_batch)
            print(e)
        
    def training_step(self,batch,batch_idx):
        
        images,targets,_=batch
        
        loss_total=self.train_val_steps(images,targets,self.train_metrics_base,"train")

        return loss_total
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        images,targets,_=batch

        loss_total=self.train_val_steps(images,targets,self.valid_metrics_base,"val")

            