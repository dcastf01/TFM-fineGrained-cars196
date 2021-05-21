
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
        # nn.CrossEntropyLoss()

    def forward(self,x):
        y0=self.model(x)
        
        return y0
    
    def train_val_steps(self,x,targets,split_metrics,split):
        embbeding=self.model.pre_classifier(x)
        y0=self.model.classifier(embbeding)
        loss={}
        for key,criterion in self.criterions.items():
            loss[f"{split}_{key}"]=criterion(embbeding,y0,targets)

        loss_total=sum(loss.values())
        
        preds0_probability=y0.softmax(dim=1)
        
        try:
            metric_value0=split_metrics["level0"](preds0_probability,targets)
            
            data_dict={
                    f"{split}_loss_total":loss_total,
                    **loss,
                    **metric_value0,
                    }
            
            
            self.insert_each_metric_value_into_dict(data_dict,prefix="")
        
        except Exception as e:
            print(e)
            sum_by_batch=torch.sum(preds0_probability,dim=1)
            logging.error("la suma de las predicciones da distintos de 1, procedemos a imprimir el primer elemento")
            print(sum_by_batch)
            print(e)
        
        return loss_total
    def training_step(self,batch,batch_idx):
        
        x,targets=batch
        target0=targets[0]
        loss_total=self.train_val_steps(x,target0,self.train_metrics_base,"train")

        return loss_total
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x,targets=batch
        target0=targets[0]
        loss_total=self.train_val_steps(x,target0,self.valid_metrics_base,"val")
        