
import logging
from typing import Optional

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import ModelsAvailable
from lit_system import LitSystem


class LitGeneralModellevel0(LitSystem):
    def __init__(self,
                 model_name:ModelsAvailable,
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
        self.model=self.create_model(model_name,img_size,num_classes,pretrained)
                
        self.criterion=nn.CrossEntropyLoss()

    def forward(self,x):
        y000=self.model(x)
        
        return y000

    def training_step(self,batch,batch_idx):
        
        x,targets=batch
        target0=targets[0]
        y0=self.model(x)

        loss0=self.criterion(y0,target0)

        loss_total=loss0
        
        preds0_probability=y0.softmax(dim=1)
        try:
            metric_value0=self.train_metrics_base["level0"](preds0_probability,target0)
            
            data_dict={"loss0":loss0,
                    "loss_total":loss_total,
                        **metric_value0}
            
            
            self.insert_each_metric_value_into_dict(data_dict,prefix="")
        except Exception as e:
            print(e)
            sum_by_batch=torch.sum(preds0_probability,dim=1)
            logging.error("la suma de las predicciones da distintos de 1, procedemos a imprimir el primer elemento")
            print(sum_by_batch)
            print(e)
     
            
        return loss_total
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x,targets=batch
        target0=targets[0]
        y0=self.model(x)

        loss0=self.criterion(y0,target0)

        loss_total=loss0
        
        preds0_probability=y0.softmax(dim=1)
        try:
            metric_value0=self.valid_metrics_base["level0"](preds0_probability,target0)
            data_dict={ "val_loss0":loss0,
                    "val_loss_total":loss_total,
                    **metric_value0}
        
            self.insert_each_metric_value_into_dict(data_dict,prefix="")
            
        except Exception as e:
            print(e)
            sum_by_batch=torch.sum(preds0_probability,dim=1)
            logging.error("la suma de las predicciones da distintos de 1, procedemos a imprimir el primer elemento")
            print(sum_by_batch)
            
            
            
    def create_model(self,model_chosen:ModelsAvailable
                     ,img_size,
                     num_classes,
                     pretrained):
            
            prefix_name=model_chosen.name[0:3]
            if prefix_name==ModelsAvailable.vit_large_patch16_224_in21k.name[0:3]:
                
                extras=dict(
                img_size=img_size
                )
                
                model=timm.create_model(model_chosen.value,pretrained=True,num_classes=num_classes,**extras)
            elif model_chosen==ModelsAvailable.resnet50:
                
                model=timm.create_model(model_chosen.value,pretrained=True,num_classes=num_classes)
   
            return model
