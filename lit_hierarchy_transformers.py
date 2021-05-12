

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from lit_system import LitSystem
from vit_base import HierarchicalTransformers
import logging
class LitHierarchyTransformers(LitSystem):
    def __init__(self,
                 class_level:dict,
                 optim,
                 lr

                  ):
        
        super().__init__(lr,optim)
        
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        self.HierarchicalTransformers=HierarchicalTransformers(class_level) #Make
         #Model
         #Released year
                
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
      
    


