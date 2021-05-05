

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from lit_system import LitSystem
from vit_base import HierarchicalTransformers
import logging
class LitHierarchyTransformers(LitSystem):
    def __init__(self,
                 #class_level:dict,

                  ):
        
        super().__init__()
        
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        self.HierarchicalTransformers=HierarchicalTransformers() #Make
         #Model
         #Released year
                
        self.criterion=F.cross_entropy

        a=self.HierarchicalTransformers(torch.rand(2,3,224,224))
        print(a)

    def forward(self,x):
        y0,y00,y000=self.HierarchicalTransformers(x)
        
        return x 
    
    
    def training_step(self,batch,batch_idx):
        x,targets=batch
        preds=self.HierarchicalTransformers(x)
        loss=self.criterion(preds,targets)

        preds_probability=nn.functional.softmax(preds,dim=1)
        metric_value=self.train_metrics_base(preds_probability,targets)
        data_dict={"loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        return loss
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, targets = batch   
        preds=self.model(x)
        loss=self.criterion(preds,targets)

        preds_probability=nn.functional.softmax(preds,dim=1)
        metric_value=self.valid_metrics_base(preds_probability,targets)
        data_dict={"val_loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
      
    


LitHierarchyTransformers()