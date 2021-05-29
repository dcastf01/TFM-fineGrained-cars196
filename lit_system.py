
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from config import Optim
from metrics import get_metrics_collections_base

import logging
class LitSystem(pl.LightningModule):
    def __init__(self,
                 class_level,
                  lr:float=0.01,
                  optim:str="adam",
                  epoch:Optional[int]=None,
                  steps_per_epoch:Optional[int]=None, #len(train_loader)
                  ):
        
        super().__init__()

        # metrics_base=
        self.train_metrics_base=nn.ModuleDict()
        self.valid_metrics_base=nn.ModuleDict()
        for key,value in class_level.items():
            self.train_metrics_base[key]=get_metrics_collections_base(value,prefix="train_"+key)
            self.valid_metrics_base[key]=get_metrics_collections_base(value,prefix="valid_"+key)

        # log hyperparameters
        self.save_hyperparameters()    
        self.lr=lr
        self.epochs=epoch
        self.steps_per_epoch=steps_per_epoch
        if isinstance(optim,str):
            self.optim=Optim[optim.lower()]
        
    
    def on_epoch_start(self):
        # torch.cuda.empty_cache()
        pass
            
    def configure_optimizers(self):
        if self.optim==Optim.adam:
            optimizer= torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim==Optim.sgd:
            optimizer= torch.optim.SGD(self.parameters(), lr=self.lr,momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(15,100,10),gamma=0.95)
        scheduler=WarmupCosineSchedule(optimizer,warmup_steps=int(self.epochs*0.1),t_total=self.epochs)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        #                                                  max_lr=self.lr, steps_per_epoch=self.steps_per_epoch,
        #                                 epochs=self.epochs, pct_start=0.2, cycle_momentum=False, div_factor=20)
        return [optimizer], [scheduler]

    def insert_each_metric_value_into_dict(self,data_dict:dict,prefix:str):
 
        on_step=False
        on_epoch=True 
        
        for metric,value in data_dict.items():
            if metric != "preds":
                
                self.log("_".join([prefix,metric]),value,
                        on_step=on_step, 
                        on_epoch=on_epoch, 
                        logger=True
                )
                
    def add_prefix_into_dict_only_loss(self,data_dict:dict,prefix:str=""):
        data_dict_aux={}
        for k,v in data_dict.items():            
            data_dict_aux["_".join([prefix,k])]=v
            
        return data_dict_aux
    
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
            
    def calculate_loss_total(self,loss:dict,split:str):
        loss_total=sum(loss.values())
        data_dict={
                    f"{split}_loss_total":loss_total,
                    **loss,
                    }
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        return loss_total
import math

from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
