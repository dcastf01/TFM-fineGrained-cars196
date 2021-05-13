
import pytorch_lightning as pl
import torch
import torch.nn as nn
from config import Optim
from metrics import get_metrics_collections_base


class LitSystem(pl.LightningModule):
    def __init__(self,
                 class_level,
                  lr:float=0.01,
                  optim:str="adam",
                  ):
        
        super().__init__()

        # metrics_base=
        level="0"
        self.train_metrics_base=nn.ModuleDict()
        self.valid_metrics_base=nn.ModuleDict()
        for _ in range(len(class_level)):
            self.train_metrics_base["level"+level]=get_metrics_collections_base(prefix="train_level"+level)
            self.valid_metrics_base["level"+level]=get_metrics_collections_base(prefix="valid_level"+level)
            level+="0"

            
        
        # self.train_metrics_base0=metrics_base.clone(prefix="train_level0")
        # self.valid_metrics_base0=metrics_base.clone(prefix="valid_level0")
        # self.train_metrics_base00=metrics_base.clone(prefix="train_level00")
        # self.valid_metrics_base00=metrics_base.clone(prefix="valid_level00")
        # self.train_metrics_base000=metrics_base.clone(prefix="train_level000")
        # self.valid_metrics_base000=metrics_base.clone(prefix="valid_level000")
        
        # log hyperparameters
        self.save_hyperparameters()    
        self.lr=lr
        if isinstance(optim,str):
            self.optim=Optim[optim.lower()]
           
    
    def on_epoch_start(self):
        torch.cuda.empty_cache()
            
    def configure_optimizers(self):
        if self.optim==Optim.adam:
            optimizer= torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim==Optim.sgd:
            optimizer= torch.optim.SGD(self.parameters(), lr=self.lr,momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(15,100,10),gamma=0.95)
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