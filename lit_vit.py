
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from lit_system import LitSystem
import timm
import logging
class LitVIT(LitSystem):
    def __init__(self,
                 class_level:dict,
                 optim,
                 lr
                  ):
        
        super().__init__(lr,optim)
        num_classes=class_level["level000"]
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        model_name="vit_base_patch16_224_in21k"
        self.model=timm.create_model(model_name,pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
         #Model
         #Released year
                
        self.criterion=F.cross_entropy

    def forward(self,x):
        y000=self.model(x)
        
        return y000

    def training_step(self,batch,batch_idx):
        
        x,targets=batch
        target000=targets[-1]
        y000=self.model(x)

        loss000=self.criterion(y000,target000)

        loss_total=loss000
        
        preds000_probability=y000.softmax(dim=1)
        try:
            metric_value000=self.train_metrics_base000(preds000_probability,target000)
            
            data_dict={"loss000":loss000,
                    "loss_total":loss_total,
                        **metric_value000}
            
            
            self.insert_each_metric_value_into_dict(data_dict,prefix="")
        except:
            sum_by_batch=torch.sum(preds000_probability,dim=1)
            logging.error("la suma de las predicciones da distintos de 1, procedemos a imprimir el primer elemento")
            print(sum_by_batch)
                
     
            
        return loss_total
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x,targets=batch
        target000=targets[-1]
        y000=self.model(x)

        loss000=self.criterion(y000,target000)

        loss_total=loss000
        
        preds000_probability=y000.softmax(dim=1)
        try:
            metric_value000=self.valid_metrics_base000(preds000_probability,target000)
            data_dict={ "val_loss000":loss000,
                    "val_loss_total":loss_total,
                    **metric_value000}
        
            self.insert_each_metric_value_into_dict(data_dict,prefix="")
            
        except:
            sum_by_batch=torch.sum(preds000_probability,dim=1)
            logging.error("la suma de las predicciones da distintos de 1, procedemos a imprimir el primer elemento")
            print(sum_by_batch)