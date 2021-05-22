



from pytorch_lightning.callbacks.base import Callback
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import wandb
import math
import matplotlib.pyplot as plt
class AccuracyEnd(Callback):
    
    def __init__(self,dataloader:DataLoader,prefix=None) -> None:
        super(AccuracyEnd,self).__init__()
        self.dataloader=dataloader

        
    def generate_accuracy_and_upload(self,trainer: 'pl.Trainer',pl_module:'pl.LightningModule'):
        all_results=[]
        all_targets=[]
        for batch in self.dataloader:
            image,target,idx=batch
            with torch.no_grad():
                results=pl_module(image.to(device=pl_module.device))
            all_results.append(results.softmax(dim=1))
            all_targets.append(target)
            
        accuracy = self.simple_accuracy(all_results, all_targets)
        trainer.logger.experiment.log({
            "Accuracy "+accuracy,
        
                })

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.generate_accuracy_and_upload(trainer,pl_module)
   
        return super().on_train_end(trainer, pl_module)
    
        
    def simple_accuracy(self,preds, labels):
        return (preds == labels).mean()