



from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from config import FreezeLayersAvailable
import matplotlib.pyplot as plt


class FreezeLayers(BaseFinetuning):
    
    
    def __init__(self,freeze_layer_enum):
        if freeze_layer_enum==FreezeLayersAvailable.freeze_all_except_last:
            self._unfreeze_at_epoch = 1000 #number so high mean never unfreeze
        elif freeze_layer_enum==FreezeLayersAvailable.freeze_all_except_last_to_epoch25:
            self._unfreeze_at_epoch = 25
            
        super().__init__()

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing ``feature_extractor``
        all_layers=list(pl_module.model.children())
        all_layers_except_last=all_layers[:-1]
        self.freeze(all_layers_except_last)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` is 10, feature_extractor will start training.
        all_layers=list(pl_module.model.children())
        all_layers_except_last=all_layers[:-1]
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=all_layers_except_last,
                optimizer=optimizer,
                train_bn=True,
            )
    
        

class AccuracyEnd(Callback):
    
    def __init__(self,dataloader:DataLoader,prefix=None) -> None:
        super(AccuracyEnd,self).__init__()
        self.dataloader=dataloader

        
    def generate_accuracy_and_upload(self,trainer: 'pl.Trainer',pl_module:'pl.LightningModule'):
        all_results=[]
        all_targets=[]
        num_correct = 0
        num_samples = 0
        for batch in self.dataloader:
            image,target,idx=batch
            y=target[0].to(pl_module.device)
            with torch.no_grad():
                # results=pl_module(image.to(device=pl_module.device))
                scores = pl_module(image.to(device=pl_module.device))
                _, predictions = scores.max(1)
                num_correct +=torch.sum(predictions == y)
                num_samples += predictions.size(0)
            # all_results.append(results.softmax(dim=1))
            # all_targets.append(target)
        accuracy=num_correct/num_samples
        # accuracy = self.simple_accuracy(all_results, all_targets)
        trainer.logger.experiment.log({
            "Accuracy ":accuracy.item(),
        
                })

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.generate_accuracy_and_upload(trainer,pl_module)
   
        return super().on_train_end(trainer, pl_module)
    
        
    def simple_accuracy(self,preds, labels):
        return (preds == labels).mean()
    
    
