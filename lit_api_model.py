
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
from models.api_models import API_Net

class LitApi(LitSystem):
    
    
    def __init__(self,
                 model_name:ModelsAvailable,
                 criterions:dict,
                 class_level:dict,
                 optim:str,
                 lr:float,
                 img_size:int,
                 pretrained:bool,
                 epoch:Optional[int]=None,
                 steps_per_epoch:Optional[int]=None,
                  ):
        

        fine_class={"level0":class_level["level0"]}
        super().__init__(fine_class,
                         lr,
                         optim,
                         epoch=epoch,
                         steps_per_epoch=steps_per_epoch)
        num_classes=class_level["level0"]
        #puede que loss_fn no vaya aquí y aquí solo vaya modelo
        
                
        self.criterions=criterions
        if "similarity" in self.criterions.keys():
            self.is_similitud_loss=True
            
        else:
            self.is_similitud_loss=False
        # nn.CrossEntropyLoss()
        self.backbone=create_model(model_name,
                                img_size,
                                num_classes,
                                pretrained,
                                self.is_similitud_loss                        
                                )
           
        self.model=API_Net(self.backbone)
        del self.backbone
        self.criterion=nn.CrossEntropyLoss()
        self.rank_criterion = nn.MarginRankingLoss(margin=0.05)
        self.softmax_layer = nn.Softmax(dim=1)
    
    def train_val_steps(self,images,targets,split_metric,split):
        
        NotImplementedError

    def training_step(self,batch,batch_idx):
        
        images,targets_hierarchical,_=batch
        targets=targets_hierarchical[0]
        loss={}
        # loss_total=self.train_val_steps(images,targets,self.train_metrics_base,"train")
        
        logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2 = self.model(images, targets, flag="train")
        batch_size = logit1_self.shape[0]
        
        self_logits = torch.zeros(2*batch_size, 200).to(self.device)
        other_logits= torch.zeros(2*batch_size, 200).to(self.device)
        self_logits[:batch_size] = logit1_self
        self_logits[batch_size:] = logit2_self
        other_logits[:batch_size] = logit1_other
        other_logits[batch_size:] = logit2_other

        # compute loss
        logits = torch.cat([self_logits, other_logits], dim=0)
        targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)
        softmax_loss = self.criterion(logits, targets)

        self_scores = self.softmax_layer(self_logits)[torch.arange(2*batch_size).to(self.device),
                                                         torch.cat([labels1, labels2], dim=0)]
        other_scores = self.softmax_layer(other_logits)[torch.arange(2*batch_size).to(self.device),
                                                         torch.cat([labels1, labels2], dim=0)]
        flag = torch.ones([2*batch_size, ]).to(self.device)
        rank_loss = self.rank_criterion(self_scores, other_scores, flag)
        loss["train_softmax_loss"]=softmax_loss
        loss["train_rank_loss"]=rank_loss
        # loss_total = softmax_loss + rank_loss
        
        self.calculate_metrics(logits,targets,self.train_metrics_base)
        loss_total=self.calculate_loss_total(loss,"train")
        return loss_total
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        images,targets_hierarchical,_=batch
        targets=targets_hierarchical[0]
        loss={}
        target_var = targets.squeeze()

        # compute output
        logits = self.model(images, targets=None, flag='val')
        softmax_loss =self.criterion(logits, target_var)
        loss["val_softmax_loss"]=softmax_loss
        self.calculate_metrics(logits,target_var,self.valid_metrics_base)
        self.calculate_loss_total(loss,"val")
        
    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        images,targets_hierarchical,_=batch
        targets=targets_hierarchical[0]
        loss={}
        target_var = targets.squeeze()

        # compute output
        logits = self.model(images, targets=None, flag='val')
        softmax_loss =self.criterion(logits, target_var)
        loss["val_softmax_loss"]=softmax_loss
        self.calculate_metrics(logits,target_var,self.valid_metrics_base)
        self.calculate_loss_total(loss,"val")