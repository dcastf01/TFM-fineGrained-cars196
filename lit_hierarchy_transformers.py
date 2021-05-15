import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from lit_system import LitSystem
import logging
import timm
from config import ModelsAvailable
from losses import ContrastiveLoss
class LitHierarchyTransformers(LitSystem):
        def __init__(self,
                     model_name:ModelsAvailable,
                        class_level:dict,
                        optim,
                        lr,
                        img_size,
                        ):
                
                super().__init__(class_level,lr,optim)
                
                self.HierarchicalTransformers=HierarchicalTransformers(model_name,class_level,img_size) 
                        
                self.criterion=nn.CrossEntropyLoss()
                self.contrastive_loss=ContrastiveLoss()
                self.projection=nn.Linear(self.HierarchicalTransformers.out_features,128)
                self.weights={
                        'level0':0.7,
                        'level00': 0.2,
                        'level000':0.1
                }

        def forward(self,x):
                embbeding,predictions=self.HierarchicalTransformers(x)
                
                return embbeding,predictions

        def training_step(self,batch,batch_idx):
                x,targets=batch
                pre_embbeding,predictions=self.HierarchicalTransformers(x)
                embbeding=torch.squeeze(self.projection(pre_embbeding))

                # loss=self.supcons_los(F.normalize(embbeding),targets[-1])
                loss_crossentropy={}
                loss_contrastive={}
                metrics={}
                
                for (level, y_pred),y_true in zip(predictions.items(),targets):
                        
                        loss_crossentropy["loss_crosentropy"+level[5:]]=\
                                self.apply_loss_correction_hierarchical(loss=self.criterion(y_pred,y_true),
                                                                        level=level
                                )
                        
                        loss_contrastive["loss_contrastive"+level[5:]]=\
                                self.apply_loss_correction_hierarchical(loss=self.contrastive_loss(embbeding,y_true),
                                                                        level=level
                                )
                                
                        preds_probability=y_pred.softmax(dim=1)
                        metrics[level]=self.train_metrics_base[level](preds_probability,y_true)

                total_loss_crossentropy=sum(loss_crossentropy.values())/len(loss_crossentropy)
                total_loss_contrastive=sum(loss_contrastive.values())/len(loss_contrastive)
                loss_total=(total_loss_crossentropy+total_loss_contrastive)/2
                data_dict={
                        "loss_crossentropy":total_loss_crossentropy,
                        "loss_contrastive":total_loss_contrastive,
                        "loss_total":loss_total,
                        
                        **loss_crossentropy,
                        **loss_contrastive,
                        **dict(ele for sub in metrics.values() for ele in sub.items()) #remove top level from dictionary
                        }

                self.insert_each_metric_value_into_dict(data_dict,prefix="")
                return loss_total
        
        def validation_step(self, batch, batch_idx):
                '''used for logging metrics'''
                x,targets=batch
                pre_embbeding,predictions=self.HierarchicalTransformers(x)
                embbeding=torch.squeeze(self.projection(pre_embbeding))
                # loss=self.supcons_los(F.normalize(embbeding),targets[-1])
                # a=self.con_loss(embbeding,targets[-1])
                loss={}
                metrics={}
                
                loss_crossentropy={}
                loss_contrastive={}
                metrics={}
                
                for (level, y_pred),y_true in zip(predictions.items(),targets):
                        
                        loss_crossentropy["loss_crosentropy"+level[5:]]=\
                                self.apply_loss_correction_hierarchical(loss=self.criterion(y_pred,y_true),
                                                                        level=level
                                )
                        
                        loss_contrastive["loss_contrastive"+level[5:]]=\
                                self.apply_loss_correction_hierarchical(loss=self.contrastive_loss(embbeding,y_true),
                                                                        level=level
                                )
                        preds_probability=y_pred.softmax(dim=1)
                        metrics[level]=self.valid_metrics_base[level](preds_probability,y_true)

                total_loss_crossentropy=sum(loss_crossentropy.values())/len(loss_crossentropy)
                total_loss_contrastive=sum(loss_contrastive.values())/len(loss_contrastive)
                loss_total=(total_loss_crossentropy+total_loss_contrastive)/2
                data_dict={
                        "val_loss_crossentropy":total_loss_crossentropy,
                        "val_loss_contrastive":total_loss_contrastive,
                        "val_loss_total":loss_total,
                        
                        **loss_crossentropy,
                        **loss_contrastive,
                        **dict(ele for sub in metrics.values() for ele in sub.items()) #remove top level from dictionary
                        }

                self.insert_each_metric_value_into_dict(data_dict,prefix="")
        
        def apply_loss_correction_hierarchical(self,loss,level):
                
                return loss*self.weights[level]
                
class HierarchicalTransformers(nn.Module):
        def __init__(self,
                     model_chosen:ModelsAvailable,
                     class_level:dict,
                     img_size,
                     ):

                super(HierarchicalTransformers,self).__init__()
                # self.encoder=ViTBase16(class_level=class_level,img_size=img_size)
                
                model_name=model_chosen.value
               
                if model_chosen.name[0:3]==ModelsAvailable.vit_large_patch16_224_in21k.name[0:3]:
                        extras=dict(
                        img_size=img_size
                        )   
                        self.encoder=timm.create_model(model_name,pretrained=True,**extras)
                        

                        self.heads=nn.ModuleDict(  {
                                level:nn.Linear(self.encoder.head.in_features,num_classes)
                                for level,num_classes in class_level.items()
                        })
                        self.out_features=self.encoder.head.in_features
                        self.encoder.head=nn.Identity()
                else:
                        self.encoder=timm.create_model(model_name,pretrained=True)
                        self.heads=nn.ModuleDict(  {
                                level:nn.Linear(self.encoder.fc.in_features,num_classes)
                                for level,num_classes in class_level.items()
                        })
                        self.out_features=self.encoder.fc.in_features
                        self.encoder.fc=nn.Identity()
            
     
                # self.encoder.forward_features=self.forward_features
                
        def forward(self, x):
                x=self.encoder(x)
                y={}
                # prelevel=None
                for level,head in self.heads.items():
                        y[level]=head(x)
                return x,y

        def forward_features(self,x,extra_tensor=None):
                B = x.shape[0]
                x = self.encoder.patch_embed(x)
                
                if extra_tensor is not None:
                        cls_tokens_from_previuos_net=torch.unsqueeze(extra_tensor,1)
                        x=torch.cat((cls_tokens_from_previuos_net,x),dim=1)
                else:
                        cls_tokens = self.encoder.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                        x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.encoder.pos_embed
                x = self.encoder.pos_drop(x)

                for blk in self.encoder.blocks:
                        x = blk(x)

                x = self.encoder.norm(x)[:, 0]
                x = self.encoder.pre_logits(x)
                return x
