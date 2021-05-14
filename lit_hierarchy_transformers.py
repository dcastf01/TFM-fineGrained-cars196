import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from lit_system import LitSystem
import logging
import timm
class LitHierarchyTransformers(LitSystem):
        def __init__(self,
                        class_level:dict,
                        optim,
                        lr,
                        img_size,
                        ):
                
                super().__init__(class_level,lr,optim)
                
                self.HierarchicalTransformers=HierarchicalTransformers(class_level,img_size) 
                        
                self.criterion=F.cross_entropy

        def forward(self,x):
                y0,y00=self.HierarchicalTransformers(x)
                
                return y0,y00

        def training_step(self,batch,batch_idx):
                x,targets=batch
                embbeding,predictions=self.HierarchicalTransformers(x)
                # a=self.con_loss(embbeding,targets[-1])
                loss={}
                metrics={}
                # a=len(targets)
                for (level, y_pred),y_true in zip(predictions.items(),targets):
                        
                        loss["loss"+level[5:]]=self.criterion(y_pred,y_true)
                        preds_probability=y_pred.softmax(dim=1)

                        metrics[level]=self.train_metrics_base[level](preds_probability,y_true)
                        
                loss_total=sum(loss.values())
                data_dict={
                        "loss_total":loss_total,
                        **loss,
                        **dict(ele for sub in metrics.values() for ele in sub.items()) #remove top level from dictionary
                        }

                self.insert_each_metric_value_into_dict(data_dict,prefix="")
                return loss_total
        
        def validation_step(self, batch, batch_idx):
                '''used for logging metrics'''
                x,targets=batch
                embbeding,predictions=self.HierarchicalTransformers(x)
                # a=self.con_loss(embbeding,targets[-1])
                loss={}
                metrics={}
                
                for (level, y_pred),y_true in zip(predictions.items(),targets):
                        loss["val_loss"+level[5:]]=self.criterion(y_pred,y_true)
                        preds_probability=y_pred.softmax(dim=1)
                        metrics[level]=self.valid_metrics_base[level](preds_probability,y_true)
                        
                loss_total=sum(loss.values())
                data_dict={
                        "val_loss_total":loss_total,
                        **loss,
                        **dict(ele for sub in metrics.values() for ele in sub.items()) #remove top level from dictionary
                        }

                self.insert_each_metric_value_into_dict(data_dict,prefix="")
        
        
        def con_loss(self,features, labels):
                B, _ = features.shape
                features = F.normalize(features)
                cos_matrix = features.mm(features.t())
                pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
                neg_label_matrix = 1 - pos_label_matrix
                pos_cos_matrix = 1 - cos_matrix
                neg_cos_matrix = cos_matrix - 0.4
                neg_cos_matrix[neg_cos_matrix < 0] = 0
                loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
                loss /= (B * B)
                return loss
class HierarchicalTransformers(nn.Module):
        def __init__(self,
                     class_level:dict,
                     img_size,
                     ):

                super(HierarchicalTransformers,self).__init__()
                # self.encoder=ViTBase16(class_level=class_level,img_size=img_size)
                
                model_name="vit_base_patch16_224_in21k"
                extras=dict(
                    img_size=img_size
                )   
                self.encoder=timm.create_model(model_name,pretrained=True,**extras)

                self.heads=nn.ModuleDict(  {
                        level:nn.Linear(self.encoder.head.in_features,num_classes)
                        for level,num_classes in class_level.items()
                })
                self.encoder.head=nn.Identity()
                
                self.encoder.forward_features=self.forward_features
                
        def forward(self, x,extra_tensor=None):
                x = self.forward_features(x,extra_tensor)
                # x=self.encoder(x)
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
