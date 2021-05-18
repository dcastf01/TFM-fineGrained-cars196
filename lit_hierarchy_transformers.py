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
                embbeding=torch.squeeze(self.projection(pre_embbeding),dim=1)

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
                loss_total=total_loss_crossentropy+total_loss_contrastive
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
                embbeding=torch.squeeze(self.projection(pre_embbeding),dim=1)
                # loss=self.supcons_los(F.normalize(embbeding),targets[-1])
                # a=self.con_loss(embbeding,targets[-1])
                loss={}
                metrics={}
                
                loss_crossentropy={}
                loss_contrastive={}
                metrics={}
                
                for (level, y_pred),y_true in zip(predictions.items(),targets):
                        
                        loss_crossentropy["val_loss_crosentropy"+level[5:]]=\
                                self.apply_loss_correction_hierarchical(loss=self.criterion(y_pred,y_true),
                                                                        level=level
                                )
                        
                        loss_contrastive["val_loss_contrastive"+level[5:]]=\
                                self.apply_loss_correction_hierarchical(loss=self.contrastive_loss(embbeding,y_true),
                                                                        level=level
                                )
                        preds_probability=y_pred.softmax(dim=1)
                        metrics[level]=self.valid_metrics_base[level](preds_probability,y_true)

                total_loss_crossentropy=sum(loss_crossentropy.values())/len(loss_crossentropy)
                total_loss_contrastive=sum(loss_contrastive.values())/len(loss_contrastive)
                loss_total=total_loss_crossentropy+total_loss_contrastive
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
                     img_size:int,
                     overlap:bool=None,
                     
                     ):

                super(HierarchicalTransformers,self).__init__()
                # self.encoder=ViTBase16(class_level=class_level,img_size=img_size)
                
                model_name=model_chosen.value
                overlap=True
                if model_chosen.name[0:3]==ModelsAvailable.vit_large_patch16_224_in21k.name[0:3]:
                        extras=dict(
                        img_size=img_size,
                        # embed_layer=
                        
                        )   
                        self.encoder=timm.create_model(model_name,pretrained=True,**extras)
                        
                        if overlap is not None:
                                self.encoder.patch_embed=PatchEmbed(
                                        img_size=img_size,
                                        exist_overlap=True,
                                        slide_step=12)

                                self.encoder.pos_embed = nn.Parameter(torch.zeros(1,
                                                                                  self.encoder.patch_embed.num_patches + 1,
                                                                                  768))

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


from timm.models.layers.helpers import to_2tuple
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None,exist_overlap:bool=False,slide_step=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        if exist_overlap:
                self.num_patches = ((img_size[0] - patch_size[0]) // slide_step + 1) * ((img_size[1] - patch_size[1]) // slide_step + 1)
                self.proj = nn.Conv2d(in_channels=in_chans,
                                        out_channels=embed_dim,
                                        kernel_size=patch_size,
                                        stride=(slide_step, slide_step))
        else:
                
                self.patch_size = patch_size
                self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
                self.num_patches = self.grid_size[0] * self.grid_size[1]

                self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
                                      
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x