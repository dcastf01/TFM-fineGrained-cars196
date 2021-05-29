import torch.nn as nn
import torch.nn.functional as F

from config import ModelsAvailable
import timm
import torch

import types
from models.layers import _projection_mlp,_prediction_mlp,PatchEmbed
from models.trasnfg import CONFIGS as vit_fg_config   
from models.trasnfg import VisionTransformer
import numpy as np


def create_model(model_chosen:ModelsAvailable,
                img_size:int,
                num_classes:int,
                pretrained:bool,
                is_loss_similarity_necessary:bool=False,
                    ):
            
            prefix_name=model_chosen.name[0:3]
            if prefix_name==ModelsAvailable.vit_large_patch16_224_in21k.name[0:3]:
                model_name=model_chosen.value.split("-")[0]
                overlap=model_chosen.name.split("_")[-1]
                if overlap=="overlap":
                    config=vit_fg_config["ViT-B_16"]
                    
                    model=VisionTransformer(config,img_size,zero_head=True,num_classes=num_classes)
                    if pretrained:
                        pretrained_dir="/home/dcast/Hierarchy-label-transformers/models/ViT-B_16.npz"
                        model.load_from(np.load(pretrained_dir))
                    # if pretrained_dir is not None:
                    #     a=torch.load(pretrained_dir)
                    #     pretrained_model = torch.load(pretrained_dir)['model']
                    #     model.load_state_dict(pretrained_model)
                    model.pre_classifier=model.transformer
                    model.classifier=model.part_head
                    model.transformer.forward=types.MethodType(forward_trasnformer_FG,model.transformer)
                else:
                    extras=dict(
                    img_size=img_size
                    )
                    
                    model=timm.create_model(model_name,pretrained=pretrained,num_classes=num_classes,**extras)
                
                    model.pre_classifier=model.forward_features
                    model.classifier=model.head
                    
                if is_loss_similarity_necessary:
                    num_ftrs: int = 768
                    proj_hidden_dim: int = 768
                    pred_hidden_dim: int = 512
                    out_dim: int = 768
                    num_mlp_layers: int = 3
                    model=create_layers_to_similitud_loss(model,num_ftrs,
                                                    proj_hidden_dim,pred_hidden_dim,out_dim,
                                                    num_mlp_layers)
                
            elif prefix_name==ModelsAvailable.resnet50.name[0:3]:
                
                model=timm.create_model(model_chosen.value,pretrained=pretrained,num_classes=num_classes)
                model.pre_classifier=types.MethodType(standar_timm_forward_features,model)
                model.classifier=model.fc
                if is_loss_similarity_necessary:
                    num_ftrs: int = 2048
                    proj_hidden_dim: int = 2048
                    pred_hidden_dim: int = 512
                    out_dim: int = 2048
                    num_mlp_layers: int = 3
                    model=create_layers_to_similitud_loss(model,num_ftrs,
                                                    proj_hidden_dim,pred_hidden_dim,out_dim,
                                                    num_mlp_layers)    
            
            elif prefix_name==ModelsAvailable.tf_efficientnet_b4_ns.name[0:3]:
                model=timm.create_model(model_chosen.value,pretrained=pretrained,num_classes=num_classes)
                model.pre_classifier=types.MethodType(standar_timm_forward_features,model)
                model.classifier=model.classifier
                if is_loss_similarity_necessary:
                    num_ftrs: int = 1280
                    proj_hidden_dim: int = 1280
                    pred_hidden_dim: int = 512
                    out_dim: int = 1280
                    num_mlp_layers: int = 3
                    model=create_layers_to_similitud_loss(model,num_ftrs,
                                                    proj_hidden_dim,pred_hidden_dim,out_dim,
                                                    num_mlp_layers)
            
            elif prefix_name==ModelsAvailable.densenet121.name[0:3]:
                model=timm.create_model(model_chosen.value,pretrained=pretrained,num_classes=num_classes)
                model.pre_classifier=types.MethodType(standar_timm_forward_features,model)
                model.classifier=model.classifier
            
            return model

def create_layers_to_similitud_loss(model,
                                    num_ftrs: int = 2048,
                                    proj_hidden_dim: int = 2048,
                                    pred_hidden_dim: int = 512,
                                    out_dim: int = 2048,
                                    num_mlp_layers: int = 3,
                ):


    model.projection_mlp = \
        _projection_mlp(num_ftrs, proj_hidden_dim, out_dim, num_mlp_layers)

    model.prediction_mlp = \
        _prediction_mlp(out_dim, pred_hidden_dim, out_dim)
    return model

def standar_timm_forward_features(self,x):
    x=self.forward_features(x)
    x = self.global_pool(x)
    if self.drop_rate:
        x = F.dropout(x, p=float(self.drop_rate), training=self.training)
    return x

def standar_timm_classifier(self,x):
    
    x = self.fc(x)
    return x

def forward_trasnformer_FG(self,input_ids):
    embedding_output = self.embeddings(input_ids)
    part_encoded = self.encoder(embedding_output)
    return part_encoded[:, 0]
