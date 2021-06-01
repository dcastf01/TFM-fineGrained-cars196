import torch
import os
from enum import Enum

from dataclasses import dataclass,asdict

ROOT_WORKSPACE: str=""
class ArchitectureType(Enum):
    standar="models normal"
    hierarchical="loss_con_hierarchical"
    api_model="api_model"
class ModelsAvailable(Enum):
    hierarchicaltransformers="version_con_multiples_heads"
    vit_base_patch16_224_in21k="vit_base_patch16_224_in21k"
    vit_base_patch16_384="vit_base_patch16_384"
    vit_base_patch16_224_in21k_overlap="vit_base_patch16_224_in21k-overlap"
    resnet50="resnet50"
    resnet101="resnet101"
    vit_large_patch16_224_in21k="vit_large_patch16_224_in21k"
    vit_large_patch32_224_in21k="vit_large_patch32_224_in21k"
    vit_base_patch16_224_miil_in21k="vit_base_patch16_224_miil_in21k"
    tf_efficientnet_b4_ns="tf_efficientnet_b4_ns"
    tf_efficientnet_b5_ns="tf_efficientnet_b5_ns"
    densenet121="densenet121"
    densenet169="densenet169"
    
class Dataset (Enum):
    grocerydataset=1
    fgvcaircraft=2
    cars196=3
class Optim(Enum):
    adam=1
    sgd=2

class TransformsAvailable(Enum):
    cars_train_transforms_basic=1
    timm_noaug=2
    timm_transforms_imagenet_train=3
    transforms_imagenet_eval=4
    cars_train_transfroms_autoaugment=5
    cars_transforms_eval=6
    cars_only_mixup=7
    cars_autoaugment_mixup=8

class CollateAvailable(Enum):
    none=1
    collate_two_images=2
    mixup=3
    collate_to_triplet_loss=4

class FreezeLayersAvailable(Enum):
    none=1
    freeze_all_except_last=2
    freeze_all_except_last_to_epoch25=3

@dataclass
class LossActive:
    
    crossentropy:bool=False
    contrastive:bool=False
    triplet_loss:bool=False
    similitud_loss:bool=False

def create_config_dict(instance):
    return asdict(instance)
@dataclass
class CONFIG(object):
    
    experiment=ModelsAvailable.tf_efficientnet_b4_ns
    experiment_name:str=experiment.name

    architecture =ArchitectureType.standar
    architecture_name:str=architecture.name
    
    loss_crossentropy_standar:bool=True
    loss_contrastive_fg:bool=False
    loss_contrastive_standar:bool=False
    loss_cosine_similarity_simsiam:bool=False
    loss_triplet:bool=False
    
    freeze_layers =FreezeLayersAvailable.none
    freeze_layers_name:str=freeze_layers.name
    # experiment_net:str=experiment.value
    PRETRAINED_MODEL:bool=True
    
    #torch config
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # TRAIN_DIR = "data/train"
    # VAL_DIR = "data/val"
    batch_size:int =50
    
    dataset=Dataset.cars196
    dataset_name:str=dataset.name
    precision_compute:int=16
    
    optim=Optim.sgd
    optim_name:str=optim.name
    transform=TransformsAvailable.cars_train_transforms_basic
    transform_name:str=transform.name
    
    collate_fn=CollateAvailable.none
    collate_fn_name:str=collate_fn.name
    
    transform_to_test=TransformsAvailable.cars_transforms_eval
    transform_to_test:str=transform_to_test.name
    
    lr:float = 0.0035
    AUTO_LR :bool= False
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 4
    SEED:int=1
    IMG_SIZE:int=448
    NUM_EPOCHS :int= 50
    # LOAD_MODEL :bool= True
    # SAVE_MODEL :bool= True
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"classification/model/checkpoint")
    
    gpu0:bool=False
    gpu1:bool=True
    notes:str="TFM experiment2"
    ignore_globs:str="*.ckpt"
    
    #hyperparameters
    

