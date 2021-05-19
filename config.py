from timm.data.transforms_factory import transforms_imagenet_train
from timm.models import resnest
import torch
import os
from enum import Enum
from typing import Union

from dataclasses import dataclass,asdict

ROOT_WORKSPACE: str=""
class ArchitectureType(Enum):
    standar="models normal"
    hierarchical="loss_con_hierarchical"
    constrastive="contrastive"
class ModelsAvailable(Enum):
    hierarchicaltransformers="version_con_multiples_heads"
    vit_base_patch16_224_in21k="vit_base_patch16_224_in21k"
    vit_base_patch16_224_in21k_overlap="vit_base_patch16_224_in21k"
    resnet50="resnet50"
    vit_large_patch16_224_in21k="vit_large_patch16_224_in21k"
    vit_large_patch32_224_in21k="vit_large_patch32_224_in21k"
    
class Dataset (Enum):
    grocerydataset=1
    fgvcaircraft=2
    cars196=3
class Optim(Enum):
    adam=1
    sgd=2

class TransformsAvailable(Enum):
    basic_transforms=1
    timm_noaug=2
    timm_transforms_imagenet_train=3
    transforms_imagenet_eval=4
    cars_transfroms_transfg=5

class losstype(Enum):
    
    crossentropy="crossentropy"
    contrastive="contrastive"
    triplet_loss="triplet_loss"
    similitud_loss="similitud"
    
@dataclass
class CONFIG(object):
    
    experiment=ModelsAvailable.vit_base_patch16_224_in21k_overlap
    experiment_name:str=experiment.name
    architecture =ArchitectureType.standar
    architecture_name:str=architecture.name
    # experiment_net:str=experiment.value
    PRETRAINED_MODEL:bool=False
    transfer_learning:bool=False #solo se entrena el head ##no funcional en este repositorio aun
    #torch config
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # TRAIN_DIR = "data/train"
    # VAL_DIR = "data/val"
    batch_size:int =16
    dataset=Dataset.cars196
    dataset_name:str=dataset.name
    precision_compute:int=16
    
    optim=Optim.adam
    optim_name:str=optim.name
    transform=TransformsAvailable.cars_transfroms_transfg
    transform_name:str=transform.name
    
    transform_to_test=TransformsAvailable.transforms_imagenet_eval
    transform_to_test:str=transform_to_test.name
    
    lr:float = 3e-4
    AUTO_LR :bool= False
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 4
    SEED:int=1
    IMG_SIZE:int=448
    NUM_EPOCHS :int= 30
    # LOAD_MODEL :bool= True
    # SAVE_MODEL :bool= True
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"classification/model/checkpoint")
    
    gpu0:bool=True
    gpu1:bool=False
    notes:str=" correr diferentes modelos para probar su funcionamiento"
    ignore_globs:str="*.ckpt"
    
    #hyperparameters
    
def create_config_dict(instance:CONFIG):
    return asdict(instance)

