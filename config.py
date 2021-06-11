import torch
import os
from enum import Enum

from dataclasses import dataclass

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
    tf_efficientnet_b7_ns="tf_efficientnet_b7_ns"
    
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
    cars_train_transforms_autoaugment_and_imagenet=9 #probar en experimento final
class CollateAvailable(Enum):
    none=1
    collate_two_images=2
    mixup=3
    collate_to_triplet_loss=4

class FreezeLayersAvailable(Enum):
    none=1
    freeze_all_except_last=2
    freeze_all_except_last_to_epoch25=3
class LossDifferentExperimentsAvailable(Enum):
    custom=0
    only_crossentropy=1
    only_contrastivefg=2
    only_triplet_loss=3
    only_similitud_loss=4
    crossentropy_and_contrastivefg=5
    crossentropy_and_triplet=6
    crossentropy_and_similitud=7

class SchedulerAvailable(Enum):
    constant=1
    cosine_descent=2
    lineal_ascent_cosine_descent=3
    
@dataclass(init=True)
class CONFIG(object):
    
    experiment=ModelsAvailable.tf_efficientnet_b7_ns
    experiment_name:str=experiment.name

    architecture =ArchitectureType.standar
    architecture_name:str=architecture.name
    
    #always in false
    # loss_crossentropy_standar:bool=False
    # loss_contrastive_fg:bool=False
    # loss_contrastive_standar:bool=False
    # loss_cosine_similarity_simsiam:bool=False
    # loss_triplet:bool=False
            
    loss_experiment_config=LossDifferentExperimentsAvailable.only_crossentropy
    loss_experiment_config_name:str=loss_experiment_config.name
    
    freeze_layers =FreezeLayersAvailable.none
    freeze_layers_name:str=freeze_layers.name
    PRETRAINED_MODEL:bool=True
    
    #torch config
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size:int =20
    
    dataset=Dataset.cars196
    dataset_name:str=dataset.name
    precision_compute:int=16
    
    optim=Optim.sgd
    optim_name:str=optim.name
    scheduler=SchedulerAvailable.lineal_ascent_cosine_descent
    scheduler_name:str=scheduler.name
    
    transform=TransformsAvailable.cars_train_transforms_autoaugment_and_imagenet
    transform_name:str=transform.name
    
    collate_fn=CollateAvailable.none
    collate_fn_name:str=collate_fn.name
    
    transform_to_test=TransformsAvailable.cars_transforms_eval
    transform_to_test:str=transform_to_test.name
    
    lr:float = 0.1
    AUTO_LR :bool= False
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 4
    SEED:int=1
    IMG_SIZE:int=448
    NUM_EPOCHS :int= 100
    # LOAD_MODEL :bool= True
    # SAVE_MODEL :bool= True
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"classification/model/checkpoint")
    
    gpu0:bool=False
    gpu1:bool=True
    notes:str="obteniendo espacio latente"
    ignore_globs:str="*.ckpt"
     
    #hyperparameters
    
