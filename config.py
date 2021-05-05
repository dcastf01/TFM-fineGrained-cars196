import torch
import os
from enum import Enum
from typing import Union

from dataclasses import dataclass,asdict

ROOT_WORKSPACE: str=""

class ModelsAvailable(Enum):
    HierarchicalTransformers=1
    
class Dataset (Enum):
    grocerydataset=1
    fgvcaircraft=2
class Optim(Enum):
    adam=1
    sgd=2


@dataclass
class CONFIG(object):
    
    experiment=ModelsAvailable.HierarchicalTransformers
    experiment_name:str=experiment.name
    experiment_net:str=experiment.value
    PRETRAINED_MODEL:bool=True
    transfer_learning:bool=True #solo se entrena el head
    #torch config
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # TRAIN_DIR = "data/train"
    # VAL_DIR = "data/val"
    batch_size:int =1
    dataset=Dataset.fgvcaircraft
    dataset_name:str=dataset.name
    
    optim=Optim.adam
    optim_name:str=optim.name
    lr:float = 1e-3
    AUTO_LR :bool= False
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 4
    SEED:int=1
    IMG_SIZE:int=224
    NUM_EPOCHS :int= 50
    LOAD_MODEL :bool= True
    SAVE_MODEL :bool= True
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"classification/model/checkpoint")
    

def create_config_dict(instance:CONFIG):
    return asdict(instance)

