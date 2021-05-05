import torch
import os
from enum import Enum
from typing import Union

from dataclasses import dataclass,asdict

ROOT_WORKSPACE: str=""

class ModelsAvailable(Enum):
    ViTBase16="vit_base_patch16_224_in21k"
    ResNet50="resnet50"
    
class Dataset (Enum):
    grocerydataset=1
class Optim(Enum):
    adam=1
    sgd=2


@dataclass
class CONFIG(object):
    
    experiment=ModelsAvailable.ResNet50
    experiment_name:str=experiment.name
    experiment_net:str=experiment.value
    PRETRAINED_MODEL:bool=True
    transfer_learning:bool=True #solo se entrena el head
    #torch config
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # TRAIN_DIR = "data/train"
    # VAL_DIR = "data/val"
    batch_size:int = 5
    dataset=Dataset.grocerydataset
    dataset_name:str=dataset.name
    
    optim=Optim.sgd
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
    USE_TRIPLETLOSS:bool=False
    

    class DATASET:
        
        
        class COMPCAR:
            
            #dataset compcar
            NUM_CLASSES:int=4455
            PASSWORD_ZIP: str="d89551fd190e38"
            PATH_ROOT: str=os.path.join(ROOT_WORKSPACE,"data","compcars")
            PATH_CSV: str=os.path.join("dataset","compcars","all_information_compcars.csv")
            PATH_IMAGES: str=os.path.join(PATH_ROOT,"image")
            PATH_LABELS: str=os.path.join(PATH_ROOT,"label")
            PATH_TRAIN_REVISITED: str=os.path.join(PATH_ROOT,"CompCars_revisited_v1.0","bbs_train.txt")
            PATH_TEST_REVISITED: str=os.path.join(PATH_ROOT,"CompCars_revisited_v1.0","bbs_test.txt")
            PATH_MAKE_MODEL_NAME: str=os.path.join(PATH_ROOT,"misc","make_model_name.mat")
            PATH_MAKE_MODEL_NAME_CLS: str=os.path.join(PATH_ROOT,"misc","make_model_names_cls.mat")
            COMPCAR_CONDITION_FILTER: str='viewpoint=="4" or viewpoint=="1"'
        class CARS196:
            #dataset cars196
            NUM_CLASSES:int=196
            PATH_ROOT:str= os.path.join(ROOT_WORKSPACE,"data","cars196")
            PATH_CSV: str=os.path.join("dataset","cars196","all_information_cars196.csv")
            PATH_IMAGES:str=os.path.join(PATH_ROOT,)
            PATH_LABELS:str=os.path.join(PATH_ROOT,)
            PATH_TRAIN:str=os.path.join(PATH_ROOT,)
            PATH_TEST:str=os.path.join(PATH_ROOT,)
            # PATH_CARS196_IMAGES=r"dataset\compcars\all_information_compcars.csv"
            # PATH_CARS196_LABELS=r"dataset\compcars\all_information_compcars.csv"

        #output plots
        PATH_OUTPUT_PLOTS: str=os.path.join("dataset","results")
    

def create_config_dict(instance:CONFIG):
    return asdict(instance)

