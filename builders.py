
from config import Dataset,ModelsAvailable,TransformsAvailable
from grocery_store_data_module import GroceryStoreDataModule
from fgvc_aircraft_data_module import FGVCAircraft
from lit_hierarchy_transformers import LitHierarchyTransformers

from lit_vit import LitVIT

import pytorch_lightning as pl
def get_transform_function(transforms:str):
    return TransformsAvailable[transforms.lower()]
    
def get_datamodule(name_dataset:str,batch_size:int,transforms:str):

    if isinstance(name_dataset,str):
        name_dataset=Dataset[name_dataset.lower()]
    
    transform_fn=get_transform_function(transforms)
    
    if name_dataset==Dataset.grocerydataset:
        
        dm=GroceryStoreDataModule(data_dir="data",
                                  batch_size=batch_size,
                                  transform_fn=transform_fn)
        
    elif name_dataset==Dataset.fgvcaircraft:
        dm=FGVCAircraft(data_dir="data",
                        batch_size=batch_size,
                        transform_fn=transform_fn)
    
    else: 
        raise ("choice a correct dataset")
    dm.prepare_data()   
    return dm


def get_system(datamodule:pl.LightningDataModule,model_choice:str,optim:str,lr:float):
    if isinstance(model_choice,str):
        model_choice=ModelsAvailable[model_choice.lower()]
    if model_choice==ModelsAvailable.hierarchicaltransformers:
        model=LitHierarchyTransformers(datamodule.classlevel,optim,lr)
    elif model_choice==ModelsAvailable.vitbaseline:
        model=LitVIT(datamodule.classlevel,optim,lr)
        
    else:
        raise("choise a correct model")
    return model