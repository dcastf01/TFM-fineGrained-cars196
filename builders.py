
import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from config import Dataset, ModelsAvailable, TransformsAvailable
from factory_augmentations import (basic_transforms, transforms_imagenet_eval,
                                   transforms_imagenet_train,
                                   transforms_noaug_train)
from fgvc_aircraft_data_module import FGVCAircraft
from grocery_store_data_module import GroceryStoreDataModule
from lit_general_model_level0 import LitGeneralModellevel0
from lit_hierarchy_transformers import LitHierarchyTransformers
from PIL import Image

def get_transform_function(transforms:str,img_size:int):
    name_transform=TransformsAvailable[transforms.lower()]
    
    if name_transform==TransformsAvailable.basic_transforms:
        transform_fn=basic_transforms(img_size=img_size)
    elif name_transform==TransformsAvailable.timm_transforms_imagenet_train:
        transform_fn=transforms_imagenet_train(img_size=img_size,interpolation=Image.BILINEAR,)
        
    elif name_transform==TransformsAvailable.timm_noaug:
        transform_fn=transforms_noaug_train(img_size=img_size)
        
    #el transforms_magenet aplica un center crop de 0.875 el paper que estoy mirnado no lo usa
    transform_fn_test= transforms_imagenet_eval(img_size=img_size) 
    return transform_fn,transform_fn_test
    
def get_datamodule(name_dataset:str,batch_size:int,transform_fn,transform_fn_test):

    if isinstance(name_dataset,str):
        name_dataset=Dataset[name_dataset.lower()]
 
    if name_dataset==Dataset.grocerydataset:
        
        dm=GroceryStoreDataModule(data_dir="data",
                                  batch_size=batch_size,
                                  transform_fn=transform_fn,
                                  transform_fn_test=transform_fn_test,
                                  )
        
    elif name_dataset==Dataset.fgvcaircraft:
        dm=FGVCAircraft(transform_fn=transform_fn,
                        transform_fn_test=transform_fn_test,
                        data_dir="data",
                        batch_size=batch_size,
                        )
    else: 
        raise ("choice a correct dataset")
    dm.prepare_data()   
    dm.setup()
    return dm


def get_system(datamodule:pl.LightningDataModule,
               model_choice:str,
               optim:str,
               lr:float,
               img_size:int,
               ):
    
    if isinstance(model_choice,str):
        model_choice=ModelsAvailable[model_choice.lower()]
    if model_choice==ModelsAvailable.hierarchicaltransformers:
        model=LitHierarchyTransformers(datamodule.classlevel,optim,lr,img_size)
    else: 
        model=LitGeneralModellevel0(model_choice,datamodule.classlevel,optim,lr,img_size)
        

    return model

def get_trainer(wandb_logger,config):
    
    gpus=[]
    if config.gpu0:
        gpus.append(0)
    if config.gpu1:
        gpus.append(1)
    logging.info( "gpus active",gpus)
    if len(gpus)>=2:
        distributed_backend="ddp"
        accelerator="dpp"
        plugins=DDPPlugin(find_unused_parameters=False)
    else:
        distributed_backend=None
        accelerator=None
        plugins=None
        
    #callbacks
    early_stopping=EarlyStopping(monitor='_val_loss_total',
                                 mode="min",
                                patience=5,
                                 verbose=True,
                                 check_finite =True
                                 )

    checkpoint_callback = ModelCheckpoint(
        monitor='_val_loss_total',
        dirpath=config.PATH_CHECKPOINT,
        filename= '-{epoch:02d}-{val_loss:.6f}',
        mode="min",
        save_last=True,
        save_top_k=3,
                        )
    learning_rate_monitor=LearningRateMonitor(logging_interval="epoch")
        
    trainer=pl.Trainer(
                    logger=wandb_logger,
                       gpus=gpus,
                       max_epochs=config.NUM_EPOCHS,
                       precision=config.precision_compute,
                    #    limit_train_batches=0.1, #only to debug
                    #    limit_val_batches=0.05, #only to debug
                    #    val_check_interval=1,
                        auto_lr_find=config.AUTO_LR,
                       log_gpu_memory=True,
                    #    distributed_backend=distributed_backend,
                    #    accelerator=accelerator,
                    #    plugins=plugins,
                       callbacks=[
                            early_stopping ,
                            # checkpoint_callback,
                            # confusion_matrix_wandb,
                            learning_rate_monitor 
                                  ],
                       progress_bar_refresh_rate=5,
                       
                       )
    
    return trainer
