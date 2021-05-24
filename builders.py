
import logging

import pytorch_lightning as pl
import torch.nn as nn
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from callbacks import AccuracyEnd
from config import (CONFIG, ArchitectureType, CollateAvailable, Dataset,
                    ModelsAvailable, TransformsAvailable)
from data_modules import (Cars196DataModule, FGVCAircraftDataModule,
                          GroceryStoreDataModule)
from factory_augmentations import (TwoCropTransform, basic_transforms,
                                   cars_test_transfroms_transFG,
                                   cars_train_transfroms_transFG,
                                   transforms_imagenet_eval,
                                   transforms_imagenet_train,
                                   transforms_noaug_train,
                                   get_normalize_parameter_by_model)
from factory_collate import collate_two_images
from lit_general_model_level0 import LitGeneralModellevel0
from lit_hierarchy_transformers import LitHierarchyTransformers
from losses import (ContrastiveLossFG, CrosentropyStandar,
                    SymNegCosineSimilarityLoss)


def get_transform_collate_function(transforms:str,
                                   img_size:int,
                                   collate_fn:str,
                                   model_name:str
                           ):
    model_enum=ModelsAvailable[model_name.lower()]
    name_transform=TransformsAvailable[transforms.lower()]
    name_collate=CollateAvailable[collate_fn.lower()]
    transform_fn_test=None
    mean,std,interpolation=get_normalize_parameter_by_model(model_enum)
    if name_transform==TransformsAvailable.basic_transforms:
        transform_fn=basic_transforms(img_size=img_size)
    elif name_transform==TransformsAvailable.timm_transforms_imagenet_train:
        transform_fn=transforms_imagenet_train(img_size=img_size,interpolation=Image.BILINEAR,)
        
    elif name_transform==TransformsAvailable.timm_noaug:
        transform_fn=transforms_noaug_train(img_size=img_size)
    
    elif name_transform==TransformsAvailable.cars_transfroms_transfg:
        transform_fn=cars_train_transfroms_transFG(img_size=img_size,
                                                   mean=mean,
                                                   std=std,
                                                   interpolation=interpolation,
                                                   )
        transform_fn_test=cars_test_transfroms_transFG(img_size=img_size,
                                                       mean=mean,
                                                       std=std,
                                                       interpolation=interpolation
                                                       )
    
    if transform_fn_test is None:   
    # if config.two_crops
    #el transforms_magenet aplica un center crop de 0.875 el paper que estoy mirnado no lo usa
        transform_fn_test= transforms_imagenet_eval(img_size=img_size) 
    
    if name_collate==CollateAvailable.collate_two_images:
        collate_fn=collate_two_images(transform_fn)
        transform_fn=None
        
    else:
        collate_fn=None
    
    return transform_fn,transform_fn_test,collate_fn
    
def get_datamodule(name_dataset:str
                   ,batch_size:int,
                   transform_fn,
                   transform_fn_test,
                   collate_fn
                   ):

    if isinstance(name_dataset,str):
        name_dataset=Dataset[name_dataset.lower()]
    
    if name_dataset==Dataset.grocerydataset:
        
        dm=GroceryStoreDataModule(
            data_dir="data",
            batch_size=batch_size,
            transform_fn=transform_fn,
            transform_fn_test=transform_fn_test,
            collate_fn=collate_fn
            )
        
    elif name_dataset==Dataset.fgvcaircraft:
        dm=FGVCAircraftDataModule(
            transform_fn=transform_fn,
            transform_fn_test=transform_fn_test,
            data_dir="data",
            batch_size=batch_size,
            collate_fn=collate_fn
            )
    
    elif name_dataset==Dataset.cars196:
        
        dm=Cars196DataModule(
            transform_fn=transform_fn,
            transform_fn_test=transform_fn_test,
            data_dir="data",
            batch_size=batch_size,
            collate_fn=collate_fn
                    )
    
    else: 
        raise ("choice a correct dataset")
    
    dm.prepare_data()   
    dm.setup()
    return dm

def get_losses_fn( config)->dict:
    losses_fn={}
    
    if config.loss_crossentropy_standar:
        losses_fn["crossentropy"]=CrosentropyStandar()
    if config.loss_contrastive_standar:
        losses_fn["contrastive_standar"]=NotImplementedError
    if config.loss_contrastive_fg:
        losses_fn["contrastive_fg"]=ContrastiveLossFG()
    if config.loss_triplet:
        losses_fn["triplet"]=NotImplementedError
    if config.loss_cosine_similarity_simsiam:
        losses_fn["similarity"]=SymNegCosineSimilarityLoss()
    if len(losses_fn)==0:
        raise("select unless one loss")
    
    return losses_fn

def get_callbacks(config,dm):
    #callbacks
    
    early_stopping=EarlyStopping(monitor='_val_loss',
                                 mode="min",
                                patience=10,
                                 verbose=True,
                                 check_finite =True
                                 )

    checkpoint_callback = ModelCheckpoint(
        monitor='_val_loss',
        dirpath=config.PATH_CHECKPOINT,
        filename= '-{epoch:02d}-{val_loss:.6f}',
        mode="min",
        save_last=True,
        save_top_k=3,
                        )
    learning_rate_monitor=LearningRateMonitor(logging_interval="epoch")
    
    Accuracytest=AccuracyEnd(dm.test_dataloader(),prefix="test")
    callbacks=[
        Accuracytest,
        learning_rate_monitor,
        early_stopping,
        
            ]
    return callbacks

def get_system( datamodule:pl.LightningDataModule,
                criterions:dict,
                architecture_type:str,
                model_choice:str,
                optim:str,
                lr:float,
                img_size:int,
                pretrained:bool,
                epochs:int,
                steps_per_epoch:int,
                callbacks:list

               ):
    
    if isinstance(model_choice,str) and isinstance(architecture_type,str):
        model_choice=ModelsAvailable[model_choice.lower()]
        architecture_type=ArchitectureType[architecture_type.lower()]
        
    if architecture_type==ArchitectureType.hierarchical:
        model=LitHierarchyTransformers(model_choice,
                                       datamodule.classlevel,
                                       optim,
                                       lr,
                                       img_size,
                                       pretrained,
                                       epochs,
                                       steps_per_epoch
                                       )
    elif architecture_type==ArchitectureType.standar: 
        model=LitGeneralModellevel0(model_choice,
                                    criterions,
                                    datamodule.classlevel,
                                    optim,
                                    lr,
                                    img_size,
                                    pretrained,
                                    epochs,
                                    steps_per_epoch
                                    )
    
    else:
        raise NotImplementedError
        

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
                                patience=8,
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

