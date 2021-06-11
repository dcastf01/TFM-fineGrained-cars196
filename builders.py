
import logging

import pytorch_lightning as pl
import torch.nn as nn
from PIL import Image
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from callbacks import AccuracyEnd, FreezeLayers, PlotLatentSpace
from config import (CONFIG, ArchitectureType, CollateAvailable, Dataset,
                    FreezeLayersAvailable, LossDifferentExperimentsAvailable,
                    ModelsAvailable, TransformsAvailable)
from data_modules import (Cars196DataModule, FGVCAircraftDataModule,
                          GroceryStoreDataModule)
from factory_augmentations import (TwoCropTransform, cars_test_transforms,
                                   cars_train_transforms_basic,
                                   cars_train_transfroms_autoaugment,
                                   get_normalize_parameter_by_model,
                                   transforms_imagenet_eval,
                                   transforms_imagenet_train,
                                   transforms_noaug_train)
from factory_collate import (collate_mixup, collate_triplet_loss,
                             collate_two_images)
from lit_api_model import LitApi
from lit_general_model_level0 import LitGeneralModellevel0
from lit_hierarchy_transformers import LitHierarchyTransformers
from losses import (ContrastiveLossFG, CrosentropyStandar,
                    SymNegCosineSimilarityLoss, TripletMarginLoss)


def get_transform_collate_function(transforms:str,
                                   img_size:int,
                                   collate_fn:str,
                                   model_name:str,
                           ):
    model_enum=ModelsAvailable[model_name.lower()]
    name_transform=TransformsAvailable[transforms.lower()]
    name_collate=CollateAvailable[collate_fn.lower()]
    transform_fn_test=None
    mean,std,interpolation=get_normalize_parameter_by_model(model_enum)
    if name_transform==TransformsAvailable.cars_train_transforms_basic:
        transform_fn=cars_train_transforms_basic(img_size=img_size,
                                        mean=mean,
                                        std=std,
                                        interpolation=interpolation)
    elif name_transform==TransformsAvailable.timm_transforms_imagenet_train:
        transform_fn=transforms_imagenet_train(img_size=img_size,interpolation=Image.BILINEAR,)
    elif name_transform==TransformsAvailable.cars_train_transforms_autoaugment_and_imagenet:
        transform_fn=transforms_imagenet_train(img_size=img_size,interpolation=Image.BILINEAR,
                                               auto_augment="original-mstd0.5")
    elif name_transform==TransformsAvailable.timm_noaug:
        transform_fn=transforms_noaug_train(img_size=img_size)
    
    elif name_transform==TransformsAvailable.cars_train_transfroms_autoaugment:
        transform_fn=cars_train_transfroms_autoaugment(img_size=img_size,
                                                   mean=mean,
                                                   std=std,
                                                   interpolation=interpolation,
                                                   )
        transform_fn_test=cars_test_transforms(img_size=img_size,
                                                       mean=mean,
                                                       std=std,
                                                       interpolation=interpolation
                                                       )
    elif name_transform==TransformsAvailable.cars_only_mixup :
        transform_fn=cars_train_transforms_basic(img_size=img_size,
                                                   mean=mean,
                                                   std=std,
                                                   interpolation=interpolation,
                                                   )
        transform_fn_test=cars_test_transforms(img_size=img_size,
                                                       mean=mean,
                                                       std=std,
                                                       interpolation=interpolation
                                                       )
        collate_fn=collate_mixup()
    
    elif name_transform==TransformsAvailable.cars_autoaugment_mixup :
        transform_fn=cars_train_transfroms_autoaugment(img_size=img_size,
                                                   mean=mean,
                                                   std=std,
                                                   interpolation=interpolation,
                                                   )
        transform_fn_test=cars_test_transforms(img_size=img_size,
                                                       mean=mean,
                                                       std=std,
                                                       interpolation=interpolation
                                                       )
        collate_fn=collate_mixup()
        
    elif name_transform==TransformsAvailable.cars_transforms_eval:
        raise "transform train not same eval"
    if transform_fn_test is None:   
    # if config.two_crops
    #el transforms_magenet aplica un center crop de 0.875 el paper que estoy mirnado no lo usa
        transform_fn_test= cars_test_transforms(img_size=img_size,
                                                   mean=mean,
                                                   std=std,
                                                   interpolation=interpolation,) 
    
    if name_collate==CollateAvailable.collate_two_images:
        collate_fn=collate_two_images(transform_fn)
        transform_fn=None
    elif name_collate==CollateAvailable.mixup:
        collate_fn=collate_mixup()
    elif name_collate==CollateAvailable.collate_to_triplet_loss:
        collate_fn=collate_triplet_loss(transform_fn)
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
    
    loss_experiment_config_name=config.loss_experiment_config_name
    loss_experiment_enum=LossDifferentExperimentsAvailable[loss_experiment_config_name.lower()]
    if loss_experiment_enum==LossDifferentExperimentsAvailable.only_crossentropy:
        losses_fn["crossentropy"]=CrosentropyStandar()
    elif loss_experiment_enum==LossDifferentExperimentsAvailable.only_contrastivefg:
        losses_fn["contrastive_fg"]=ContrastiveLossFG()
    elif loss_experiment_enum==LossDifferentExperimentsAvailable.only_similitud_loss:
        losses_fn["similarity"]=SymNegCosineSimilarityLoss()
    elif loss_experiment_enum==LossDifferentExperimentsAvailable.only_triplet_loss:
        losses_fn["triplet_loss"]=TripletMarginLoss()
    elif loss_experiment_enum==LossDifferentExperimentsAvailable.crossentropy_and_contrastivefg:
        losses_fn["crossentropy"]=CrosentropyStandar()
        losses_fn["contrastive_fg"]=ContrastiveLossFG()
    elif loss_experiment_enum==LossDifferentExperimentsAvailable.crossentropy_and_triplet:
        losses_fn["crossentropy"]=CrosentropyStandar()
        losses_fn["triplet_loss"]=TripletMarginLoss()
    elif loss_experiment_enum==LossDifferentExperimentsAvailable.crossentropy_and_similitud:
        losses_fn["crossentropy"]=CrosentropyStandar()
        losses_fn["similarity"]=SymNegCosineSimilarityLoss()
    elif loss_experiment_enum==LossDifferentExperimentsAvailable.custom:
        pass
    if len(losses_fn)==0:
        raise("select unless one loss")
    
    return losses_fn

def get_callbacks(config,dm):
    #callbacks
    
    early_stopping=EarlyStopping(monitor='_valid_level0Accuracy',
                                 mode="max",
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
    
    accuracytest=AccuracyEnd(dm.test_dataloader())
    plt_latent_space=PlotLatentSpace(dm.test_dataloader())
    freeze_layers_name=config.freeze_layers_name
    freeze_layer_enum=FreezeLayersAvailable[freeze_layers_name.lower()]
    if freeze_layer_enum==FreezeLayersAvailable.none:
        callbacks=[
            accuracytest,
            learning_rate_monitor,
            early_stopping,
            plt_latent_space,
            ]
    else:
        freeze_layers=FreezeLayers(freeze_layer_enum)
        callbacks=[
            accuracytest,
            learning_rate_monitor,
            early_stopping,
            freeze_layers,
            plt_latent_space
                ]
    
    return callbacks

def get_system( datamodule:pl.LightningDataModule,
                criterions:dict,
                architecture_type:str,
                model_choice:str,
                optim:str,
                scheduler_name:str,
                lr:float,
                img_size:int,
                pretrained:bool,
                epochs:int,
                steps_per_epoch:int,
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
                                    scheduler_name,
                                    epochs,
                                    steps_per_epoch,
                                    )
    elif architecture_type==ArchitectureType.api_model:
        model=LitApi(model_choice,
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

def get_trainer(wandb_logger,
                callbacks:list,
                config):
    
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
        
    trainer=pl.Trainer(
                    logger=wandb_logger,
                       gpus=gpus,
                       max_epochs=config.NUM_EPOCHS,
                       precision=config.precision_compute,
                    #    limit_train_batches=0.01, #only to debug
                    #    limit_val_batches=0.05, #only to debug
                    #    val_check_interval=1,
                        auto_lr_find=config.AUTO_LR,
                       log_gpu_memory=True,
                       distributed_backend=distributed_backend,
                       accelerator=accelerator,
                       plugins=plugins,
                       callbacks=callbacks,
                       progress_bar_refresh_rate=5,
                       )
    
    return trainer

