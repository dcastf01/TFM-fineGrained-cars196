

import datetime
import logging
import pytorch_lightning as pl

import torch
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin


from config import CONFIG,create_config_dict

from builders import get_datamodule,get_system, get_transform_function
from autotune import autotune_lr
def main():
    logging.info("empezando setup del experimento")
    torch.backends.cudnn.benchmark = True
    config=CONFIG()
    config_dict=create_config_dict(config)
    wandb.init(
            project="Hierarchy-label-transformers",
            entity='dcastf01',
                            
            config=config_dict
                )
    wandb.run.name=config.experiment_name+" "+\
                    datetime.datetime.utcnow().strftime("%Y-%m-%d %X")

    # wandb.run.save()
    
    wandb_logger=WandbLogger(
        #offline=True,
                )
    config=wandb.config
    #get transform_fn
    
    transfrom_fn=get_transform_function(config.transform_name,
                                        config.IMG_SIZE)
    #get datamodule
    dm=get_datamodule(config.dataset_name,
                      config.batch_size,
                      transfrom_fn
                      )
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

    #get system
    model=get_system(dm,
                     config.experiment_name,
                     config.optim_name,
                     config.lr)
    #create trainer
    
    trainer=pl.Trainer(
                    logger=wandb_logger,
                       gpus=[1],
                       max_epochs=config.NUM_EPOCHS,
                       precision=config.precision_compute,
                    #    limit_train_batches=0.1, #only to debug
                    #    limit_val_batches=0.05, #only to debug
                    #    val_check_interval=1,
                        auto_lr_find=config.AUTO_LR,

                       log_gpu_memory=True,
                    #    distributed_backend='ddp',
                    #    accelerator="dpp",
                    #    plugins=DDPPlugin(find_unused_parameters=False),
                       callbacks=[
                            early_stopping ,
                            # checkpoint_callback,
                            # confusion_matrix_wandb,
                            # learning_rate_monitor 
                                  ],
                       progress_bar_refresh_rate=5,
                       )
    
    model=autotune_lr(trainer,model,dm,get_auto_lr=config.AUTO_LR)

    logging.info("empezando el entrenamiento")
    trainer.fit(model,datamodule=dm)
    
    
if __name__=="__main__":
    main()