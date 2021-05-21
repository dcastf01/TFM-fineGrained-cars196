

import datetime
import logging
import pytorch_lightning as pl

import torch
import wandb

from pytorch_lightning.loggers import WandbLogger


from config import CONFIG,create_config_dict

from builders import get_datamodule, get_losses_fn,get_system, get_transform_function,get_trainer
from autotune import autotune_lr
import os 



def main():
    os.environ["WANDB_IGNORE_GLOBS"]="*.ckpt"
    logging.info("empezando setup del experimento")
    torch.backends.cudnn.benchmark = True
    config=CONFIG()
    config_dict=create_config_dict(config)
    wandb.init(
            project="Hierarchical-label-transformers",
            entity='dcastf01',
                            
            config=config_dict
                )
    config=wandb.config
    print(config)
    wandb.run.name=config.experiment_name[:5]+" "+\
                    datetime.datetime.utcnow().strftime("%b %d %X")
                    
    wandb.run.notes=config.notes
    # wandb.run.save()
    
    wandb_logger=WandbLogger(
        #offline=True,
        log_model =False
                )
    
    #get transform_fn
    
    transfrom_fn,transform_fn_test=get_transform_function(config.transform_name,
                                        config.IMG_SIZE)
    #get datamodule
    dm=get_datamodule(config.dataset_name,
                      config.batch_size,
                      transfrom_fn,
                      transform_fn_test
                      )
    
    #get losses
    
    losses=get_losses_fn(config)
    
    
    #get system
    model=get_system(   datamodule=dm,
                        criterions=losses,
                        architecture_type=config.architecture_name,
                        model_choice= config.experiment_name,
                        optim=config.optim_name,
                        lr= config.lr,
                        img_size=config.IMG_SIZE,
                        pretrained=config.PRETRAINED_MODEL,
                        epochs=config.NUM_EPOCHS,
                        steps_per_epoch=len(dm.train_dataloader())
                     )
    
    #create trainer
    trainer=get_trainer(wandb_logger,config)
    
    model=autotune_lr(trainer,
                      model,
                      dm,
                      get_auto_lr=config.AUTO_LR,
                      model_name=config.experiment_name,
                      dataset_name=config.dataset_name
                      )

    logging.info("empezando el entrenamiento")
    trainer.fit(model,datamodule=dm)


if __name__=="__main__":
    main()