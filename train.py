

import datetime
import logging
import pytorch_lightning as pl

import torch
import wandb

from pytorch_lightning.loggers import WandbLogger


from config import CONFIG,create_config_dict

from builders import get_datamodule,get_system, get_transform_function,get_trainer
from autotune import autotune_lr

def main():
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
    wandb.run.name=config.experiment_name+" "+\
                    datetime.datetime.utcnow().strftime("%Y-%m-%d %X")
                    
    wandb.run.notes=config.notes
    # wandb.run.save()
    
    wandb_logger=WandbLogger(
        #offline=True,
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
    

    #get system
    model=get_system(dm,
                     config.experiment_name,
                     config.optim_name,
                     config.lr,
                     config.IMG_SIZE
                     )
    #create trainer
    trainer=get_trainer(wandb_logger,config)
    
    model=autotune_lr(trainer,model,dm,get_auto_lr=config.AUTO_LR)

    logging.info("empezando el entrenamiento")
    trainer.fit(model,datamodule=dm)


if __name__=="__main__":
    main()