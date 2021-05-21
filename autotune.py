import logging
import pytorch_lightning as pl
from pytorch_lightning.tuner.lr_finder import lr_find
def autotune_lr(trainer:pl.Trainer,
                model,
                data_module:pl.LightningDataModule,
                get_auto_lr:bool=False,
                model_name:str="",
                dataset_name:str=""):
    
        
    if get_auto_lr:
        logging.info("Buscando el learning rate óptimo")
        # Run learning rate finder
        # lr_finder = trainer.tuner.lr_find(model,data_module,)
        # algo=lr_find(trainer,model)
        lr_finder = trainer.tuner.lr_find(model=model,datamodule=data_module,max_lr=1e-2,min_lr=1e-6)

        # # Results can be found in
        # lr_finder.results
        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig("autolr "+model_name+" "+dataset_name+".jpg")
        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        logging.info("El lr óptimo es {new_lr}")
        # update hparams of the model
        model.hparams.lr = new_lr
        
    return model