import pytorch_lightning as pl
from loaders import GroceryStoreLoader
import os
from torch.utils.data import DataLoader
from config import CONFIG
from template_data_module import TemplateDataModule


class GroceryStoreDataModule(TemplateDataModule):
    def __init__(self,
                 transform_fn,
                 transform_fn_test,
                 data_dir: str = "data",
                 batch_size: int = 32):
        
        data_dir = os.path.join(data_dir,"GroceryStoreDataset")
        self.batch_size = batch_size
        classlevel={'level0':43 ,
                        'level00': 82,  #extraer esta variable del dataset más adelante
                            }
        super().__init__(
            transform_fn=transform_fn,
            transform_fn_test=transform_fn_test,
            data_dir=data_dir,
            batch_size=batch_size,
            classlevel=classlevel
            
                        )
    def prepare_data(self):
        # manual download from 
        #https://github.com/marcusklasson/GroceryStoreDataset
        pass
        

    def setup(self, stage=None):
        self.grocery_store_train = GroceryStoreLoader(self.transform_fn,
                                                      self.data_dir, split="train")
        self.grocery_store_val= GroceryStoreLoader(self.transform_fn_test,
                                                   self.data_dir, split="val")
        self.grocery_store_test = GroceryStoreLoader(self.transform_fn_test,
                                                     self.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(self.grocery_store_train, batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=CONFIG.NUM_WORKERS,
                          pin_memory=True,
                          drop_last=False
                          )

    def val_dataloader(self):
        return DataLoader(self.grocery_store_val, batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=CONFIG.NUM_WORKERS,
                          pin_memory=True,
                          drop_last=False
                          )

    def test_dataloader(self):
        return DataLoader(self.grocery_store_test, batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=CONFIG.NUM_WORKERS,
                          pin_memory=True,
                          drop_last=False
                          )