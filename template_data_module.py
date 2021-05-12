import pytorch_lightning as pl
import os
from torchvision.datasets.folder import default_loader

class TemplateDataModule(pl.LightningDataModule):
    def __init__(self,
                 transform_fn,
                 classlevel:dict,
                 data_dir: str = "data",
                 batch_size: int = 32
                 ):

        self.transform_fn=transform_fn
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.classlevel=classlevel
        self.loader=default_loader
        
    # def prepare_data(self) -> None:
    #     return NotImplementedError    
    # def setup(self, stage=None):
    #     return NotImplementedError

    # def train_dataloader(self):
    #     return NotImplementedError

    # def val_dataloader(self):
    #     return NotImplementedError

    # def test_dataloader(self):
    #     return NotImplementedError