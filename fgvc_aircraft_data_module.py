#https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
#https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/aircraft.py

import os 
from torchvision.datasets.utils import download_url,extract_archive
from torch.utils.data import DataLoader
from loaders import FGVCAircraftLoader
from config import CONFIG
from template_data_module import TemplateDataModule
class FGVCAircraft(TemplateDataModule):
    def __init__(self,
                 transform_fn,
                 transform_fn_test,
                 data_dir: str = "data",
                batch_size: int = 32,
                 ):
        
        self.root=data_dir
        self.url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
        self.class_types = ('variant', 'family', 'manufacturer')
        self.splits = ('train', 'val', 'trainval', 'test')
        self.img_folder = os.path.join(self.root,'fgvc-aircraft-2013b', 'data', 'images')
        classlevel={
                          #extraer esta variable del dataset más adelante
                        'level0':100,
                        'level00': 70,
                        'level000':30 ,
                            }
        
        super().__init__(
            transform_fn=transform_fn,
            transform_fn_test=transform_fn_test,
            data_dir=data_dir,
            batch_size=batch_size,
            classlevel=classlevel
                        )
        
    def prepare_data(self):
        
        # download
        tar_name = self.url.rpartition('/')[-1]
        if not os.path.isfile(os.path.join(self.root,tar_name)):
            
     
            self.download()

    def setup(self, stage=None):
        self.FGVCaircraft_train = FGVCAircraftLoader(self.transform_fn,
                                                     self.data_dir, split="train")
        self.FGVCaircraft_val= FGVCAircraftLoader(self.transform_fn_test,
                                                  self.data_dir, split="val")
        self.FGVCaircraft_test = FGVCAircraftLoader(self.transform_fn_test,
                                                    self.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(self.FGVCaircraft_train, batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=CONFIG.NUM_WORKERS,
                          pin_memory=True,
                          drop_last=False
                          )

    def val_dataloader(self):
        return DataLoader(self.FGVCaircraft_val, batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=CONFIG.NUM_WORKERS,
                          pin_memory=True,
                          drop_last=False
                          )

    def test_dataloader(self):
        return DataLoader(self.FGVCaircraft_test, batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=CONFIG.NUM_WORKERS,
                          pin_memory=True,
                          drop_last=False
                          )
    
    
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')
        
        
