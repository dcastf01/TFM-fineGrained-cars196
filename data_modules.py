#https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
#https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/aircraft.py

import logging
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url, extract_archive

from config import CONFIG
from loaders import Cars196Loader, FGVCAircraftLoader, GroceryStoreLoader
from template_data_module import TemplateDataModule

from tqdm import tqdm
class FGVCAircraftDataModule(TemplateDataModule):
    def __init__(self,
                 transform_fn,
                 transform_fn_test,
                 data_dir: str = "data",
                batch_size: int = 32,
                collate_fn=None
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
            classlevel=classlevel,
            collate_fn=collate_fn
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
                          drop_last=False,
                          collate_fn=self.collate_fn
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
        return os.path.exists(os.path.join(self.root, self.img_folder)) 
            # and  os.path.exists(self.classes_file)

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
            
class GroceryStoreDataModule(TemplateDataModule):
    def __init__(self,
                 transform_fn,
                 transform_fn_test,
                 data_dir: str = "data",
                 batch_size: int = 32,
                 collate_fn=None
                 ):
        
        data_dir = os.path.join(data_dir,"GroceryStoreDataset")
        self.batch_size = batch_size
        classlevel={
            'level0': 82,
            'level00':43 ,
                          #extraer esta variable del dataset más adelante
                            }
        super().__init__(
            transform_fn=transform_fn,
            transform_fn_test=transform_fn_test,
            data_dir=data_dir,
            batch_size=batch_size,
            classlevel=classlevel,
            collate_fn=collate_fn
            
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
                          drop_last=False,
                          collate_fn=self.collate_fn
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

class Cars196DataModule(TemplateDataModule):
    
    def __init__(self,
                 transform_fn,
                 transform_fn_test,
                 data_dir: str, 
                 batch_size: int,
                 collate_fn=None
                 ):
        
        self.root=os.path.join(data_dir,"cars196")
        self.img_folder_train=os.path.join(self.root,"cars_train")
        self.img_folder_test=os.path.join(self.root,"cars_test")
        self.img_folder_crop_train=os.path.join(self.root,"crop_cars_train")
        self.img_folder_crop_test=os.path.join(self.root,"crop_cars_test")
        
        self.train_annos_mat=os.path.join(self.root,"devkit","cars_train_annos.mat")
        self.test_annos_mat=os.path.join(self.root,"devkit","cars_test_annos_withlabels.mat")
        
        self.urls=[
            "http://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
            "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
            "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"
            
                    ]
        self.extra_urls_not_compress="http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat"
        self.class_types = ('make', 'model', 'released year')
        
        classlevel={
                          #extraer esta variable del dataset más adelante
                        'level0':196,
                        'level00': 189,
                        'level000':49 ,
                            }
        
        super().__init__(transform_fn,
                         transform_fn_test,
                         classlevel,
                         data_dir=data_dir,
                         batch_size=batch_size,
                         collate_fn=collate_fn
                         )
        
    def prepare_data(self):
        # manual download from 
        #https://ai.stanford.edu/~jkrause/cars/car_dataset.html
        
        self.download()
        self.crop_images()
        
    def setup(self, stage=None):
        self.cars196_train = Cars196Loader( self.transform_fn,
                                            self.img_folder_crop_train,
                                            self.train_annos_mat,
                                            self.root,
                                            split="train"
                                                      )
        self.cars196_val= Cars196Loader(    self.transform_fn_test,
                                            self.img_folder_crop_test,
                                            self.test_annos_mat,
                                            self.root,
                                            split="val"
                                            )
        self.cars196_test = Cars196Loader(  self.transform_fn_test,
                                            self.img_folder_crop_test,
                                            self.test_annos_mat,
                                            self.root,
                                            split="test"
                                            )

    def train_dataloader(self):
        return DataLoader(self.cars196_train, batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=CONFIG.NUM_WORKERS,
                          pin_memory=True,
                          drop_last=False,
                          collate_fn=self.collate_fn                          
                          )

    def val_dataloader(self):
        return DataLoader(self.cars196_val, batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=CONFIG.NUM_WORKERS,
                          pin_memory=True,
                          drop_last=False
                          )

    def test_dataloader(self):
        return DataLoader(self.cars196_test, batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=CONFIG.NUM_WORKERS,
                          pin_memory=True,
                          drop_last=False
                          )
              
    def _check_exists(self):
        return os.path.exists( self.img_folder_train)\
            and os.path.exists(self.img_folder_test)

    def download(self):
        if self._check_exists():
            return
        
        os.makedirs(os.path.join(self.root),exist_ok=True)
        # prepare to download data to PARENT_DIR/cars196
        for url in self.urls:
            print('Downloading %s...' % url)
            tar_name = url.rpartition('/')[-1]
            download_url(url, root=self.root, filename=tar_name)
            tar_path = os.path.join(self.root, tar_name)
            print('Extracting %s...' % tar_path)
            extract_archive(tar_path)
            print('Done!')
        download_url(self.extra_urls_not_compress,
                     root=os.path.join(self.root,"devkit"),
                     filename=self.test_annos_mat)
        
    def crop_images(self):
        def crop_split(metas,original_folder,extracted_folder):
            #https://github.com/phongdinhv/stanford-cars-model/blob/master/data_processing/extract_cars.py
            from scipy import io as mat_io
            from skimage import io as img_io
            labels_meta = mat_io.loadmat(metas)

            for img_ in tqdm(labels_meta['annotations'][0],miniters=100):
                x_min = img_[0][0][0]
                y_min = img_[1][0][0]

                x_max = img_[2][0][0]
                y_max = img_[3][0][0]

                if len(img_) == 6:
                    img_name = img_[5][0]
                elif len(img_) == 5:
                    img_name = img_[4][0]
                try:
                    img_in = img_io.imread(os.path.join(original_folder, img_name))
                except Exception:
                    print("Error while reading!")
                else:
                    # print(img_in.shape)
                    cars_extracted = img_in[y_min:y_max, x_min:x_max]
                    logging.info(x_min, y_min, x_max, y_max, cars_extracted.shape, img_in.shape, img_name)

                    img_io.imsave(os.path.join(extracted_folder, img_name), cars_extracted)
        
        if not os.path.exists( self.img_folder_crop_train):
            os.makedirs(self.img_folder_crop_train,exist_ok=True)
            crop_split(
                        metas=self.train_annos_mat,
                        original_folder=self.img_folder_train,
                        extracted_folder=self.img_folder_crop_train
                        )
        if not os.path.exists( self.img_folder_crop_test):
            os.makedirs(self.img_folder_crop_test,exist_ok=True)
            crop_split(
                        metas=self.test_annos_mat,
                        original_folder=self.img_folder_test,
                        extracted_folder=self.img_folder_crop_test
                
                        )
            

