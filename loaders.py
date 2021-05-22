import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import albumentations as A
from torchvision import transforms
import logging
from tqdm import tqdm
class Loader(Dataset):
    def __init__(self,df:pd.DataFrame,transform_fn) -> None:
        super().__init__()
        
        self.data=df
        self.loader=default_loader        
        self.transform=transform_fn

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return NotImplementedError
        
    def apply_transform(self,img):
        if self.transform.__class__ is A.core.composition.Compose:
            img=np.array(img)
            augmentations=self.transform(image=img)
            img=augmentations["image"]
        elif self.transform.__class__==transforms.transforms.Compose:
            img=self.transform(img)
        else:
            pass
        return img
class GroceryStoreLoader(Loader):
    
    def __init__(self,
                 transform_fn,
                 input_size,
                 root_dir:str=os.path.join("data","GroceryStoreDataset"),
                 hierarchylevel:int=2,
                 split:str="train",
                 ) -> None:
        
        
        self.root_dir=os.path.join(root_dir,"dataset")
        self.split=split
        self.num_class_level00=43
        self.num_class_level0=81
        

        if self.split=="train":
            file_with_images=os.path.join(self.root_dir,"train.txt")
        elif self.split=="val":
            file_with_images=os.path.join(self.root_dir,"val.txt")
        elif self.split=="test":
            file_with_images=os.path.join(self.root_dir,"test.txt")
        else:
            raise("insert a valid split")

        df=pd.read_csv(file_with_images,delimiter="," )
        super().__init__(df,
                         transform_fn=transform_fn,
                         input_size=input_size)
        
    
    def __getitem__(self, index):
        fname=self.data.iloc[index,0]
        img_path=os.path.join(self.root_dir,fname)
        img=self.loader(img_path)
 
        label_level00=torch.tensor(int(self.data.iloc[index,1]))
        label_level0=torch.tensor(int(self.data.iloc[index,2]))
        img=self.apply_transform(img)
        
        return img,(label_level0,label_level00),fname

class FGVCAircraftLoader(Loader):
    
    def __init__(self,
                 transform_fn,
                 root_dir:str="data",
                 hierarchylevel:int=2,
                 split:str="train",   
                 ) -> None:
        
        self.root_dir=os.path.join(root_dir,"fgvc-aircraft-2013b","data")
        self.root_images=os.path.join(self.root_dir,"images")
        self.split=split
        self.num_class_level000=30
        self.num_class_level00=70
        self.num_class_level0=100
        
        df=self.create_df_from_txts()
        
        super().__init__(df,
                         transform_fn=transform_fn,
                         )
        
    def create_df_from_txts(self) ->pd.DataFrame:
        root_all_txt="images"
        parts={
                "level000": "manufacturer",
                "level00": "family",
                "level0": "variant",
                    }
        
        assert self.split in ["train","val","test"]
        
        df=pd.DataFrame()
        for level,middle_name in parts.items():
            filename_txt="_".join([root_all_txt,middle_name,self.split+".txt"])
            filepath_txt=os.path.join(self.root_dir,filename_txt)
            df_aux=pd.read_csv(filepath_txt,names=["all"],dtype=str)
            df_aux["filename"]=df_aux["all"].astype(str).str[0:7]
            df_aux[level]=df_aux["all"].astype(str).str[8:]
            df_aux.drop(labels="all",inplace=True,axis=1)
            df=pd.concat([df,df_aux],axis=1)
        
        
        df = df.loc[:,~df.columns.duplicated()]
        
        df["label000"]=df.groupby(["level000"]).ngroup()
        df["label00"]=df.groupby(["level00"]).ngroup()
        df["label0"]=df.groupby(["level0"]).ngroup()
        
        return df

    def __getitem__(self, index):
        fname=self.data.iloc[index,0]
        img_path=os.path.join(self.root_images,fname+".jpg")
        img=self.loader(img_path)

        label_level000=torch.tensor(int(self.data.iloc[index]["label000"]))
        label_level00=torch.tensor(int(self.data.iloc[index]["label00"]))
        label_level0=torch.tensor(int(self.data.iloc[index]["label0"]))
        
        img=self.apply_transform(img)
        return img,(label_level0,label_level00,label_level000),fname
    
class Cars196Loader(Loader):
    #https://github.com/phongdinhv/stanford-cars-model/blob/master/data_processing/data_loaders.py
    def __init__(self,
                 transform_fn,
                 root_images:str,
                 meta_mat_file:str,
                 root_dir:str="data",
                 
                 hierarchylevel:int=3,
                 split:str="train",
                 
                 ) -> None:
        
        self.root_dir=root_dir
        self.root_images=root_images
        label_mat_file=os.path.join(self.root_dir,"devkit","cars_meta.mat")
        self.labels=self._get_labels(label_mat_file)
        self.split=split
        self.csv_path=os.path.join(self.root_dir,"devkit",self.split+".csv")
        df=self._create_df(meta_mat_file)
        super().__init__(df,
                         transform_fn,
                         
                         )
        
    def __getitem__(self, index):
        filename=str(self.data.iloc[index]["filename"])
        img_path=os.path.join(self.root_images,filename+".jpg")
        img=self.loader(img_path)
        label_level000=torch.tensor(int(self.data.iloc[index]["label000"]))
        label_level00=torch.tensor(int(self.data.iloc[index]["label00"]))
        label_level0=torch.tensor(int(self.data.iloc[index]["label0"]))
        img=self.apply_transform(img)
        return img,(label_level0,label_level00,label_level000),filename
    
    def _load_mat_file(self,path_mat):
        from scipy import io as mat_io
        return mat_io.loadmat(path_mat)
        
    def _get_labels(self,label_mat_file):
        mat_file = self._load_mat_file(label_mat_file)
        labels=np.array(mat_file["class_names"])

        labels=np.squeeze(labels,axis=0)
        return labels
    
    def _create_df(self,path_mat):
        
        
        if os.path.exists(self.csv_path):
            logging.info("pre_loading csv {self.csv_path}")
            df=pd.read_csv(self.csv_path,index_col="Unnamed: 0")
            df["filename"]=df["filename"].astype(str).str.zfill(5)
        else:
            logging.info("creating csv {self.csv_path}")
            mat = self._load_mat_file(path_mat)
            
            df=pd.DataFrame()
            for example in tqdm(mat['annotations'][0],miniters=100):

                image_name = example[-1].item().split('.')[0]
                label = self.labels[int(example[4].item()) - 1][0]

                features = {
                        'label': label,
                        'filename':str(image_name),
                    }
                
                df=df.append(features,ignore_index=True) 
            
            df["make_id"]=df.label.str.split(" ").str[0]
            df["model_id"]=df.label.str.split(" ").str[1:-1].str.join(" ")
            df["released_year"]=df.label.str.split(" ").str[-1]
            df["model_id_and_released_year"]=df["model_id"].astype(str)+df["released_year"].astype(str)
            df['label0'] = df.groupby(["make_id",'model_id','released_year']).ngroup()
            df['label00'] = df.groupby(["make_id",'model_id']).ngroup()
            df['label000'] = df.groupby(['make_id']).ngroup()
            
            df.to_csv(self.csv_path)
        return df