import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from augmentations import basic_transform


class Loader(Dataset):
    def __init__(self,df:pd.DataFrame) -> None:
        super().__init__()
        
        self.data=df
        
        self.loader=default_loader
        self.transform=basic_transform
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return NotImplementedError
        
        
class GroceryStoreLoader(Loader):
    
    def __init__(self,
                 root_dir:str=os.path.join("data","GroceryStoreDataset"),
                 hierarchylevel:int=2,
                 split:str="train") -> None:
        
        
        self.root_dir=os.path.join(root_dir,"dataset")
        self.split=split
        self.num_class_level0=43
        self.num_class_level00=81
        

        if self.split=="train":
            file_with_images=os.path.join(self.root_dir,"train.txt")
        elif self.split=="val":
            file_with_images=os.path.join(self.root_dir,"val.txt")
        elif self.split=="test":
            file_with_images=os.path.join(self.root_dir,"test.txt")
        else:
            raise("insert a valid split")

        df=pd.read_csv(file_with_images,delimiter=",")
        super().__init__(df)
        
        
    
    def __getitem__(self, index):
    
        img_path=os.path.join(self.root_dir,self.data.iloc[index,0])
        img=self.loader(img_path)
        img=np.array(img)

        
        label_level0=torch.tensor(int(self.data.iloc[index,1]))
        label_level00=torch.tensor(int(self.data.iloc[index,2]))
        
        if self.transform:
            augmentations=self.transform(image=img)
            img=augmentations["image"]
        return img,(label_level0,label_level00)


class FGVCAircraftLoader(Loader):
    
    def __init__(self,
                 root_dir:str="data",
                 hierarchylevel:int=2,
                 split:str="train") -> None:
        
        
        self.root_dir=os.path.join(root_dir,"fgvc-aircraft-2013b","data")
        self.root_images=os.path.join(self.root_dir,"images")
        self.split=split
        self.num_class_level0=30
        self.num_class_level00=70
        self.num_class_level000=100
        
        
        df=self.create_df_from_txts()
        
        print(df.head(1))
        super().__init__(df)
        
    def create_df_from_txts(self) ->pd.DataFrame:
        root_all_txt="images"
        parts={
                "level0": "manufacturer",
                "level00": "family",
                "level000": "variant",
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
        
        df["label0"]=df.groupby(["level0"]).ngroup()
        df["label00"]=df.groupby(["level00"]).ngroup()
        df["label000"]=df.groupby(["level000"]).ngroup()
        
        return df

    def __getitem__(self, index):
    
        img_path=os.path.join(self.root_images,self.data.iloc[index,0]+".jpg")
        img=self.loader(img_path)
        img=np.array(img)

        
        label_level0=torch.tensor(int(self.data.iloc[index]["label0"]))
        label_level00=torch.tensor(int(self.data.iloc[index]["label00"]))
        label_level000=torch.tensor(int(self.data.iloc[index]["label000"]))
        
        if self.transform:
            augmentations=self.transform(image=img)
            img=augmentations["image"]
        return img,(label_level0,label_level00,label_level000)
    
    
