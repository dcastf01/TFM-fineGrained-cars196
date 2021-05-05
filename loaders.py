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


