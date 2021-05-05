
from config import Dataset
from grocery_store_data_module import GroceryStoreDataModule
from fgvc_aircraft_data_module import FGVCAircraft

def get_datamodule(name_dataset:str):

    if isinstance(name_dataset,str):
        name_dataset=Dataset[name_dataset]
    
    if name_dataset==Dataset.grocerydataset:
        
        dm=GroceryStoreDataModule(data_dir="data")
        
    elif name_dataset==Dataset.fgvcaircraft:
        dm=FGVCAircraft(data_dir="data")
    
    else: 
        raise ("choice a correct dataset")
    dm.prepare_data()   
    return dm