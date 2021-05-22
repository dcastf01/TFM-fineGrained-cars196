

from lightly.data.collate import BaseCollateFunction

from factory_augmentations import cars_train_transfroms_transFG

def collate_two_images(transform):
    # transform=cars_train_transfroms_transFG(img_size=img_size)
    collate_fn=BaseCollateFunction(transform)                        
    return collate_fn