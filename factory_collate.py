import torch.nn as nn

from lightly.data.collate import BaseCollateFunction
from typing import List
from factory_augmentations import cars_train_transfroms_transFG
import torch
from timm.data.mixup import FastCollateMixup
def collate_two_images(transform):
    # transform=cars_train_transfroms_transFG(img_size=img_size)
    collate_fn=BaseCollateFunction(transform)                        
    return collate_fn


def collate_triplet_loss(transform):
    return TripletCollateFunction(transform)


def collate_mixup():
    collate_fn=FastCollateMixup(mixup_alpha=0.3,
                                cutmix_alpha=0.3,
                                prob=0.75,
                                num_classes=100 #ver como modificar de forma semiautomatica
                                )
    
    
class TripletCollateFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, batch: List[tuple]):
        """Turns a batch of tuples into a tuple of batches.
            Args:
                batch:
                    A batch of tuples of images, labels, and filenames 
            Returns:
                A tuple of images, labels, and filenames. The images consist of 
                two batches corresponding to the two transformations of the
                input images.
        """
        
        batch_size = len(batch)

        # list of transformed images
        transforms = [self.transform(batch[i % batch_size][0]).unsqueeze_(0)
                      for i in range(2 * batch_size)]
        
       #esta es la idea pero con la etiqueta contraria a la que se tenga
        while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])
        # list of filenames
        fnames = [item[2] for item in batch]

        # tuple of transforms
        transforms = (
            torch.cat(transforms[:batch_size], 0),
            torch.cat(transforms[batch_size:], 0)
        )

        return transforms, labels, fnames
        