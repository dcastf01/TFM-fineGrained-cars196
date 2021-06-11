import os

import matplotlib.offsetbox as osb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
# for resizing images to thumbnails
import torchvision.transforms.functional as functional
from matplotlib import rcParams as rcp
from PIL import Image
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
# for clustering and 2d representations
from sklearn import random_projection
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb
from config import FreezeLayersAvailable
import seaborn as sns

class FreezeLayers(BaseFinetuning):
    
    
    def __init__(self,freeze_layer_enum):
        if freeze_layer_enum==FreezeLayersAvailable.freeze_all_except_last:
            self._unfreeze_at_epoch = 1000 #number so high mean never unfreeze
        elif freeze_layer_enum==FreezeLayersAvailable.freeze_all_except_last_to_epoch25:
            self._unfreeze_at_epoch = 25
            
        super().__init__()

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing ``feature_extractor``
        all_layers=list(pl_module.model.children())
        all_layers_except_last=all_layers[:-1]
        self.freeze(all_layers_except_last)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` is 10, feature_extractor will start training.
        all_layers=list(pl_module.model.children())
        all_layers_except_last=all_layers[:-1]
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=all_layers_except_last,
                optimizer=optimizer,
                train_bn=True,
            )
    
class PlotLatentSpace(Callback):
    def __init__(self,dataloader:DataLoader) -> None:
        super(PlotLatentSpace,self).__init__()
        self.dataloader=dataloader
        self.path_to_data=dataloader.dataset.root_images
        self.each_epoch=5
        

    def get_scatter_plot_with_thumbnails(self,trainer,pl_module,embeddings_2d,filenames):
        """Creates a scatter plot with image overlays.
        """
        # initialize empty figure and add subplot
        fig = plt.figure()
        fig.suptitle('Scatter Plot of the cars196 Dataset')
        ax = fig.add_subplot(1, 1, 1)
        # shuffle images and find out which images to show
        shown_images_idx = []
        shown_images = np.array([[1., 1.]])
        iterator = [i for i in range(embeddings_2d.shape[0])]
        np.random.shuffle(iterator)
        for i in iterator:
            # only show image if it is sufficiently far away from the others
            dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
            if np.min(dist) < 2e-10:
                continue
            shown_images = np.r_[shown_images, [embeddings_2d[i]]]
            shown_images_idx.append(i)

        # plot image overlays
        for idx in shown_images_idx:
            thumbnail_size = int(rcp['figure.figsize'][0] * 2.)
            path = os.path.join(self.path_to_data, filenames[idx]+".jpg")
            img = Image.open(path)
            img = functional.resize(img, thumbnail_size)
            img = np.array(img)
            img_box = osb.AnnotationBbox(
                osb.OffsetImage(img, cmap=plt.cm.gray_r),
                embeddings_2d[idx],
                pad=0.2,
            )
            ax.add_artist(img_box)

        # set aspect ratio
        ratio = 1. / ax.get_data_ratio()
        ax.set_aspect(ratio, adjustable='box')
        # Save just the portion _inside_ the second axis's boundaries
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('ax2_figure.png', bbox_inches=extent)

        # Pad the saved area by 10% in the x-direction and 20% in the y-direction
        fig.savefig('ax2_figure_expanded.png', bbox_inches=extent.expanded(1.5, 1.5))
        self.upload_image(trainer,fig,"eliminare")
        
    def upload_image(self,trainer,image,text:str):
       
        trainer.logger.experiment.log({
            f"{text}/examples": [
                wandb.Image(image) 
                ],
            })
    def create_embbedings_2d(self,trainer,pl_module):
        embeddings = []
        filenames = []
        labels=[]
        rng=np.random.default_rng()
        classes_selected=np.array(list(range(0,198,10))    )
        # disable gradients for faster calculations
        pl_module.eval()
        with torch.no_grad():
            for i, (x, y_true, fnames) in enumerate(self.dataloader):
                y_true=y_true[0] #debido a que el loader te devuelve tres etiquetas
                # move the images to the gpu
                x = x.to(device=pl_module.device)
                # embed the images with the pre-trained backbone
                y = pl_module.model.pre_classifier(x)
                y = y.squeeze()
                for embbeding,label in zip(y,y_true):
                    if any([(label == class_selected).all() for class_selected in classes_selected]):
                    # if label in class_selected:
                        # store the embeddings and filenames in lists
                        embeddings.append(torch.unsqueeze(embbeding,dim=0))
                        # filenames = filenames + list(fnames)
                        labels.append(label.item())
                # if i*x.shape[0]>250:
                #     break

        # concatenate the embeddings and convert to numpy
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.cpu().numpy()
        tl=TSNE()
        embeddings_2d=tl.fit_transform(embeddings)
        # projection = random_projection.GaussianRandomProjection(n_components=2)
        # embeddings_2d = projection.fit_transform   (embeddings)
        return embeddings_2d ,labels
    def plot_emmbedings(self,trainer,pl_module,embedding,labels):
        fig=plt.figure(figsize=(10,10))
        number_labels=len(set(labels))
        color_pallete=sns.color_palette("tab20",n_colors=number_labels)#[:number_labels]
        sns.scatterplot(embedding[:,0], embedding[:,1], hue=labels,palette=color_pallete)
        plt.title(f"epoch {pl_module.current_epoch} ")
        fig.savefig('ax2_figureonlyxlabel.png')
        self.upload_image(trainer,fig,"Latent_space")
        
    def on_train_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if pl_module.current_epoch % self.each_epoch==0 or pl_module.current_epoch==0:
            embeddings_2d,filenames=self.create_embbedings_2d(trainer,pl_module)
            self.plot_emmbedings(trainer,pl_module,embeddings_2d,filenames)
                                 
        return super().on_epoch_end(trainer, pl_module)    
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        embeddings_2d,filenames=self.create_embbedings_2d(trainer,pl_module)
        self.plot_emmbedings(trainer,pl_module,embeddings_2d,filenames)
   
        return super().on_train_end(trainer, pl_module)
    
class AccuracyEnd(Callback):
    
    def __init__(self,dataloader:DataLoader) -> None:
        super(AccuracyEnd,self).__init__()
        self.dataloader=dataloader

        
    def generate_accuracy_and_upload(self,trainer: 'pl.Trainer',pl_module:'pl.LightningModule'):
        all_results=[]
        all_targets=[]
        num_correct = 0
        num_samples = 0
        for batch in self.dataloader:
            image,target,idx=batch
            y=target[0].to(pl_module.device)
            with torch.no_grad():
                # results=pl_module(image.to(device=pl_module.device))
                scores = pl_module(image.to(device=pl_module.device))
                _, predictions = scores.max(1)
                num_correct +=torch.sum(predictions == y)
                num_samples += predictions.size(0)
            # all_results.append(results.softmax(dim=1))
            # all_targets.append(target)
        accuracy=num_correct/num_samples
        # accuracy = self.simple_accuracy(all_results, all_targets)
        trainer.logger.experiment.log({
            "Accuracy ":accuracy.item(),
        
                })

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.generate_accuracy_and_upload(trainer,pl_module)
   
        return super().on_train_end(trainer, pl_module)
    
        
    def simple_accuracy(self,preds, labels):
        return (preds == labels).mean()
    
    
