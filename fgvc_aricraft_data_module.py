#https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
#https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/aircraft.py
import pytorch_lightning as pl
import os 
from torchvision.datasets.utils import download_url,extract_archive
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader

class FGVCAircraft(pl.LightningDataModule):
    def __init__(self,data_dir: str = "data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.root=data_dir
        
        self.url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
        self.class_types = ('variant', 'family', 'manufacturer')
        self.splits = ('train', 'val', 'trainval', 'test')
        self.img_folder = os.path.join(self.data_dir,'fgvc-aircraft-2013b', 'data', 'images')
        
        self.loader=default_loader
            
    def prepare_data(self):
        
        # download
        self.download()

    def setup(self, stage=None):
        
        self.mnist_test = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
    
    
    
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
        
        
