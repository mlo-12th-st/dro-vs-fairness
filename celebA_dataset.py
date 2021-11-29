"""
celebA_dataset.py

Script to download and preprocess the CelebA dataset
    using the torchvision.datasets implementation
    found here: https://pytorch.org/vision/0.8/datasets.html#celeba
    
Returns a torch.utils.data.Dataset object with access to the __getitem__
    and __len__ methods for the CelebA dataset along with the corresponding
    torch.utils.data.DataLoader object for iterating through the dataset

CelebA dataset source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Thanks to stackOverflow for custom Dataset class:
    https://stackoverflow.com/questions/65528568/how-do-i-load-the-celeba-dataset-on-google-colab-using-torch-vision-without-ru
"""

import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image

## Create a custom Dataset class
class CelebADataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, attr_arr, transform=None):
        """
        root_dir (string): Directory with all the images
        attr_arr: array with attribute labels
        transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        image_names = os.listdir(root_dir)
    
        self.root_dir = root_dir
        self.transform = transform 
        self.image_names = image_names
        self.attr_arr = attr_arr

    def __len__(self): 
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        return image, target attribute, spurious attribute
        """
        
        # Get the path to the image 
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img, self.attr_arr[idx,0], self.attr_arr[idx,1]


def get_attr(attr_file, target_attr, spur_attr):
    """
    Parse attribute labels from file

    Parameters
    ----------
    attr_file : str
        path to space-delimited file with attribute labels.
    target_attr : str
        target attribute name.
    spur_attr : str
        spurious attribute name.

    Returns
    -------
    attr_arr : numpy.ndarray
        n x 2 array of 0s and 1s, first col is target attribute,
        second col is spurious attribute.

    """
    df = pd.read_csv(attr_file, header=0, delim_whitespace=True)
    df_dro = df[[target_attr, spur_attr]]
    attr_arr = df_dro.values
    attr_arr[attr_arr == -1] = 0
    return attr_arr


def load_data(batch_size=4, image_size=128, train_test_split=0.8,
              target_attr='Blond_Hair', spur_attr='Male'):
    """
    Load the celebA dataset from ./data/celeba/img_align_celeba
    and return the Pytorch dataset/dataloader

    Parameters
    ----------
    batch_size : int, optional. The default is 4.
    image_size : int, optional
        spatial size of training images, images are
        resized to this size. The default is 128.
    train_test_split : float, optional.
        percentage of data allocated to training.
        The default is 0.8.
    target_attr : str, optional.  
        target attribute for model to learn.  The
        default is "Blond_Hair"
    spur_attr : str, optional.
        spurious attribute to test for dist. 
        robustness.  The default is "Male"

    Returns
    -------
    train_data : torch.utils.data.Dataset
        pytorch celebA train dataset (162079 samples).
    test_data : torch.utils.data.Dataset
        pytorch celebA test dataset (40520 samples).
    trainloader : torch.utils.data.DataLoader
        dataloader for pytorch celebA training dataset.
    testloader : torch.utils.data.DataLoader
        dataloader for pytorch celebA test dataset.

    """
    
    # Root directory for the dataset
    data_root = './data/celeba'
    # folder for image data
    img_folder = f'{data_root}/img_align_celeba_subset'
    # file for attribute labels
    attr_file = f'{data_root}/list_attr_celeba_subset.txt'
    
    # Transformations to be applied to each individual image sample
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])
    ])
    
    # Load attribute labels
    attr_arr = get_attr(attr_file, target_attr, spur_attr)
    
    # Load the dataset from file and apply transformations
    celebA_data = CelebADataset(img_folder, attr_arr, transform)
    
    train_size = int(train_test_split * len(celebA_data))
    test_size = len(celebA_data) - train_size
    train_data, test_data = torch.utils.data.random_split(celebA_data, 
                                                          [train_size, test_size])
    
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size,
                                              shuffle=True)
    
    testloader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False)
    
    return train_data, test_data, trainloader, testloader
    
    