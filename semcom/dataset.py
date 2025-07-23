from io import BytesIO
from PIL import Image, ImageOps
import torch.utils
from torch.utils.data import Dataset
from torchvision import transforms
import os
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets
import torch
import cv2
import lmdb
from torch.utils.data import random_split
def image2tensor(image):
    image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
    return (image-0.5)/0.5

def tensor2image(tensor):
    images = tensor.cpu().clamp_(0., 1.).detach().permute(1,2,0).numpy()*255.
    images = images.astype(np.uint8)
    return images

class ImageDataset():
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Get a list of all image file paths in the 'train' directory
        self.image_paths = [os.path.join(data_dir, img) for img in sorted(os.listdir(data_dir)) if img.endswith(".jpg") or img.endswith(".png")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image,img_path 
    
class ImageFolderDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = sorted(file_paths)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image,image_path 
    
class CelebAHQ256(Dataset):
    def __init__(self, args,train_ddpm=False):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        mean=(0.5,0.5,0.5)
        std=(0.5,0.5,0.5)
        args.normalize = transforms.Normalize(mean, std)
        args.unnorm = transforms.Normalize(
            (-torch.tensor(mean) / torch.tensor(std)).tolist(), (1.0 / torch.tensor(std)).tolist())
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            args.normalize,
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            args.normalize,
        ])
 
        self.image_paths = [os.path.join(args.data_dir, img) for img in sorted(os.listdir(self.data_dir)) if img.endswith(".jpg")]
        num_train,num_val,num_test=int(len(self.image_paths)*0.8),int(len(self.image_paths)*0.1),int(len(self.image_paths)*0.1)
        self.train_path=self.image_paths[:num_train]
        self.val_path=self.image_paths[num_train+1:num_train+num_val]
        self.test_path=self.image_paths[-num_test:]   
        if train_ddpm:
            self.train_dataset = ImageFolderDataset(self.train_path+self.val_path, transform=self.transform)
        else:
            self.train_dataset = ImageFolderDataset(self.train_path, transform=self.test_transform)
        self.val_dataset = ImageFolderDataset(self.val_path, transform=self.test_transform)
        self.test_dataset = ImageFolderDataset(self.test_path, transform=self.test_transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=8)
    
    
class FFHQ256(Dataset):
    def __init__(self, args,train_ddpm=False):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        mean=(0.5,0.5,0.5)
        std=(0.5,0.5,0.5)
        args.normalize = transforms.Normalize(mean, std)
        args.unnorm = transforms.Normalize(
            (-torch.tensor(mean) / torch.tensor(std)).tolist(), (1.0 / torch.tensor(std)).tolist())
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            args.normalize,
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            args.normalize,
        ])
 
        self.image_paths = [os.path.join(args.data_dir, img) for img in sorted(os.listdir(self.data_dir)) if img.endswith(".png")]
        self.test_path=self.image_paths[0:1000]
        self.val_path=self.image_paths[1000:2001]
        self.train_path=self.image_paths[2001:]   
        self.train_dataset = ImageFolderDataset(self.train_path, transform=self.test_transform)
        self.val_dataset = ImageFolderDataset(self.val_path, transform=self.test_transform)
        self.test_dataset = ImageFolderDataset(self.test_path, transform=self.test_transform)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=8)
    
class ImageNet(Dataset):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        mean=(0.5,0.5,0.5)
        std=(0.5,0.5,0.5)
        args.normalize = transforms.Normalize(mean, std)
        args.unnorm = transforms.Normalize(
            (-torch.tensor(mean) / torch.tensor(std)).tolist(), (1.0 / torch.tensor(std)).tolist())
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            args.normalize,
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            args.normalize,
        ])

        traindir = os.path.join(self.data_dir, 'train')
        full_dataset = datasets.ImageFolder(
            traindir,
            transform=self.transform)
        train_ratio = 0.8
        train_size = int(train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        self.test_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'val'), transform=self.test_transform)

    def train_dataloader(self):
        train_sampler = torch.utils.data.RandomSampler(
            self.train_dataset, num_samples=100000)
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=8, drop_last=True, sampler=train_sampler)

    def val_dataloader(self):
        train_sampler = torch.utils.data.RandomSampler(
            self.val_dataset, num_samples=10000)
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=8, drop_last=True, sampler=train_sampler)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=8)

class Kodak(Dataset):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        mean=(0.5,0.5,0.5)
        std=(0.5,0.5,0.5)
        args.normalize = transforms.Normalize(mean, std)
        args.unnorm = transforms.Normalize(
            (-torch.tensor(mean) / torch.tensor(std)).tolist(), (1.0 / torch.tensor(std)).tolist())
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            args.normalize,
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            args.normalize,
        ])
 
        self.image_paths = [os.path.join(args.data_dir, img) for img in sorted(os.listdir(self.data_dir)) if img.endswith(".png")]
  
        self.test_dataset = ImageFolderDataset(self.image_paths, transform=self.test_transform)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=8)