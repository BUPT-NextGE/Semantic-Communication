import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
class ImageFolder_stl(Dataset):
    """
    Image shape is (96, 96, 3)  --> 1x1 128x128 x 3 patches
    输入文件夹路径，文件夹里是一堆图片
    返回这些图片每一张的img(tensor0-1),patch(tensor0-1),path单张图片的路径
    训练集loader
    """

    def __init__(self, folder_path):
        # self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.files = glob.glob('%s/*.*' % folder_path)

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        img = Image.open(path)
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            #transforms.ToTensor(),
            # transforms.Normalize([0.4431991, 0.42826223, 0.39535823], [0.25746644, 0.25306803, 0.26591763])
        ])
        img = transform(img)
        # pad = ((24, 24), (0, 0), (0, 0))

        # img = np.pad(img, pad, 'constant', constant_values=0) / 255
        # img = np.pad(img, pad, mode='edge') / 255.0
        img = np.array(img)
        # print(img.shape)
        # img = np.transpose(img, (2, 0, 1))
        # img = torch.from_numpy(img).float()#from_numpy完成numpy到tensor转变
        # img = img.float()
        # 转成numpy才能reshape

        patches = np.reshape(img, (3, 1, 128, 1, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))
        img = torch.from_numpy(img).float()  # 不改变类型，

        return img, patches, path

    def get_random(self):
        i = np.random.randint(0, len(self.files))
        return self[i]

    def __len__(self):
        return len(self.files)

class ImageFolder_imagenet(Dataset):
    '''
     Image shape 是未知的imagenet图片  --> 1x1 256*256 x 3 patches
    输入文件夹路径，文件夹里是一堆图片
    返回这些图片每一张的img(tensor0-1),patch(tensor0-1),path单张图片的路径
    '''



    def __init__(self, folder_path):
        # self.files = sorted(glob.glob('%s/*.*' % folder_path))#%s的位置将会被替换成新的字符串，替换的内容是%后面的folder_path
        self.files = glob.glob('%s/*.*' % folder_path)
    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        img = Image.open(path)
        # h, w, c = img.shape
        transform = transforms.Compose([
            transforms.Resize((256, 256)),#resize函数必须是PIL才可以
            transforms.ToTensor(),#0-255到0-1
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            #transforms.ToTensor(),
            # transforms.Normalize([0.4431991, 0.42826223, 0.39535823], [0.25746644, 0.25306803, 0.26591763])
        ])

        img = transform(img)
        # pad = ((24, 24), (0, 0), (0, 0))

        # img = np.pad(img, pad, 'constant', constant_values=0) / 255
        # img = np.pad(img, pad, mode='edge') / 255.0
        img = np.array(img)
        # print(img.shape)
        # img = np.transpose(img, (2, 0, 1))
        # img = torch.from_numpy(img).float()#from_numpy完成numpy到tensor转变
        # img = img.float()
        #转成numpy才能reshap

        patches = np.reshape(img, (3, 1, 256, 1, 256))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))
        img = torch.from_numpy(img).float()#不改变类型，仍在0-255之间
        patches = torch.from_numpy(patches).float()
        return img, patches, path

    def get_random(self):
        i = np.random.randint(0, len(self.files))
        return self[i]

    def __len__(self):
        return len(self.files)