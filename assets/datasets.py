import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
import imageio
import numpy as np
from utils import translation, translation_short, calculate_mean_and_std


mean, std = calculate_mean_and_std('E:/NEW_CUBE_DATASET/train/cube_rotation1',
                           'E:/NEW_CUBE_DATASET/train/cube_rotation2',
                           'E:/NEW_CUBE_DATASET/val/cube_rotation')

print("Finish Calculating Mean and Std")

normalize = transforms.Normalize(
    mean=mean,
    std=std
)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


class trainset(Dataset):
    def __init__(self, img_path_1, img_path_2, target_path_1, target_path_2):
        #定义好 image 的路径
        self.img1 = img_path_1
        self.img2 = img_path_2
        self.target1 = target_path_1
        self.target2 = target_path_2

    def __getitem__(self, index):
        #print(os.path.join(self.img1, 'render') + str(index) + '.png')
        #print(os.path.join(self.img2, 'render') + str(index) + '.png')
        img1 = imageio.imread(os.path.join(self.img1, 'render') + str(index) + '.png')[:,:,:3]
        img2 = imageio.imread(os.path.join(self.img2, 'render') + str(index) + '.png')[:,:,:3]
        img1 = preprocess(img1)
        img2 = preprocess(img2)
        length1 = len(np.loadtxt(os.path.join(self.target1, 'render') + str(index) + '.csv', dtype=np.string_)[0])
        if length1 == 35:
            target1 = translation_short(self.target1, index)
        elif length1 == 99:
            target1 = translation(self.target1, index)
        else:
            print("Loading data with error!")
        length2 = len(np.loadtxt(os.path.join(self.target2, 'render') + str(index) + '.csv', dtype=np.string_)[0])
        if length2 == 35:
            target2 = translation_short(self.target2, index)
        elif length2 == 99:
            target2 = translation(self.target2, index)
        else:
            print("Loading data with error!")
        target1 = torch.from_numpy(target1)
        target2 = torch.from_numpy(target2)
        return img1, img2, target1, target2

    def __len__(self):
        return len(os.listdir(self.img1))



class valset(Dataset):
    def __init__(self, img_path, target_path):
        #定义好 image 的路径
        self.img = img_path
        self.target = target_path

    def __getitem__(self, index):
        img = imageio.imread(os.path.join(self.img, 'render') + str(index) + '.png')[:,:,:3]
        img = preprocess(img)
        target = translation_short(self.target, index)
        target = torch.from_numpy(target)
        return img, target

    def __len__(self):
        return len(os.listdir(self.img))