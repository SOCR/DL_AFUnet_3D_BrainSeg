import torch
import torch.utils.data
import numpy as np
from PIL import Image
import os
import cv2
import time
import random
import nibabel as nib
import torchvision
from skimage.transform import resize


def load_3d(train=True):
  if train == True:
    root = '/content/drive/My Drive/BraTS2021_Training_Data/'
    list1 = os.listdir(root)
    list1 = list1[:300]

    data_dict = {}
    data_dict['X_train'] = torch.zeros(300, 4, 128, 128, 64)
    data_dict['y_train'] = torch.zeros(300, 1, 128, 128, 64)

    k = 0
    for i in list1:
        current_file = os.path.join(root, i)

        test_load_flair = nib.load(os.path.join(current_file, (i+'_flair.nii'))).get_fdata()
        test_load_seg = nib.load(os.path.join(current_file, (i + '_seg.nii'))).get_fdata()
        test_load_t1 = nib.load(os.path.join(current_file, (i + '_t1.nii'))).get_fdata()
        test_load_t1ce = nib.load(os.path.join(current_file, (i + '_t1ce.nii'))).get_fdata()
        test_load_t2 = nib.load(os.path.join(current_file, (i + '_t2.nii'))).get_fdata()

        test_load_flair = resize(test_load_flair, (128, 128, 64))
        test_load_seg = resize((test_load_seg==1)+(test_load_seg==2)+(test_load_seg==4), (128, 128, 64))
        test_load_t1 = resize(test_load_t1, (128, 128, 64))
        test_load_t1ce = resize(test_load_t1ce, (128, 128, 64))
        test_load_t2 = resize(test_load_t2, (128, 128, 64))

        test_load_seg = torch.tensor(test_load_seg)
        test_load_flair = torch.tensor(test_load_flair)
        test_load_t1 = torch.tensor(test_load_t1)
        test_load_t1ce = torch.tensor(test_load_t1ce)
        test_load_t2 = torch.tensor(test_load_t2)

        test_load_seg = test_load_seg.to('cuda')
        test_load_flair = test_load_flair.to('cuda')
        test_load_t1 = test_load_t1.to('cuda')
        test_load_t1ce = test_load_t1ce.to('cuda')
        test_load_t2 = test_load_t2.to('cuda')

        test_load_seg = test_load_seg.to(torch.float32)
        test_load_flair = test_load_flair.to(torch.float32)
        test_load_t1 = test_load_t1.to(torch.float32)
        test_load_t1ce = test_load_t1ce.to(torch.float32)
        test_load_t2 = test_load_t2.to(torch.float32)

        test_load_seg = test_load_seg.view(1, 128, 128, 64)
        test_load_flair = test_load_flair.view(1, 128, 128, 64)
        test_load_t1 = test_load_t1.view(1, 128, 128, 64)
        test_load_t1ce = test_load_t1ce.view(1, 128, 128, 64)
        test_load_t2 = test_load_t2.view(1, 128, 128, 64)

        data_dict['X_train'][k] = torch.cat((test_load_flair, test_load_t1, test_load_t1ce, test_load_t2), 0)
        data_dict['y_train'][k] = test_load_seg
        k+=1
        print(k)
  else:
    root = '/content/drive/My Drive/BraTS2021_Testing_Data/'
    list1 = os.listdir(root)
    # list1.remove('.DS_Store')
    # list_random = random.sample(list1, 300)
    # list1 = list1[400:]

    data_dict = {}
    data_dict['X_train'] = torch.zeros(50, 4, 128, 128, 64)
    data_dict['y_train'] = torch.zeros(50, 1, 128, 128, 64)

    k = 0
    for i in list1:
        current_file = os.path.join(root, i)

        test_load_flair = nib.load(os.path.join(current_file, (i+'_flair.nii'))).get_fdata()
        test_load_seg = nib.load(os.path.join(current_file, (i + '_seg.nii'))).get_fdata()
        test_load_t1 = nib.load(os.path.join(current_file, (i + '_t1.nii'))).get_fdata()
        test_load_t1ce = nib.load(os.path.join(current_file, (i + '_t1ce.nii'))).get_fdata()
        test_load_t2 = nib.load(os.path.join(current_file, (i + '_t2.nii'))).get_fdata()

        test_load_flair = resize(test_load_flair, (128, 128, 64))
        test_load_seg = resize((test_load_seg==1)+(test_load_seg==2)+(test_load_seg==4), (128, 128, 64))
        test_load_t1 = resize(test_load_t1, (128, 128, 64))
        test_load_t1ce = resize(test_load_t1ce, (128, 128, 64))
        test_load_t2 = resize(test_load_t2, (128, 128, 64))

        test_load_seg = torch.tensor(test_load_seg)
        test_load_flair = torch.tensor(test_load_flair)
        test_load_t1 = torch.tensor(test_load_t1)
        test_load_t1ce = torch.tensor(test_load_t1ce)
        test_load_t2 = torch.tensor(test_load_t2)

        test_load_seg = test_load_seg.to('cuda')
        test_load_flair = test_load_flair.to('cuda')
        test_load_t1 = test_load_t1.to('cuda')
        test_load_t1ce = test_load_t1ce.to('cuda')
        test_load_t2 = test_load_t2.to('cuda')

        test_load_seg = test_load_seg.to(torch.float32)
        test_load_flair = test_load_flair.to(torch.float32)
        test_load_t1 = test_load_t1.to(torch.float32)
        test_load_t1ce = test_load_t1ce.to(torch.float32)
        test_load_t2 = test_load_t2.to(torch.float32)

        test_load_seg = test_load_seg.view(1, 128, 128, 64)
        test_load_flair = test_load_flair.view(1, 128, 128, 64)
        test_load_t1 = test_load_t1.view(1, 128, 128, 64)
        test_load_t1ce = test_load_t1ce.view(1, 128, 128, 64)
        test_load_t2 = test_load_t2.view(1, 128, 128, 64)

        data_dict['X_train'][k] = torch.cat((test_load_flair, test_load_t1, test_load_t1ce, test_load_t2), 0)
        data_dict['y_train'][k] = test_load_seg
        k+=1
        print(k)


  return data_dict


def load_training_samples(divide=1):
    """
    This is a function used to load the original and target images.
    """
    path = '/content/drive/My Drive/TCGA-LGG/original/'
    path2 = '/content/drive/My Drive/TCGA-LGG/mask/'
    training_set = {}
    target_set = {}
    list_dir = os.listdir(path)
    
    for i in range(len(list_dir)//divide):
      pic = Image.open(os.path.join(path, list_dir[i]))
      img = np.array(pic)
      training_set[i] = img
      pic2 = Image.open(os.path.join(path2, list_dir[i]))
      img2 = np.array(pic2)
      target_set[i] = img2
      # print(i)
    return training_set, target_set


def data_dict_transform(train, target, divide=1):
    """
    This is a function for turn the data_dict into a numpy array like object
    """
    data_dict = {}
    k = -1
    data_dict['X_train'] = torch.ones(len(train)//divide, 256, 256, 3)
    data_dict['y_train'] = torch.ones(len(train)//divide, 256, 256, 1)
    for i in range(len(train)//divide):
      k += 1
      data_dict['X_train'][k, :] = train[i]
      data_dict['y_train'][k, :] = target[i]
    data_dict['X_train'] = data_dict['X_train'].permute(0, 3, 1, 2)
    data_dict['y_train'] = data_dict['y_train'].permute(0, 3, 1, 2)
    return data_dict


