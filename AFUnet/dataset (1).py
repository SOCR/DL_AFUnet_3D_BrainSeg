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


def load_3d(seed=1):
    """
    This is a function used to load the original and target 3D samples.
    """
    random.seed(seed)
    root = '/content/BraTS2021_Training_Data/'
    list1 = os.listdir(root)
    # list1.remove('.DS_Store')
    list_random = random.sample(list1, 400)

    data_dict = {}
    data_dict_train = {}
    data_dict_val = {}
    data_dict_test = {}
    data_dict['X_train'] = torch.zeros(400, 4, 128, 128, 64)
    data_dict['y_train'] = torch.zeros(400, 1, 128, 128, 64)
    data_dict_train['X_train'] = torch.zeros(300, 4, 128, 128, 64)
    data_dict_train['y_train'] = torch.zeros(300, 1, 128, 128, 64)
    data_dict_val['X_train'] = torch.zeros(50, 4, 128, 128, 64)
    data_dict_val['y_train'] = torch.zeros(50, 1, 128, 128, 64)
    data_dict_test['X_train'] = torch.zeros(50, 4, 128, 128, 64)
    data_dict_test['y_train'] = torch.zeros(50, 1, 128, 128, 64)

    k = 0
    for i in list_random:
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
        # print(k)

    data_dict_train['X_train'] = data_dict['X_train'][:300, :, :, :, :]
    data_dict_train['y_train'] = data_dict['y_train'][:300, :, :, :, :]
    data_dict_val['X_train'] = data_dict['X_train'][300:350, :, :, :, :]
    data_dict_val['y_train'] = data_dict['y_train'][300:350, :, :, :, :]
    data_dict_test['X_train'] = data_dict['X_train'][350:400, :, :, :, :]
    data_dict_test['y_train'] = data_dict['y_train'][350:400, :, :, :, :]

    return data_dict_train, data_dict_val, data_dict_test, data_dict


def load_training_samples(seed=10):
    random.seed(seed)
    """
    This is a function used to load the original and target images.
    """
    # path = '/content/drive/My Drive/TCGA-LGG-PNG/original/'
    # path2 = '/content/drive/My Drive/TCGA-LGG-PNG/mask/'
    path = '/content/TCGA-LGG-PNG/original/'
    path2 = '/content/TCGA-LGG-PNG/mask/'
    training_set = {}
    target_set = {}
    list_dir = os.listdir(path)
    
    for i in range(len(list_dir)):
      pic = cv2.imread(os.path.join(path, list_dir[i]), cv2.IMREAD_UNCHANGED)
      training_set[i] = pic
      pic2 = cv2.imread(os.path.join(path2, list_dir[i]), cv2.IMREAD_UNCHANGED)
      target_set[i] = pic2
      # print(i)
    
    train_inter_pairs = {}
    for i in range(len(training_set)):
      train_inter_pairs[i] = torch.tensor(training_set[i], dtype=torch.float32)

    target_inter_pairs = {}
    for i in range(len(target_set)):
      target_inter_pairs[i] = torch.tensor(target_set[i], dtype=torch.float32)
      target_inter_pairs[i] = target_inter_pairs[i].view(256, 256, 1)

    data_dict = data_dict_transform(train_inter_pairs, target_inter_pairs)

    list_idx = list(range(0, 3929))
    random.shuffle(list_idx)
    train_idx = list_idx[:3000]
    val_idx = list_idx[3000:3429]
    test_idx = list_idx[3429:3929]

    data_dict_train = {}
    data_dict_val = {}
    data_dict_test = {}
    data_dict_train['X_train'] = torch.ones(3429, 3, 256, 256)
    data_dict_train['y_train'] = torch.ones(3429, 1, 256, 256)
    data_dict_val['X_train'] = torch.ones(429, 3, 256, 256)
    data_dict_val['y_train'] = torch.ones(429, 1, 256, 256)
    data_dict_test['X_train'] = torch.ones(500, 3, 256, 256)
    data_dict_test['y_train'] = torch.ones(500, 1, 256, 256)

    data_dict_train['X_train'] = data_dict['X_train'][train_idx, :, :, :]
    data_dict_train['y_train'] = data_dict['y_train'][train_idx, :, :, :]
    data_dict_val['X_train'] = data_dict['X_train'][val_idx, :, :, :]
    data_dict_val['y_train'] = data_dict['y_train'][val_idx, :, :, :]
    data_dict_test['X_train'] = data_dict['X_train'][test_idx, :, :, :]
    data_dict_test['y_train'] = data_dict['y_train'][test_idx, :, :, :]

    return  data_dict_train, data_dict_val, data_dict_test, data_dict


def data_dict_transform(train, target):
    """
    This is a function for turning the training and target data into a dictionary
    """
    data_dict = {}
    k = -1
    data_dict['X_train'] = torch.ones(len(train), 256, 256, 3)
    data_dict['y_train'] = torch.ones(len(train), 256, 256, 1)
    for i in range(len(train)):
      k += 1
      data_dict['X_train'][k, :] = train[i]
      data_dict['y_train'][k, :] = target[i]
    data_dict['X_train'] = data_dict['X_train'].permute(0, 3, 1, 2)
    data_dict['y_train'] = data_dict['y_train'].permute(0, 3, 1, 2)/255
    return data_dict


