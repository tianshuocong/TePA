import sys 
sys.path.append("..") 
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from collections import OrderedDict
import heapq
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import copy
import time


def test_tent(dataloader, model_in):
    correct = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        model = copy.deepcopy(model_in)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct.append(predicted.eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    acc = round(correct.mean()*100, 2)
    return acc


def prepare_test_data(corruption, level, trans = "norm_false"):

    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    NORM = ((0.4913, 0.4821 ,0.4465), (0.2470, 0.2434, 0.261))

    if trans == "norm_true":
        te_transforms = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(), transforms.Normalize(*NORM)])
    elif trans == "norm_false":
        te_transforms = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])

    path = "/root/autodl-nas/"
    tesize = 10000
    if corruption == 'original':
        #print('Test on the original test set')
        teset = torchvision.datasets.CIFAR10(root=path+"dataset/",train=False, download=False, transform=te_transforms)
    elif corruption in common_corruptions:
        #print('Test on %s level %d' %(corruption, level))
        teset_raw = np.load(path+'dataset/CIFAR-10-C/%s.npy' %(corruption))
        teset_raw = teset_raw[(level-1)*tesize: level*tesize]
        teset = torchvision.datasets.CIFAR10(root=path+"dataset/",train=False, download=False, transform=te_transforms)
        teset.data = teset_raw
    else:
        raise Exception('Corruption not found!')

    set_1 = torch.utils.data.Subset(teset, range(9000))
    set_2 = torch.utils.data.Subset(teset, range(9000, 10000))

    loader_1 = torch.utils.data.DataLoader(set_1, batch_size=200, shuffle=True, num_workers=0)
    loader_2 = torch.utils.data.DataLoader(set_2, batch_size=200, shuffle=False, num_workers=0)

    return loader_1, loader_2
