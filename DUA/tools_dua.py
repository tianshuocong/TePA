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
import os
from utils.rotation import *

def my_makedir(name):
	try:
		os.makedirs(name)
	except OSError:
		pass


def prepare_test_data_dua(corruption, level, trans = "norm_true"):

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

    loader_1 = torch.utils.data.DataLoader(set_1, batch_size=1, shuffle=False, num_workers=0)
    loader_2 = torch.utils.data.DataLoader(set_2, batch_size=200, shuffle=True, num_workers=0)

    return set_1, set_2, loader_1, loader_2



def adapt_single(mom_pre, image, model):

    image = torch.squeeze(image)

    ## HYPERPARAMETERS
    NORM = ((0.4913, 0.4821 ,0.4465), (0.2470, 0.2434, 0.261))
    
    tr_transform_adapt = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            transforms.Normalize(*NORM)
        ])
    decay_factor = 0.94
    min_momentum_constant = 0.005
    mom_new = (mom_pre * decay_factor)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d):
            m.train()
            m.momentum = mom_new + min_momentum_constant
    mom_pre = mom_new
    inputs = [(tr_transform_adapt(image)) for _ in range(64)]
    inputs = torch.stack(inputs)
    inputs = inputs.cuda()
    #print(inputs.shape)
    inputs_ssh, labels_ssh = rotate_batch(inputs, 'rand')
    inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
    _ = model(inputs_ssh)
    return model, mom_pre


def test_base_dua(dataloader, net):
    net.eval()
    correct = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        NORM = ((0.4913, 0.4821 ,0.4465), (0.2470, 0.2434, 0.261))
        #NORM = ( (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        #NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform_test = transforms.Compose([
            transforms.Normalize(*NORM)
        ])
        inputs = transform_test(inputs)
        with torch.no_grad():
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    acc = round(correct.mean()*100, 2)
    return acc


def setseed(manualSeed) :
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    os.environ['PYTHONHASHSEED'] = str(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

