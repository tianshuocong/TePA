import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np


common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']



def prepare_test_data_ttt(corruption, level, trans = "norm_false"):

    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


    NORM = ((0.4913, 0.4821 ,0.4465), (0.2470, 0.2434, 0.261))

    if trans == "norm_true":
        te_transforms = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(), transforms.Normalize(*NORM)])
    else:
        te_transforms = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])

    path = "/root/autodl-nas/"  ## change it to your own path

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

    loader_1 = torch.utils.data.DataLoader(set_1, batch_size=1, shuffle=True, num_workers=0)
    loader_2 = torch.utils.data.DataLoader(set_2, batch_size=1, shuffle=True, num_workers=0)

    return loader_1, loader_2



def prepare_train_data(args):
  
    print('Preparing data...')
  
    if args.dataset == 'cifar10':
        trset = torchvision.datasets.CIFAR10(root=args.dataroot,train=True, download=True, transform=tr_transforms)
    else:
        raise Exception('Dataset not found!')
      
    if not hasattr(args, 'workers'):
        args.workers = 1
      
    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,shuffle=True, num_workers=args.workers)
  
    return trset, trloader
    
