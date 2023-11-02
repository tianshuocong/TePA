from __future__ import print_function
import argparse
from argparse import Namespace
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy
from tools_dua import *
import torchvision.transforms as T
from models.resnet import resnet18
from models_kuangliu import *



if __name__ == '__main__':

    transform_to_img = T.ToPILImage()

    logging.basicConfig(filename="./log/dua_poison.log",
                        filemode='a',
                        format='%(asctime)s, %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d  %H:%M:%S',
                        level=logging.DEBUG)

    cr_list_ori = ["glass_blur",  "fog", "contrast"]
    cr_list = ["original", "glass_blur",  "fog", "contrast"]
    level_list = [5]

    flag_visual = 1

    surrogate_model = VGG('VGG11').cuda()
    checkpoint = torch.load("./ckpt/surrogate/cinic10/kuangliu/vgg11-0215.pth")
    surrogate_model.load_state_dict(checkpoint['net'])
   
    ### online
    for level in level_list:

        if level == 1:
            for_list = cr_list_ori
        else:
            for_list = cr_list

        for corruption in cr_list:

            mean_list = []
            std_list = []
            logging.info(f"  Corruption: {corruption} (lv.{level}) ")

            for num in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:  

                acc_list = []

                for i in range(3):

                    setseed(i+1024)
                    _, _, seed_loader, _ = prepare_test_data_dua("original", 1, "norm_false")
                    net = resnet18(pretrained=True).cuda()
                    mom_pre = 0.1
                  
                    #### poison process ####
                    for batch_idx, (inputs, labels) in enumerate(seed_loader):
                        if batch_idx < num:
                            inputs = inputs.cuda()
                            labels = labels.cuda()
                            image = torch.squeeze(inputs)
                            noise = torch.from_numpy(np.random.normal(size=image.size(), loc=0, scale=0.8))
                            noise = noise.type(torch.FloatTensor)
                            noise = noise.cuda()
                            image = torch.clamp((32/255)*noise+image, min=0, max=1)
                            net, mom_pre = adapt_single(mom_pre, image, net)

                        else:
                            break
                    ###########################

                    _, _, _, eval_loader = prepare_test_data_dua(corruption, level, "norm_false")
                    acc = test_base_dua(eval_loader, net)
                    acc_list.append(acc)

                arr = numpy.array(acc_list)
                acc_mean = round(numpy.mean(arr), 2)
                acc_std =  round(numpy.std(arr), 2)
                

                logging.info(f"  Number: {num} Acc (mean: {acc_mean}, std: {acc_std})")

                mean_list.append(acc_mean)
                std_list.append(acc_std)

            logging.info(f" Mean: {mean_list}")
            logging.info(f" Std: {std_list}")
            flag_visual = 0
