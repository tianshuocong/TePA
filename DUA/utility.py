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
from models.resnet import resnet18




def eval_base():

    logging.basicConfig(filename="./log/dua_1_utility_base.log",
                            filemode='a',
                            format='%(asctime)s, %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d  %H:%M:%S',
                            level=logging.DEBUG)

    cr_list_ori = ["original", "glass_blur", "fog", "contrast"]
    cr_list = ["glass_blur", "fog", "contrast"]
    level_list = [1,3,5]

    ## base
    acc_base_list = []
    for level in level_list:

        if level == 1:
            for_list = cr_list_ori
        else:
            for_list = cr_list

        for corruption in for_list:

            set_1, set_2, loader_1, loader_2 = prepare_test_data_dua(corruption, level, "norm_false")

            base_model = resnet18(pretrained=True).cuda()
            acc_base = test_base_dua(loader_2, base_model)
            #acc_base = test_base_dua(loader_2, base_model)

            logging.info(f"[Base Acc], Corruption {corruption}, level {level}:  {acc_base:.2f}")
            acc_base_list.append(acc_base)

    logging.info(acc_base_list)




def eval_online():

    logging.basicConfig(filename="./log/dua_1_utility_online.log",
                            filemode='a',
                            format='%(asctime)s, %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d  %H:%M:%S',
                            level=logging.DEBUG)

    cr_list = ["original", "glass_blur", "fog", "contrast"]
    level = 5
  
    for num in [10, 20, 30, 40, 50, 100, 150, 200]: 

        mean_list = []
        std_list = []

        for corruption in cr_list:
            logging.info(f"  Corruption: {corruption} (lv.{level})")
    
            acc_list = []
            for i in range(3):

                setseed(i+1024)
                set_1, set_2, loader_1, loader_2 = prepare_test_data_dua(corruption, level, "norm_false")
                
                net = resnet18(pretrained=True).cuda()
                mom_pre = 0.1

                for batch_idx, (inputs, labels) in enumerate(loader_1):
                    if batch_idx < num:
                        image = inputs.cuda()
                        net, mom_pre = adapt_single(mom_pre, image, net)

                    else:
                        break

                acc = test_base_dua(loader_2, net)
                acc_list.append(acc)

            arr = numpy.array(acc_list)
            acc_mean = round(numpy.mean(arr), 2)
            acc_std =  round(numpy.std(arr), 2)

            logging.info(f"  Warmup: {num} : Acc (mean: {acc_mean}, std: {acc_std})")

            mean_list.append(acc_mean)
            std_list.append(acc_std)

        logging.info(f" Mean: {mean_list}")
        logging.info(f" Std: {std_list}")




if __name__ == '__main__':

    eval_base()    

    eval_online()    

