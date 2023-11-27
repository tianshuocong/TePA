import logging
import torch
import torch.optim as optim
from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy
import tent
import norm
import os
from conf import cfg, load_cfg_fom_args
import numpy
from tools_tent import *

from models.resnet import resnet18
from setup import *

#logger = logging.getLogger(__name__)



def eval_base():

    logging.basicConfig(filename="./log/tent_1_utility_base.log",
            filemode='a',
            format='%(asctime)s, %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d  %H:%M:%S',
            level=logging.DEBUG)

    load_cfg_fom_args()

    base_model = resnet18(pretrained=True).cuda()

    # Base Acc
    cr_list_ori = ["original", "gaussian_noise", "glass_blur", "fog", "contrast"]
    cr_list = ["gaussian_noise", "glass_blur", "fog", "contrast"]
    level_list = [1,3,5]

    for level in level_list:

        logging.info(f"  Level: {level}")

        if level == 1:
            for_list = cr_list_ori
        else:
            for_list = cr_list    

        acc_base_list = []
        for corruption in for_list:

            #logging.info(f"Dataset: {corruption} (level {level})")

            teloader_1, teloader_2 = prepare_test_data(corruption, level, "norm_true")
            acc_base = test_base_tent(teloader_2, base_model)
            acc_base_list.append(acc_base)
    
            logging.info(f"[Base Acc], Corruption {corruption}, level {level}:  {acc_base:.2f}")

        logging.info(f"  Base Acc: {acc_base_list}")


def eval_online():

    logging.basicConfig(filename="./log/tent_1_utility_online.log",
            filemode='a',
            format='%(asctime)s, %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d  %H:%M:%S',
            level=logging.DEBUG)

    load_cfg_fom_args()


    cr_list = ["fog", "contrast"]
    level = 5
    
    base_model = resnet18(pretrained=True).cuda()
    model = setup_tent(base_model)

    # ## Online Acc
    for num_warmup in [40]:
        
        mean_list = []
        std_list = []
        logging.info(f"\n\n")
        logging.info(f"Warmup with {num_warmup}")

        for corruption in cr_list:

            acc_each_list = []

            for i in range(3):

                setseed(i+1024)

                model.reset()
                teloader_1, teloader_2 = prepare_test_data(corruption, level, "norm_true")

                acc_list = []
            
                ## warmup
                for batch_idx, (inputs, labels) in enumerate(teloader_1):
                    if batch_idx < num_warmup:
                        inputs, labels = inputs.cuda(), labels.cuda() 
                        _ = model(inputs)
                    else:
                        break

                acc_online = test_tent(teloader_2, model)
                acc_each_list.append(acc_online)

            arr = numpy.array(acc_each_list)
            acc_mean = round(numpy.mean(arr), 2)
            acc_std =  round(numpy.std(arr), 2)
            logging.info(f"  Corruption: {corruption} (lv.{level}) Acc (mean: {acc_mean}, std: {acc_std})")
        

            mean_list.append(acc_mean)
            std_list.append(acc_std)

        logging.info(f" Mean: {mean_list}")
        logging.info(f" Std: {std_list}")



if __name__ == '__main__':

    eval_base()

    eval_online()

    
