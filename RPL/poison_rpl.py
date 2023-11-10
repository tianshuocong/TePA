import logging
import torch
import torch.optim as optim
import rpl
import norm
import os
from conf import cfg, load_cfg_fom_args
import numpy
from tools_rpl import *
from models.resnet import resnet18
from models.vgg import vgg11_bn
from setup import *
from models_surrogate import *
from poigen import *

def evaluate(description):

    logging.basicConfig(filename="./log/rpl_2_attack2num_last.log",
                filemode='a',
                format='%(asctime)s, %(name)s %(levelname)s %(message)s',
                datefmt='%Y-%m-%d  %H:%M:%S',
                level=logging.DEBUG)

    load_cfg_fom_args()

    base_model = resnet18(pretrained=True).cuda()
    model = setup_rpl(base_model)

    surrogate_model = VGG('VGG11').cuda()
    checkpoint = torch.load("../ckpt/surrogate_vgg11.pth")
    surrogate_model.load_state_dict(checkpoint['net'])
    
    teloader_1, teloader_2 = prepare_test_data("original", 1)

    cr_list = ["original","glass_blur","fog", "contrast"]
    level_list = [5]

    for level in level_list:

        mean_list = []
        std_list = []

        for corruption in cr_list:

            logging.info(f"Dataset: {corruption} (level {level})")

            #for num_warmup in [0,5,10,15,20,25,30,35,40]:
            for num_warmup in [40]:

                acc_each_list = []

                for i in range(1):

                    setseed(i+1024)
                    
                    model.reset()
                    #print(i)
                    seed_loader, _ = prepare_test_data("original", 1, "norm_false")

                    acc_list = []
                
                    ## warmup
                    for batch_idx, (inputs, labels) in enumerate(seed_loader):
                        if batch_idx < num_warmup:
                            inputs, labels = inputs.cuda(), labels.cuda() 
                            inputs = poison_data(inputs, labels, 1.0, batch_idx, surrogate_model)
                            _ = model(inputs)
                        else:
                            break

                    _, eval_loader = prepare_test_data(corruption, level, "norm_true")

                    acc_online = test_online_rpl(eval_loader, model)
                    print(acc_online)
                    acc_each_list.append(acc_online)

                arr = numpy.array(acc_each_list)
                acc_mean = round(numpy.mean(arr), 2)
                acc_std =  round(numpy.std(arr), 2)
                logging.info(f"  Warmup with {num_warmup},  Acc (mean: {acc_mean}, std: {acc_std})")
            

            mean_list.append(acc_mean)
            std_list.append(acc_std)

        logging.info(f" Mean: {mean_list}")
        logging.info(f" Std: {std_list}")

 

if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')
