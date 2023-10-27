from __future__ import print_function
import sys
sys.path.append("../..") 
import argparse
from tqdm import tqdm
from PIL import Image
from subprocess import call
import os
import torch
import torch.nn as nn

import logging
from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
from utils.rotation import *
from utils.build_model import *
from utils.evaluation import *
import copy
import numpy
import random
import os
from ttt_attacks import *

import torchvision.transforms as T


def load_local_ttt(branch, dirname):

  ext, head_rot, head_cls = build_target(arch="res18", dataset="cifar10", layer_branch=branch)

  cp_ext = torch.load(dirname + "/ext.pth")
  cp_head_rot = torch.load(dirname + "/head_rot.pth")

  ext.load_state_dict(cp_ext, strict=True)
  head_rot.load_state_dict(cp_head_rot, strict=True)

  net_rot = ExtractorHead(ext, head_rot)
  print("Load Lovel successfully!")

  return net_rot



def attack_online():

  logging.basicConfig(filename="/root/code/res18_C10/ttt/log/ttt_2_attack2num_local_layer3.log",
            filemode='a',
            format='%(asctime)s, %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d  %H:%M:%S',
            level=logging.DEBUG)

  cr_list = ["original", "glass_blur", "fog", "contrast"]
  level_list = [5]

  milestone = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

  for level in level_list:

    for corruption in cr_list:

      logging.info(f"\n\n\n")
      logging.info(f"Corruption: {corruption} (lv.{level}) ")

      acc_list = [[] for i in range(3)]

      for i in range(3):

        setseed(i+1024)

        seed_loader, _ = prepare_test_data_ttt("original", 1, "norm_false")
        _, eval_loader = prepare_test_data_ttt(corruption, level, "norm_false")

        ### Load model : head output dim 4 ###
        ext_target, head_rot_target, head_cls_target = load_target("res18", "cifar10", 4)
        
        #net_local = load_local_ttt(branch=4, dirname = "/root/code/res18_C10/ttt/checkpoints/cinic/surrogate/res18_layer4_200_128/")
        net_local = load_local_ttt(branch=3, dirname = "/root/code/res18_C10/ttt/checkpoints/cinic/surrogate/res18_layer3_200_128/")
        ######################################

        for batch_idx, (inputs_seed, labels) in enumerate(seed_loader):
          if batch_idx < 105:
            if batch_idx in milestone:
              print(f"checking performance... at num {batch_idx}")
              ext_eval = copy.deepcopy(ext_target)
              head_rot_eval = copy.deepcopy(head_rot_target)
              head_cls_eval = copy.deepcopy(head_cls_target)

              
              acc = test_online_ttt(eval_loader, ext_eval, head_rot_eval, head_cls_eval)
              acc_list[i].append(acc)
              logging.info(f"poison {batch_idx}, acc is {acc}")
            
            inputs_seed = inputs_seed.cuda()
            inputs_poisoned = DIM(inputs_seed, net_local)
            adapt_single_ttt(inputs_poisoned, ext_target, head_rot_target)
          else:
            break

      res = numpy.array(acc_list)
      acc_mean = numpy.mean(res, axis=0)
      acc_std = numpy.std(res, axis=0)
      logging.info(f"Mean : {numpy.round(acc_mean,2)}")
      logging.info(f"Std  : {numpy.round(acc_std,2)}")



if __name__ == '__main__':

  attack_online()
  



