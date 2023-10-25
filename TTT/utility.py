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
import torch.optim as optim
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


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--dataroot', default='../../autodl-nas/dataset/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--fix_bn', action='store_true')
parser.add_argument('--fix_ssh', action='store_true')
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=1, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--threshold', default=1, type=float)
parser.add_argument('--dset_size', default=0, type=int)
########################################################################
parser.add_argument('--outf', default='.')
parser.add_argument('--resume', default=None)

args = parser.parse_args()



def eval_base():
	logging.basicConfig(filename="./log/ttt_utility_base.log",
					filemode='a',
					format='%(asctime)s, %(name)s %(levelname)s %(message)s',
					datefmt='%Y-%m-%d  %H:%M:%S',
					level=logging.DEBUG)

	cr_list_ori = ["original", "glass_blur", "fog", "contrast"]
	cr_list = ["glass_blur", "fog", "contrast"]
	level_list = [1, 3, 5]

	for level in level_list:
		if level == 1:
			for_list = cr_list_ori
		else:
			for_list = cr_list

		for corruption in for_list :

			teloader_1, teloader_2 = prepare_test_data_ttt(corruption, level)

	 		## LOAD TARGET MODEL ##
			ext_target, head_rot_target, head_cls_target = load_target("res18", "cifar10", 4)
			classifier_target = ExtractorHead(ext_target, head_cls_target)

      			## EVALUATION
			acc_base = test_base_ttt(teloader_2, classifier_target)

			## Logger
			logging.info(f"[Frozen target model Acc], Corruption {corruption}, level {level}:  {acc_base:.2f}")

	logging.info(f"##########################\n\n")



def eval_online():
	logging.basicConfig(filename="./log/ttt_utility_online.log",
						filemode='a',
						format='%(asctime)s, %(name)s %(levelname)s %(message)s',
						datefmt='%Y-%m-%d  %H:%M:%S',
						level=logging.DEBUG)

	cr_list = ["original", "glass_blur", "fog", "contrast"]
	level_list = [1,3,5]

	for num in [0]: ## You can set the number of warming-up i.i.d. samples

		mean_list = []
		std_list = []
		logging.info(f"Warmup with {num}")

		for level in level_list:

			if level == 1:
				for_list = cr_list
			else:
				for_list = cr_list

			for corruption in for_list:

				acc_list = []
				for i in range(3):

					setseed(i+1024)
          
					teloader_1, teloader_2 = prepare_test_data_ttt(corruption, level)

					### Load model ###
					ext_target, head_rot_target, head_cls_target = load_target("res18", "cifar10", 4)

					for batch_idx, (inputs, labels) in enumerate(teloader_1):
						if batch_idx < num:
							inputs = inputs.cuda()
							adapt_single_ttt(inputs, ext_target, head_rot_target)
						else:
							break

					acc = test_online_ttt(teloader_2, ext_target, head_rot_target, head_cls_target)
					acc_list.append(acc)

				arr = numpy.array(acc_list)
				acc_mean = round(numpy.mean(arr), 2)
				acc_std =  round(numpy.std(arr), 2)

				logging.info(f"  Corruption: {corruption} (lv.{level}) Acc (mean: {acc_mean}, std: {acc_std})")
				
				mean_list.append(acc_mean)
				std_list.append(acc_std)

		logging.info(f" Mean: {mean_list}")
		logging.info(f" Std: {std_list}")




if __name__ == '__main__':

	## Evaluate the performance of the frozen target model (Table 1)
	eval_base()
	
	## ## Evaluate the performance of the frozen target model (Figure 4&22)
	eval_online()
	



