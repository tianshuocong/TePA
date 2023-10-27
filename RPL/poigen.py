import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from tools_rpl import *


def input_diversity(x):

    resize_rate = 0.8
    diversity_prob = 0.5
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    if resize_rate < 1:
        img_size = img_resize
        img_resize = x.shape[-1]

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

    return padded if torch.rand(1) < diversity_prob else x


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def DIM(image, surrogate_model):

    image = image.cuda()
    eps = 32/255
    alpha = 1/255
    mu = 1.0
    g = 0
    decay=1.0

    momentum = torch.zeros_like(image).detach().cuda()

    ori_image = image.data
    ori_image = ori_image.cuda()

    num_restart = 3

    for rst in range(num_restart):

        noise = torch.from_numpy(np.random.normal(size=ori_image.size(), loc=0, scale=0.8))
        noise = noise.type(torch.FloatTensor)
        noise = torch.clamp(noise, min=-eps, max=eps)
        noise = noise.cuda()
        x_start = torch.clamp(noise+ori_image, min=0, max=1)
        image = x_start

        for xx in range(200):
            
            surrogate_model.train()
            surrogate_model.zero_grad = True
            surrogate_model = surrogate_model.cuda()

            image.requires_grad = True
            output = surrogate_model(input_diversity(image))
            loss_val = softmax_entropy(output).mean(0)

            grad = torch.autograd.grad(loss_val, image, retain_graph=False, create_graph=False)[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum * decay
            momentum = grad

            adv_image = image.detach() + alpha * grad.sign()
            noise = torch.clamp(adv_image - ori_image, min = -eps, max = eps)
            image = torch.clamp(ori_image + noise, min=0, max = 1).detach_()

        if rst == 0:
            loss_best = loss_val
        if loss_val >= loss_best:
            image_final = image

    return image_final




def random_split(id_, loader, x_in, label, total_num, choose_num):

    indices_all = torch.arange(total_num)
    indices_adv = torch.tensor(random.sample(range(total_num), choose_num))

    combined = torch.cat((indices_all, indices_adv))
    uniques, counts = combined.unique(return_counts=True)
    indices_left = uniques[counts == 1]

    #sampled_adv = x_in[indices_adv]
    sampled_left = x_in[indices_left]

    #label_adv = label[indices_adv]
    label_left = label[indices_left]

    for idx, (inputs, labels) in enumerate(loader):
        if idx == id_:
            sampled_adv = inputs[indices_adv].cuda()
            label_adv = labels[indices_adv].cuda()
            break


    return sampled_left, sampled_adv, label_left, label_adv




def poison_data(xin, label_in, p, batch_idx, surrogate_model, iid_data="original", iid_level=1):

    # seed image is original
    if p == 0.0:
        inputs = xin
        labels = label_in

    elif p == 1.0:
        seed_loader, _ = prepare_test_data("original", 1, "norm_false")
        for idx, (inputs_seed, labels) in enumerate(seed_loader):
            if idx == batch_idx:
                x_seed = inputs_seed.cuda()

        inputs = DIM(x_seed, surrogate_model)
        labels = label_in
        NORM = ((0.4913, 0.4821 ,0.4465), (0.2470, 0.2434, 0.261))
        transform_test = transforms.Compose([
             transforms.Normalize(*NORM)
        ])
        inputs = transform_test(inputs)

    else:

        #sample_left, sample_adv = random_split(batch_idx, test_loader_seed, xin, p)

        seed_loader, _ = prepare_test_data("original", 1, "norm_false")
        bs = xin.size(0)

        sample_left, sample_adv, label_left, label_adv = \
                    random_split(batch_idx, seed_loader, xin, label_in, bs, int(p*bs))
        #print(sample_adv.shape)
        inputs_adv = DIM(sample_adv, surrogate_model)
        NORM = ((0.4913, 0.4821 ,0.4465), (0.2470, 0.2434, 0.261))
        transform_test = transforms.Compose([
             transforms.Normalize(*NORM)
        ])
        inputs_adv = transform_test(inputs_adv)

        inputs = torch.cat((inputs_adv, sample_left),0)
        labels = torch.cat((label_adv, label_left),0)

    return inputs
