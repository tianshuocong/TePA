from utils.build_model import *
import torch
from utils.rotation import *


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



def DIM(image, model):

    ## hyper
    iter_out = 3
    iter_in = 20
    eps = 32/255
    mu = 1.0
    g = 0
    decay=1.0

    momentum = torch.zeros_like(image).detach().cuda()
    loss = nn.CrossEntropyLoss()
    
    ori_image = image.data
    ori_image = ori_image.cuda()

    for xx in range(iter_out):
        for ii in range(4):

            label = torch.zeros(1, dtype=torch.long) + ii
            label = label.cuda()

            model.eval()
            model.zero_grad()
            model = model.cuda()

            for i in range(iter_in):

                if i < 10:
                    alpha = 4/255
                elif i < 15:
                    alpha = 2/255
                else:
                    alpha = 1/255

                model.zero_grad()
                image.requires_grad = True
                inputs = torch.rot90(image, ii, (2,3))
                inputs = inputs.cuda()
                output = model(input_diversity(inputs))
                loss_val = loss(output, label)

                grad = torch.autograd.grad(loss_val, image, retain_graph=False, create_graph=False)[0]
                grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
                grad = grad + momentum * decay
                momentum = grad
                #grad = torch.rot90(grad, -ii, (2,3))

                adv_image = image.detach() + alpha * grad.sign()
                noise = torch.clamp(adv_image - ori_image, min = -eps, max = eps)
                image = torch.clamp(ori_image + noise, min=0, max = 1).detach_()

    return image 
