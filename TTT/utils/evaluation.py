import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from utils.build_model import *
import torch.optim as optim
from utils.rotation import *
from utils.prepare_dataset import *

def adapt_single_ttt(image, ext, head):
    image = torch.squeeze(image)
    NORM = ((0.4913, 0.4821 ,0.4465), (0.2470, 0.2434, 0.261))
    tr_transforms_adapt = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize(*NORM)])                          
    head.eval()
    ext.train()
    model = ExtractorHead(ext, head)
    optimizer_adapt = optim.SGD(model.parameters(), lr = 0.001)
    criterion_ssh = nn.CrossEntropyLoss().cuda()
    model = model.cuda()
    for iteration in range(1):
        inputs = [tr_transforms_adapt(image) for _ in range(32)]  ## small batch
        inputs = torch.stack(inputs)
        inputs_ssh, labels_ssh = rotate_batch(inputs, 'expand')
        inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
        optimizer_adapt.zero_grad()
        outputs = model(inputs_ssh)
        loss_ssh = criterion_ssh(outputs, labels_ssh)
        loss_ssh.backward()
        optimizer_adapt.step()
    return loss_ssh


def test_base_ttt(dataloader, net):
    net.eval()
    net = net.cuda()
    correct = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        NORM = ((0.4913, 0.4821 ,0.4465), (0.2470, 0.2434, 0.261))
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


def test_online_ttt(dataloader, ext_in, head_rot_in, head_cls_in):
    correct = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        ext = copy.deepcopy(ext_in)
        head_rot =  copy.deepcopy(head_rot_in)
        head_cls =  copy.deepcopy(head_cls_in)
        adapt_single_ttt(inputs, ext, head_rot)
        NORM = ((0.4913, 0.4821 ,0.4465), (0.2470, 0.2434, 0.261))
        transform_test = transforms.Compose([
            transforms.Normalize(*NORM)
        ])
        inputs = transform_test(inputs)
        net = ExtractorHead(ext, head_cls)
        net = net.cuda()
        net.eval()
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        correct.append(predicted.eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    acc = round(correct.mean()*100, 2)
    return acc

