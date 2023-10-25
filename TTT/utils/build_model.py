import torch
from models import *
import torch.nn.functional as F
import copy

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer

def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)


class ViewFlatten(nn.Module):
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ExtractorHead(nn.Module):
    def __init__(self, ext, head):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head

    def forward(self, x):
        return self.head(self.ext(x))


class head(torch.nn.Module):
    def __init__(self, insize, outsize):
        super(head, self).__init__()
        self.backdone = nn.Sequential(
            nn.Linear(insize, outsize), 
        )
    def forward(self, x):
        y = self.backdone(x)
        return y

class head_deep(torch.nn.Module):
    def __init__(self, insize, outsize):
        super(head_deep, self).__init__()
        self.backdone = nn.Sequential(
            nn.Linear(insize, 100), 
            nn.ReLU(True),
            nn.Linear(100, outsize)
        )
    def forward(self, x):
        y = self.backdone(x)
        return y


class make_head(nn.Module):
    def __init__(self, layers, insize, outsize):
        super(make_head, self).__init__()
        self.layers = layers
        # self.insize = insize
        # self.outsize = outsize
        self.ln = nn.Linear(insize, outsize)
    def forward(self, x):
        out = self.layers(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.ln(out)
        return out


class make_ext(nn.Module):
    def __init__(self, layers):
        super(make_ext, self).__init__()
        self.layers = layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(x)
        return out

def build_target(arch, dataset, layer_branch):

    if arch == 'res18':
        net = ResNet18()
        embedding = 512
    elif arch == 'res50':
        net = ResNet50()
        embedding = 2048


    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn = get_layer(net, name)
            gn = nn.GroupNorm(8, bn.num_features)
            set_layer(net, name, gn)
    
    # net.conv1, net.bn1, net.layer1, net.layer2, net.layer3, net_layer4
    if layer_branch == 2:
        # print("Build from layer 2")
        layers = copy.deepcopy([net.conv1, net.bn1, net.layer1, net.layer2])
        ext = make_ext(nn.Sequential(*layers))
        layer_1 = copy.deepcopy([net.layer3, net.layer4])        
        if dataset == "cifar10":
            head_cls = make_head(nn.Sequential(*layer_1), embedding, 10)
        elif dataset == "cifar100":
            head_cls = make_head(nn.Sequential(*layer_1), embedding, 100)
        layer_2 = copy.deepcopy([net.layer3, net.layer4])
        head_rot = make_head(nn.Sequential(*layer_2), embedding, 4)
        
    elif layer_branch == 3:
        # print("Build from layer 3")
        layers = copy.deepcopy([net.conv1, net.bn1, net.layer1, net.layer2, net.layer3])
        ext = make_ext(nn.Sequential(*layers))

        layer_1 = copy.deepcopy(net.layer4)        
        if dataset == "cifar10":
            head_cls = make_head(nn.Sequential(*layer_1), embedding, 10)
        elif dataset == "cifar100":
            head_cls = make_head(nn.Sequential(*layer_1), embedding, 100)

        layer_2 = copy.deepcopy(net.layer4)
        head_rot = make_head(nn.Sequential(*layer_2), embedding, 4)

    elif layer_branch == 4:
        # print("Build from layer 2")
        ext = net
        if dataset == "cifar10":
            head_cls = head(embedding, 10)
        elif dataset == "cifar100":
            head_cls = head(embedding, 100)

        head_rot = head(embedding, 4)

    return ext, head_rot, head_cls


def load_target(arch = "resnet18", dataset = "cifar10", branch = 4):

    ext, head_rot, head_cls = build_target(arch, dataset, branch)

    cp_ext = torch.load("./checkpoints/" + dataset + "/target/" + arch + "_layer"+str(branch)+"/ext.pth", map_location=device)
    cp_head_cls = torch.load("./checkpoints/" + dataset + "/target/" + arch + "_layer"+str(branch)+"/head_cls.pth", map_location=device)
    cp_head_rot = torch.load("./checkpoints/" + dataset + "/target/" + arch + "_layer"+str(branch)+"/head_rot.pth", map_location=device)

    ext.load_state_dict(cp_ext, strict=True)
    head_cls.load_state_dict(cp_head_cls, strict=True)
    head_rot.load_state_dict(cp_head_rot, strict=True)

    net_rot = ExtractorHead(ext, head_rot)
    net_cls = ExtractorHead(ext, head_cls)

    return ext, head_rot, head_cls









