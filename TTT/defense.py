import torch
import torchvision.transforms as transforms

def BDR(X_in,target_depth):
    r=256/(2**target_depth)
    x_out=torch.round(X_in*255/r)*r/255 
    return x_out

transform = transforms.Compose([transforms.ToTensor()])


def JPEG(X_in, ratio):
        X_out=torch.zeros_like(X_in)
        for j in range(X_in.size(0)):
            x_np=transforms.ToPILImage()(X_in[j].detach().cpu())
            x_np.save('./'+'j.jpg',quality=ratio)
            X_out[j]=transform(Image.open('./'+'j.jpg'))
        return X_out


def R_and_P(X_in):
    rnd = np.random.randint(32, 40,size=1)[0]
    h_rem = 40 - rnd
    w_rem = 40 - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left
    X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
    return  X_out 
