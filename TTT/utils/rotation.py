import torch
import torch.utils.data
import numpy as np


def rotate_batch_with_labels(batch, labels):
    length = int(len(batch)/4)
    img_0 = batch[0:length]
    img_90 = torch.rot90(batch[length:2*length], 1, (2,3))
    img_180 = torch.rot90(batch[2*length:3*length], 2, (2,3))
    img_270 = torch.rot90(batch[3*length:4*length], 3, (2,3))
    images = torch.cat((img_0, img_90, img_180, img_270), 0)
    return images


def rotate_batch(batch, label):
    if label == 'rand':
        labels = torch.randint(4, (len(batch),), dtype=torch.long)
    elif label == 'expand':
        labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
                    torch.zeros(len(batch), dtype=torch.long) + 1,
                    torch.zeros(len(batch), dtype=torch.long) + 2,
                    torch.zeros(len(batch), dtype=torch.long) + 3])
        batch = batch.repeat((4,1,1,1))
    else:
        assert isinstance(label, int)
        labels = torch.zeros((len(batch),), dtype=torch.long) + label
    return rotate_batch_with_labels(batch, labels), labels
