import numpy as np
import torch
import random

def sample_coords(label,ignore_label=255,sample_num=None,sample_ignore=False):
'''
:param label:
    Point labels used to select annotated samples. The labeled points
    may belong to either the foreground or the background.
    Shape: batch * x * y * z.

:param sample_num:
    Number of points to sample.
    If None, the number of sampled points is determined by the sample
    with the fewest labeled points.
    If specified, sample_num is compared with the minimum number of
    labeled points among all samples. If sample_num is smaller, the
    specified number of points is sampled; otherwise, the minimum
    number of labeled points is used.

:param ignore_label:
    Label value used for unlabeled points.

:return:
'''
    batch_coords_list = []#batch_size,N,3
    for sample in label:
        sample=sample[0]
        if sample_ignore:
            nonzero_coords = torch.nonzero(sample==ignore_label)
        else:
            nonzero_coords = torch.nonzero(sample != ignore_label)
        batch_coords_list.append(nonzero_coords)
    min_N=min([len(i) for i in batch_coords_list])

    if any([sample_num==None,sample_num>min_N]):
        sample_num=min_N
    d=[i for i in range(min_N)]
    sample_id=random.sample(d,sample_num)

    #coords=[random.sample(list(i),sample_num) for i in batch_coords_list]#batch,N,3
    coords=[torch.tensor(i[sample_id,:]) for i in batch_coords_list]
    coords=torch.stack(coords,dim=0)
    coords_=coords.to(torch.long)
    B,_,_=coords.shape
    label_coords=[]
    for i in range(B):
        label_coords.append(label[i, :, coords_[i, :, 0], coords_[i, :, 1], coords_[i, :, 2]])
    label_coords=torch.stack(label_coords,dim=0)

    return coords,label_coords






