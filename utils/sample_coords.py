import numpy as np
import torch
import random

def sample_coords(label,ignore_label=255,sample_num=None,sample_ignore=False):
    '''
    :param label:点标签，要从里面挑选出标记的样本，可能是前景，也可能是背景 ,batch*x*y*z
    :param sample_num: 采样点个数,如果是None，那就采样最少标记点的样本的点个数，如果指定了，那就和最少样本点的样本点个数进行对比，如果小于，则按照指定的个数采样，否则就按照最小样本点采样
    ignore_label:没有标注的点的标记
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






