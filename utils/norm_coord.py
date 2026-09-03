import torch

def make_coord(shape, ranges=None, flatten=True):
    """ 生成中心化坐标
    ranged in [-1, 1]
    e.g.
    shape:图像的大小，例如[512,512,256]
    例如第一维是2，那第一维的坐标就会变成[-0.5,0.5]
        shape = [2] get (-0.5, 0.5)
        shape = [3] get (-0.67, 0, 0.67)
    """
    coord_seqs = []#用于收集不同维度的归一化坐标
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (n - 1)
        seq = v0 + r * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1) # [H, W, Z, 3]
    if flatten:
        ret = ret.view(-1, ret.shape[-1]) # [H*W*Z, 3]
    return ret
