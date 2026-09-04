import torch

def make_coord(shape, ranges=None, flatten=True):
"""
Generate centered coordinates in the range [-1, 1].

Args:
    shape: Image dimensions, e.g., [512, 512, 256].

For example, if the size of the first dimension is 2,
the coordinates along that dimension will be [-0.5, 0.5].

Examples:
    shape = [2] -> (-0.5, 0.5)
    shape = [3] -> (-0.67, 0, 0.67)
"""
    coord_seqs = []
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
