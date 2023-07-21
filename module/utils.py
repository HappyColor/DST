
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
import warnings
    
def create_PositionalEncoding(input_dim, max_seq_len=2000): 
    position_encoding = np.array([ 
        [pos / np.power(10000, 2.0 * (j // 2) / input_dim) for j in range(input_dim)] 
        for pos in range(max_seq_len)]) 
    
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

    position_encoding = torch.from_numpy(position_encoding.astype(np.float32))
    position_encoding = nn.Parameter(position_encoding, requires_grad=False) 
    
    return position_encoding

def _get_activation_fn(activation: str='relu', module: bool=False):
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return nn.ReLU() if module else F.relu
    elif activation == "gelu":
        return nn.GELU() if module else F.gelu
    elif activation == "tanh":
        return nn.Tanh() if module else torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

def add_position(x, position=None, mask=None):
    '''add position information to the input x

    x: B, T, C
    position: T, C
    mask: B, T
    '''
    if position is None:
        return x
    else:
        B, T = x.shape[:2]
        position = position[:T].unsqueeze(dim=0).repeat(B, 1, 1)  # -> B, T, C
        position = position*((1 - mask.unsqueeze(-1).type_as(x))) if mask is not None else position
        return x + position

@torch.no_grad()
def arbitrary_segment_mask(start_index: torch.Tensor, end_index: torch.Tensor, len_out: int, reverse: bool = False, return_bool: bool = True):
    '''
    Args:
        start_index: (b t), the first true value
        end_index: (b t), the last true value
        len_out: length of mask 
        reverse: reverse the output mask (Default: False)
        return_bool: if True, return torch.BoolTensor, otherwise torch.FloatTensor
    Returns:
        mask: (b t len_out), the padded values are marked as True if reverse is False
    '''
    b, t = start_index.shape
    start_index = start_index.unsqueeze(dim=-1)  # b t 1
    end_index = end_index.unsqueeze(dim=-1)

    mask = torch.arange(0, len_out, device=start_index.device).unsqueeze(dim=0).unsqueeze(dim=1).expand(b, t, -1)
    mask = ((mask - start_index) < 0) | ((mask - end_index) > 0)

    if reverse:
        mask = ~mask

    if not return_bool:
        mask = mask.float()

    return mask

@torch.no_grad()
def arbitrary_point_mask(index: torch.Tensor, len_out: int, reverse: bool = False, return_bool: bool = True):
    '''
    Args:
        index: (b t), index of the masked point
        len_out: length of mask 
        reverse: reverse the output mask (Default: False)
        return_bool: if True, return torch.BoolTensor, otherwise torch.FloatTensor
    Returns:
        mask: (b t len_out), the specified points are marked as True if reverse is False
    '''
    b, t = index.shape
    index = index.unsqueeze(dim=-1)  # b t 1

    mask = torch.arange(0, len_out, device=index.device).unsqueeze(dim=0).unsqueeze(dim=1).expand(b, t, -1)
    mask = (mask - index) == 0

    if reverse:
        mask = ~mask

    if not return_bool:
        mask = mask.float()

    return mask

def inverse_sigmoid(y):
    assert 0 < y < 1, f'y out of range (0 < y < 1), current y = {y}'
    return math.log(y/(1-y))
