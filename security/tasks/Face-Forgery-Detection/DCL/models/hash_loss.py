import torch
import torch.nn as nn
import numpy as np
import random, string

class Hash_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, global_feature_1, global_feature_2):

        B, S = global_feature_1.size()
        positive_pair_difference = torch.sum( torch.abs( global_feature_1 - global_feature_2 ), dim = 1 ) / S
        positive_pair_difference =  torch.sum( positive_pair_difference ) / B

        return torch.exp(positive_pair_difference)

def Hash(self, _input):
    _input = torch.abs( _input )
    return torch.round( 1 / ( 1 + torch.exp( -1 * 1 * ( _input - torch.median( _input, 1, keepdim = True )[0] ) ) ) )
