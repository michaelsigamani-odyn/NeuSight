from ..model import ModelBase
import torch
import pandas as pd

class HeuristicMM(ModelBase):
    def __init__(self, config, tag=None, device="cuda"):
        super().__init__(config,tag=tag,device=device) # name and feature

        self.f2i_dict = dict(zip(self.features, range(len(self.features))))

    def get_feature(self, x, features):
        features_idx = self.f2i(features)
        x = x[:,features_idx]
        x = x.view(-1, len(features_idx))
        x = x
        return x

    def f2i(self, feature):
        if isinstance(feature, list):
            return list(map(lambda x: self.f2i_dict[x], feature))
        else:
            return [self.f2i_dict[feature]]
    
    def forward(self, opname, x, device=None, tiles=None, culib=None, label=None):
        B = self.get_feature(x, "B")
        M = self.get_feature(x, "M")
        N = self.get_feature(x, "N")
        K = self.get_feature(x, "K")

        dev_flops = self.get_feature(x, "SingleFLOPs") # glops

        utilization = 0.705 # from a100
 
        op_flops = 2.0 * B * (M * N * K) / (10**9)

        time = op_flops / (dev_flops*utilization)

        return time