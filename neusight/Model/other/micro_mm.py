from ..model import ModelBase
import torch
import pandas as pd

class MicroMM(ModelBase):
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

        op_flops = 2.0 * B * (M * N * K) # flops (not Gflops)

        # retrieve coef and bias
        device = device.replace(' ', '_')
        if opname[0] == "bmm":
            coef = pd.read_csv("data/micro/bmm.csv").query(f"Device == '{device}'")["coef"].item()
            bias = pd.read_csv("data/micro/bmm.csv").query(f"Device == '{device}'")["intercept"].item()
        elif opname[0] == "linear":
            coef = pd.read_csv("data/micro/linear.csv").query(f"Device == '{device}'")["coef"].item()
            bias = pd.read_csv("data/micro/linear.csv").query(f"Device == '{device}'")["intercept"].item()
        else:
            raise NotImplementedError

        time = coef * op_flops + bias

        return time