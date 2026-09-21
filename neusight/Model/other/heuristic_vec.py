from ..model import ModelBase
import torch

mem_dict = {
    "add" : 3.,
    "mul" : 3.,
    "pow" : 3.,
    "div" : 3.,
    "addu": 2.,
    "mulu": 2.,
    "powu": 2.,
    "divu": 2.,
    "tanh": 2.,
    "ln"  : 5., # compute mean(1) + compute var(1) + compute element-wise sub/mul(2) 
    "softmax" : 2.,
    "relu" : 2.,
    "gelu" : 2.,
}

class HeuristicVEC(ModelBase):
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
        assert(len(opname) == 1)
        memPerO = [mem_dict[x] for x in opname]
        memPerO = torch.Tensor(memPerO).to(self.device)
        memPerO = memPerO.view(-1,1)

        B = self.get_feature(x, "B")
        H = self.get_feature(x, "H")

        memPerO = self.get_feature(x, "MemPerO")
        O = B * H
        op_mem = O * memPerO / (10**9)

        membw_utilization = 0.705

        mem_bw = self.get_feature(x, "Mem_Bw") # gb/s

        time = op_mem / (mem_bw * membw_utilization)

        return time