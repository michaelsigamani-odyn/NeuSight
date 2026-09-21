from ..model import ModelBase
import pandas as pd

class MicroVEC(ModelBase):
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
        
    def get_params(self, opname, device, coef_path, bias_path):
        coef = pd.read_csv(coef_path).query(f"Device == '{device}'")["coef"].item()
        bias = pd.read_csv(bias_path).query(f"Device == '{device}'")["intercept"].item()
        return coef, bias
    
    def forward(self, opname, x, device=None, tiles=None, culib=None, label=None):
        assert(len(opname) == 1)

        B = self.get_feature(x, "B")
        H = self.get_feature(x, "H")

        memPerO = self.get_feature(x, "MemPerO")
        O = B * H
        op_mem = O * memPerO

        # retrieve coef and bias
        # print(opname)
        assert(len(opname) == 1)

        device = device.replace(' ', '_')
        if "ln" in opname[0]:
            coef = pd.read_csv("data/micro/ln.csv").query(f"Device == '{device}'")["coef"].item()
            bias = pd.read_csv("data/micro/ln.csv").query(f"Device == '{device}'")["intercept"].item()
        elif "softmax" in opname[0]:
            coef = pd.read_csv("data/micro/softmax.csv").query(f"Device == '{device}'")["coef"].item()
            bias = pd.read_csv("data/micro/softmax.csv").query(f"Device == '{device}'")["intercept"].item()
        else:
            coef = pd.read_csv("data/micro/elem.csv").query(f"Device == '{device}'")["coef"].item()
            bias = pd.read_csv("data/micro/elem.csv").query(f"Device == '{device}'")["intercept"].item()

        time = op_mem * coef + bias

        self.last_op_mem = op_mem
        self.last_coef = coef
        self.last_bias = bias

        return time