from ..model import ModelBase

class RooflineVEC(ModelBase):
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
        H = self.get_feature(x, "H")

        assert(len(opname) == 1)
        memPerO = self.get_feature(x, "MemPerO")
        O = B * H
        mem = O * memPerO / (10**9)

        mem_bw = self.get_feature(x, "Mem_Bw") # gb/s

        time = mem / mem_bw

        return time