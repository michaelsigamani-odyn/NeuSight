from ..model import ModelBase
import torch
from ..meta import MetaTable

class HABITATVEC(ModelBase):
    def __init__(self, config, tag, device):
        super().__init__(config,tag=tag,device=device) # name and feature

        self.f2i_dict = dict(zip(self.features, range(len(self.features))))

    def set_meta_table(self, table_name):
        self.meta_table = MetaTable(table_name, self.features)

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

    def forward(self, opname, x, device=None, tiles=None, culib=None):
        # tiles
        b = self.get_feature(x, "B")
        h = self.get_feature(x, "H")

        assert(tiles is None and culib is not None)
        assert(culib is not None)
        src_lat, src_mem_bw = self.meta_table.get_exact_match(x, culib=culib, opname=opname, B=b, H=h)

        if src_lat is None:

            imem = b * h
            omem = b * h
            mem = imem + omem
            mem_bw = self.get_feature(x, "Mem_Bw")
            pred = mem / (mem_bw * (2**30)) # assume mem bound
            time = [pred]
        else:
            dst_mem_bw = self.get_feature(x, "Mem_Bw")
            dst_lat = src_lat * (src_mem_bw / dst_mem_bw)
            time = [dst_lat]

        time = torch.Tensor(time)
        time = time.to(device="cuda")
        time = time.view(-1,1)

        return time