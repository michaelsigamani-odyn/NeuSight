
import torch.nn as nn
from ..model import ModelBase
import torch
import pickle
from ..meta import MetaTable

class MLPBlock(nn.Module):
    def __init__(self, layers, layer_size, dropout_rate, act):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layers = nn.ModuleList()

        for idx in range(layers):
            self.layers.append(nn.Linear(layer_size, layer_size))
            self.layers.append(act)
            self.layers.append(self.dropout)

    def forward(self, x, kname=None, culib=None):
        for layer in self.layers:
            x = layer(x) 

        return x

class HabitatWaveMM(ModelBase):
    def __init__(self, config, tag, device):
        super().__init__(config,tag=tag,device=device) # name and feature

        self.lr = self.config["lr"]
        self.train_batch = int(self.config["train_batch"])
        self.val_batch = int(self.config["val_batch"])
        self.loss = self.config["loss"]

        self.f2i_dict = dict(zip(self.features, range(len(self.features))))

        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.prelu = nn.PReLU()
        self.gelu = nn.GELU()

        self.softplusp1 = nn.Softplus(beta=.1)
        self.softplus = nn.Softplus(beta=1)
        self.softplus10 = nn.Softplus(beta=10)
        self.softplus100 = nn.Softplus(beta=100)
        self.softplus1000 = nn.Softplus(beta=1000)

        act_dict = {
            "sigmoid" : self.sigmoid,
            "leakyrelu" : self.leakyrelu,
            "relu" : self.relu,
            "tanh" : self.tanh,
            "prelu" : self.prelu,
            "gelu" : self.gelu,
        }

        self.arch = config["arch"]

        if self.arch == "MLP":
            self.layer_size = config["hidden_size"]
            self.num_layers = config["num_layers"]
            self.act = config["act"]
            self.act = act_dict[self.act]
            self.dropout_rate = config["dropout_rate"]
            num_input = len(self.features) # no batch
            self.net = MLPBlock(input_dim=num_input, output_dim=1, hidden_dim=self.layer_size, num_layers=self.num_layers, act=self.act, dropout_rate=self.dropout_rate)
        else:
            assert(0)

        self.record = False

        self.record_entries = []

        self.bus_m = []
        self.bus_s = []
        self.bias_m = []

        self.alpha_m = []
        self.alpha_s = []
        self.beta_m = []
        self.beta_s = []
        self.gamma_m = []
        self.gamma_s = []

        self.tpw_ape_list = []

        self.eval_ = False

    def set_meta_table(self, table_name):
        print("here---------")
        self.meta_table = MetaTable(table_name, self.features)

    def get_feature(self, x, features):
        features_idx = self.f2i(features)
        x = x[:,features_idx]
        x = x.view(-1, len(features_idx))
        x = x
        return x

    def stats(self):
        
        entry = {
        }

        # clear
        return entry

    def f2i(self, feature):
        if isinstance(feature, list):
            return list(map(lambda x: self.f2i_dict[x], feature))
        else:
            return [self.f2i_dict[feature]]
    
    def set_record(self, set):
        self.record = set
    
    def dump(self, dataset_name, file_name):
        assert(self.record == True)
        file_name.parent.mkdir(parents=True, exist_ok=True)
        print(f"dumping pkl : {file_name}")

        with open(file_name, 'wb') as handle:
            pickle.dump(self.record_entries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def compute_loss(self, criterion, pred, label):
        self.last_label = label
        return criterion(pred / self.last_num_wave, label / self.last_num_wave)

    def ceil_multiple(self, x, y):
        # round x to the cloest multiple of y
        return torch.ceil(x/y)*y

    def compute_num_block(self, x, tiles):

        tile_m_dim = tiles[:,0].unsqueeze(-1)
        tile_k_dim = tiles[:,2].unsqueeze(-1)

        nb_m = torch.ceil(self.get_feature(x, "M") / tile_m_dim)
        nb_k = torch.ceil(self.get_feature(x, "K") / tile_k_dim)

        b = self.get_feature(x, "B")

        num_block = b * nb_m * nb_k
        return num_block

    def get_tiles(self, x, tiles, culib, opname):
        if tiles is None:
            tiles = self.meta_table.get_tile(x, culib, opname)
            tiles = tiles.to(self.device)
        tiles = tiles.view(-1,2)

        tm = tiles[:,1].unsqueeze(-1) # tile y
        tn = self.get_feature(x, "N")
        tk = tiles[:,0].unsqueeze(-1) # tile x

        tiles = torch.cat([tm, tn, tk], dim=1) # batch, 3
        
        self.last_tiles = tiles

        return tiles
    
    def compute_wave_time(self, opname, x, tiles):
        # input dims
        num_sm = self.get_feature(x, "Num_Sm") # batch, num_features

        x = torch.concat([x[:,0:1], tiles, x[:, 4:]], dim=1) # batch, num_features

        # wave quantization
        num_block = self.compute_num_block(x, tiles)

        # wave quantization
        num_wave = torch.ceil(num_block / num_sm)

        # detach for training
        num_wave_detached = num_wave.detach() # use predicted ones
        mean = [250.87275466185835, 107.787917, 254.35666354734897, 79.161162,
                729.6136648939812, 21.96855955994796, 
                64.25512975600849, 11272.378027069592]
        mean = torch.Tensor(mean).to(self.device)
        std = [268.6129960372509, 33.396921, 270.01908351630647, 29.839103,
               477.43142292836427, 11.78054605739545, 
               25.806090981495384, 4825.90879007596]
        std = torch.Tensor(std).to(self.device)
        x = (x-mean) / std

        time_per_wave = self.net(x)

        # stats
        self.last_x = x
        self.last_num_wave = num_wave_detached
        self.last_time_per_wave = time_per_wave
        self.last_num_block = num_block

        return num_wave, time_per_wave

    def forward(self, opname, x, tiles=None, culib=None, label=None, device=None, ):
        tiles = self.get_tiles(x=x, tiles=tiles, culib=culib,opname=opname)

        # compute wave nums
        num_wave, time_per_wave = self.compute_wave_time(opname=opname, x=x, tiles=tiles)

        # prediction
        time = num_wave * time_per_wave

        # stats
        self.last_pred = time

        # record for test pickle
        if self.record:
            entry = {
                "f2i" : self.f2i_dict,
                "input" : x.cpu().squeeze(),
                "pred" : self.last_pred.item(),
                "Latency" : label.item(),
                "opname" : opname[0],
            }
            self.record_entries.append(entry)

        return time
    