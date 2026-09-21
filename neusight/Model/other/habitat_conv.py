import torch.nn as nn
from ..model import ModelBase
from .transformer_block import TransformerBlock
import torch

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

class HABITATConv(ModelBase):
    def __init__(self, config, tag, device):
        super().__init__(config,tag=tag,device=device) # name and feature

        self.lr = self.config["lr"]
        self.train_batch = int(self.config["train_batch"])
        self.val_batch = int(self.config["val_batch"])
        self.loss = self.config["loss"]

        self.arch = self.config["arch"]

        if self.arch == "MLP":
            self.layer_size = config["hidden_size"]
            self.num_layers = config["num_layers"]
            self.dropout_rate = config["dropout_rate"]

            self.sigmoid = nn.Sigmoid()
            self.leakyrelu = nn.LeakyReLU()
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()
            self.selu = nn.SELU()

            act_dict = {
                "sigmoid" : self.sigmoid,
                "leakyrelu" : self.leakyrelu,
                "relu" : self.relu,
                "tanh" : self.tanh,
                "selu" : self.selu,
            }
            self.act = config["act"]
            self.act = act_dict[self.act]

            self.fc1 = nn.Linear(len(self.features), self.layer_size)
            self.mlp = MLPBlock(self.num_layers-2, self.layer_size, self.dropout_rate, self.act)
            self.fc2 = nn.Linear(self.layer_size, 1)

        elif self.arch == "TRANS":
            num_input = len(self.features)
            self.net = TransformerBlock(config, num_inputs=num_input, num_output=1)
        else:
            assert(0)

    def forward(self, opname, x, device=None, kname=None, culib=None):

        # "B", "M", "N", "K", "Mem_Bw", "Dev_Mem", "Num_Sm", "SingleFLOPs"

        mean = [250.87275466185835, 276.3080478396823, 254.35666354734897, 256.34518293657135, 
                729.6136648939812, 21.96855955994796, 
                64.25512975600849, 11272.378027069592]
        mean = torch.Tensor(mean).to(self.device)
        std = [268.6129960372509, 272.99853966187067, 270.01908351630647, 274.0288296265487, 
               477.43142292836427, 11.78054605739545, 
               25.806090981495384, 4825.90879007596]
        std = torch.Tensor(std).to(self.device)
        x = (x-mean) / std

        if self.arch == "MLP":
            x = self.fc1(x)
            x = self.relu(x)
            x = self.mlp(x)
            x = self.fc2(x)
            return x
        elif self.arch == "TRANS":
            return self.net(x)
        else:
            assert(0)
            
