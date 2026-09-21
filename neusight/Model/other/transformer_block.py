import torch
import torch.nn as nn
import math
from torch import Tensor

class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, act, dropout_rate):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        assert(num_layers >= 2)

        self.layers = nn.ModuleList()

        self.act = act

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(self.act)
        self.layers.append(self.dropout)

        for idx in range(num_layers-2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(self.act)
            self.layers.append(self.dropout)

        self.layers.append(nn.Linear(hidden_dim, output_dim)) # no activation on output

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class Embedding(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout_rate):
        super(Embedding, self).__init__()

        assert(input_size == output_size)

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.act = torch.nn.ReLU()

        self.layers = torch.nn.ModuleList()
        for i in range(output_size):
            self.layers.append(MLPBlock(1, hidden_size, hidden_size, num_layers, self.act, dropout_rate))

    def forward(self, x):
        embeds = []
        for idx, layer in enumerate(self.layers):
            embeds.append(layer(x[:,idx].unsqueeze(-1)).unsqueeze(1)) # batch, 1, hidden
        x = torch.concat(embeds, dim=1) # batch, output, hidden
        return x


class AttentionPooling(nn.Module):
    def __init__(self, num_output, num_votes, input_size, hidden_size, dropout_rate):
        super().__init__()

        self.act = torch.nn.ReLU()
        self.input_size = input_size

        # voting network
        self.voting_network = torch.nn.ModuleList()
        self.voting_network.append(
            nn.Linear(input_size, hidden_size)
        )
        self.voting_network.append(self.act)
        self.voting_network.append(nn.Dropout(dropout_rate))

        self.voting_network.append(
            nn.Linear(hidden_size, num_votes)
        )

        self.voting_network.append(
            nn.Softmax(dim=1) # batch, num_votes
        )

        # all networks
        self.all_networks = torch.nn.ModuleList()
        for idx in range(num_votes):
            new_network = torch.nn.ModuleList()

            new_network.append(
                nn.Linear(input_size, hidden_size)
            )
            new_network.append(self.act)
            new_network.append(nn.Dropout(dropout_rate))

            new_network.append(
                nn.Linear(hidden_size, hidden_size)
            )
            new_network.append(self.act)
            new_network.append(nn.Dropout(dropout_rate))

            new_network.append(
                nn.Linear(hidden_size, num_output) # batch, num_output
            )

            self.all_networks.append(new_network)

    def forward(self, x):
        # outputs
        # x = x.reshape(-1, self.input_size)

        all_outputs = []
        for network in self.all_networks:
            y = x
            for layer in network:
                y = layer(y)
            all_outputs.append(y.unsqueeze(-1)) # batch, num_output, 1
        all_outputs = torch.cat(all_outputs, dim=-1) # batch, num_output, num_votes

        # votes
        for layer in self.voting_network:
            x = layer(x)
        all_votes = x # batch, num_votes
        all_votes = all_votes.unsqueeze(1) # batch, 1, num_votes

        # weighted sum
        outputs = torch.sum(all_outputs * all_votes, dim=-1) # batch, num_output

        return outputs


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term) # seq, batch, embedding
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # print(pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config, num_inputs, num_output):
        super().__init__()

        hidden_size=config["hidden_size"]
        ffd = config["ffd"]
        num_layers = config["num_layers"]
        num_emb_layers = config["num_emb_layers"]
        num_heads = config["num_heads"]
        num_votes = config["num_votes"]
        self.dropout_rate = config["dropout_rate"]

        # token_size = config["token_size"]
        token_size = num_inputs

        self.layers = nn.ModuleList()

        self.layers.append(
            Embedding(num_inputs, token_size, hidden_size, num_emb_layers, self.dropout_rate)
        )

        self.layers.append(
            PositionalEncoding(hidden_size)
        )

        self.layers.append(nn.Dropout(self.dropout_rate))

        self.layers.append (
            torch.nn.TransformerEncoder (
                encoder_layer=torch.nn.TransformerEncoderLayer (
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=ffd,
                    activation='relu',
                    batch_first=True,
                    dropout=self.dropout_rate
                ),
                num_layers=num_layers
            )
        )

        self.layers.append (
            torch.nn.Flatten()
        )

        self.layers.append (
            AttentionPooling(num_output, num_votes, token_size * hidden_size, ffd, self.dropout_rate)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    