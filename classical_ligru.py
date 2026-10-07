"""Classical benchmark LiGRU (no quantum layer) — mirrors quantum_ligru.py's structure."""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Parameter, Linear, ModuleList, Dropout, init


class ClassicalGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(ClassicalGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.update_whh, self.update_b = self.create_gate_parameters()
        self.candidate_whh, self.candidate_b = self.create_gate_parameters()

        self.x_proj_update = nn.Linear(self.input_dim, self.hidden_dim)
        self.x_proj_candidate = nn.Linear(self.input_dim, self.hidden_dim)

        # Classical replacement for the quantum + bridge layers
        self.c_proj_update = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.c_proj_candidate = nn.Linear(self.hidden_dim, self.hidden_dim)

    def create_gate_parameters(self):
        hidden_weights = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(hidden_weights)
        bias = nn.Parameter(torch.zeros(self.hidden_dim))
        return hidden_weights, bias

    def forward(self, x, h):
        seq_len = x.size(1)

        x_update_emb = torch.tanh(self.x_proj_update(x))
        x_candidate_emb = torch.tanh(self.x_proj_candidate(x))

        c_up_seq = self.c_proj_update(x_update_emb)
        c_cand_seq = self.c_proj_candidate(x_candidate_emb)

        output_hiddens = []
        for i in range(seq_len):
            update_gate = F.sigmoid((h @ self.update_whh) + c_up_seq[:, i, :] + self.update_b)
            candidate_hidden = F.tanh((h @ self.candidate_whh) + c_cand_seq[:, i, :] + self.candidate_b)

            h = (update_gate * candidate_hidden) + ((1 - update_gate) * h)
            output_hiddens.append(h.unsqueeze(1))

        return torch.concat(output_hiddens, dim=1)


class ClassicalMultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(ClassicalMultiLayerGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(ClassicalGRUCell(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(ClassicalGRUCell(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x, h):
        output_hidden = self.layers[0](x, h[0])
        new_hidden = [output_hidden[:, -1, :]]

        for i in range(1, self.num_layers):
            output_hidden = self.layers[i](self.dropout(output_hidden), h[i])
            new_hidden.append(output_hidden[:, -1, :])

        linear_out = self.linear(self.dropout(output_hidden))
        return linear_out, torch.stack(new_hidden, dim=0)
