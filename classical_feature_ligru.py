"""Classical-feature LiGRU — identical to quantum_ligru.py except QuantumLayer
is replaced with a parameter-matched classical block (MLP or trig features).
This isolates whether the *quantum circuit specifically* matters, or whether
any nonlinear parametrized map of the same size gives the same benefit."""
import torch
import torch.nn.functional as F
from torch import nn

from models.baseline import ClassicalCircuitReplacement

NUM_QUBITS = 2          # keep identical to quantum_ligru.py for a fair match
NUM_QUANTUM_LAYERS = 1
TARGET_PARAMS = NUM_QUANTUM_LAYERS * NUM_QUBITS * 3   # matches StronglyEntanglingLayers param count
MODE = "mlp"             # switch to "trig" for the trig-feature ablation variant


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.update_whh, self.update_b = self.create_gate_parameters()
        self.candidate_whh, self.candidate_b = self.create_gate_parameters()

        self.x_proj_update = nn.Linear(self.input_dim, self.hidden_dim)
        self.x_proj_candidate = nn.Linear(self.input_dim, self.hidden_dim)

        self.q_proj_update = nn.Linear(self.hidden_dim, NUM_QUBITS)
        self.q_proj_candidate = nn.Linear(self.hidden_dim, NUM_QUBITS)

        self.q_out_update = nn.Linear(NUM_QUBITS, self.hidden_dim)
        self.q_out_candidate = nn.Linear(NUM_QUBITS, self.hidden_dim)

        # only these two lines differ from quantum_ligru.py's GRUCell
        self.update_q = ClassicalCircuitReplacement(NUM_QUBITS, TARGET_PARAMS, mode=MODE)
        self.candidate_q = ClassicalCircuitReplacement(NUM_QUBITS, TARGET_PARAMS, mode=MODE)

    def create_gate_parameters(self):
        hidden_weights = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(hidden_weights)
        bias = nn.Parameter(torch.zeros(self.hidden_dim))
        return hidden_weights, bias

    def forward(self, x, h):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x_update_emb = torch.tanh(self.x_proj_update(x)) * torch.pi
        x_candidate_emb = torch.tanh(self.x_proj_candidate(x)) * torch.pi

        x_up_flat = x_update_emb.view(-1, self.hidden_dim)
        x_cand_flat = x_candidate_emb.view(-1, self.hidden_dim)

        q_up_flat = self.update_q(self.q_proj_update(x_up_flat))
        q_cand_flat = self.candidate_q(self.q_proj_candidate(x_cand_flat))

        q_up_flat = self.q_out_update(q_up_flat)
        q_cand_flat = self.q_out_candidate(q_cand_flat)

        q_up_seq = q_up_flat.view(batch_size, seq_len, self.hidden_dim)
        q_cand_seq = q_cand_flat.view(batch_size, seq_len, self.hidden_dim)

        output_hiddens = []
        for i in range(seq_len):
            update_gate = F.sigmoid((h @ self.update_whh) + q_up_seq[:, i, :] + self.update_b)
            candidate_hidden = F.tanh((h @ self.candidate_whh) + q_cand_seq[:, i, :] + self.candidate_b)
            h = (update_gate * candidate_hidden) + ((1 - update_gate) * h)
            output_hiddens.append(h.unsqueeze(1))

        return torch.concat(output_hiddens, dim=1)


class MultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(MultiLayerGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([GRUCell(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.layers.append(GRUCell(hidden_dim, hidden_dim))

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