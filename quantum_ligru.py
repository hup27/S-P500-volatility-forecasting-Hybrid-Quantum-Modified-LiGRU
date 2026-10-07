"""Quantum-hybrid LiGRU. Identical code was duplicated in two notebook cells."""
import pennylane as qml
import torch
import torch.nn.functional as F
from torch import nn

NUM_QUBITS = 2
NUM_QUANTUM_LAYERS = 1

dev = qml.device("default.qubit", wires=NUM_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS))
    qml.BasicEntanglingLayers(weights, wires=range(NUM_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (NUM_QUANTUM_LAYERS, NUM_QUBITS)}
        init_method = torch.nn.init.uniform_
        self.layer = qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)
        with torch.no_grad():
            self.layer.weights.data = self.layer.weights.data * 2 * torch.pi

    def forward(self, x):
        return self.layer(x)


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.update_whh, self.update_b = self.create_gate_parameters()
        self.candidate_whh, self.candidate_b = self.create_gate_parameters()

        self.x_proj_update = nn.Linear(self.input_dim, self.hidden_dim)
        self.x_proj_candidate = nn.Linear(self.input_dim, self.hidden_dim)

        # Bridge classical hidden_dim <-> num_qubits
        self.q_proj_update = nn.Linear(self.hidden_dim, NUM_QUBITS)
        self.q_proj_candidate = nn.Linear(self.hidden_dim, NUM_QUBITS)

        self.q_out_update = nn.Linear(NUM_QUBITS, self.hidden_dim)
        self.q_out_candidate = nn.Linear(NUM_QUBITS, self.hidden_dim)

        self.update_q = QuantumLayer()
        self.candidate_q = QuantumLayer()

    def create_gate_parameters(self):
        hidden_weights = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(hidden_weights)
        bias = nn.Parameter(torch.zeros(self.hidden_dim))
        return hidden_weights, bias

    def forward(self, x, h):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x_update_emb = self.x_proj_update(x)
        x_candidate_emb = self.x_proj_candidate(x)

        x_update_emb = torch.tanh(x_update_emb) * torch.pi
        x_candidate_emb = torch.tanh(x_candidate_emb) * torch.pi

        x_up_flat = x_update_emb.view(-1, self.hidden_dim)
        x_cand_flat = x_candidate_emb.view(-1, self.hidden_dim)

        x_up_q_input = self.q_proj_update(x_up_flat)
        x_cand_q_input = self.q_proj_candidate(x_cand_flat)

        q_up_flat = self.update_q(x_up_q_input)
        q_cand_flat = self.candidate_q(x_cand_q_input)

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

        self.layers = nn.ModuleList()
        self.layers.append(GRUCell(input_dim, hidden_dim))
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
