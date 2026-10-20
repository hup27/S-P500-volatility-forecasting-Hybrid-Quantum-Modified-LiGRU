"""All model architectures for the LiGRU volatility experiments.

Consolidated from models/{baseline, classical_ligru, classical_feature_ligru,
quantum_ligru, quantum_ligru_frozen, quantum_ligru_shared,
quantum_ligru_without_entanglement, quantum_ligru_linear_layer}.py.

Every variant used to define its own ``GRUCell`` / ``MultiLayerGRU`` /
``QuantumLayer`` / ``circuit``, so they are prefixed here to avoid clashes.
The code of each class is unchanged -- only the names differ -- and the order
in which parameters are created is untouched, so seeded initialisation should give
the same weights as before.

    old file                                 ->  MultiLayerGRU class in this file
    ---------------------------------------      --------------------------------
    classical_ligru.py                           ClassicalMultiLayerGRU
    classical_feature_ligru.py                   ClassicalFeatureMultiLayerGRU
    quantum_ligru.py                             QuantumMultiLayerGRU
    quantum_ligru_frozen.py                      FrozenQuantumMultiLayerGRU
    quantum_ligru_shared.py                      SharedQuantumMultiLayerGRU
    quantum_ligru_without_entanglement.py        NoEntanglementQuantumMultiLayerGRU
    quantum_ligru_linear_layer.py                GateMapMultiLayerGRU
"""
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_QUBITS = 2
NUM_QUANTUM_LAYERS = 1

# One simulator shared by every circuit below (each file used to create its own).
dev = qml.device("default.qubit", wires=NUM_QUBITS)


# ===========================================================================
# Parameter-matched classical baselines (was models/baseline.py)
# ===========================================================================
# Parameter-matched classical baselines for ablation against quantum LiGRU circuits.
# ===========================================================================


class TrigFeatureLayer(nn.Module):
    """Fixed (or optionally learnable) random trig feature map —
    classical analog of a quantum encoding/entangling layer."""
    def __init__(self, in_dim, n_features, learnable=False):
        super().__init__()
        W = torch.randn(in_dim, n_features) * 2 * torch.pi
        b = torch.rand(n_features) * 2 * torch.pi
        if learnable:
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer('W', W)
            self.register_buffer('b', b)

    def forward(self, x):
        proj = x @ self.W + self.b
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ParamMatchedMLP(nn.Module):
    """MLP whose hidden width is auto-tuned to match a target parameter count,
    so it's a fair capacity-matched baseline against a quantum circuit layer."""
    def __init__(self, in_dim, out_dim, target_params, n_layers=2):
        super().__init__()
        hidden = self._solve_hidden(in_dim, out_dim, target_params, n_layers)
        layers, d = [], in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self.hidden = hidden  # log this in your results table

    def _count(self, in_dim, out_dim, h, n_layers):
        d, total = in_dim, 0
        for _ in range(n_layers - 1):
            total += d * h + h
            d = h
        return total + d * out_dim + out_dim

    def _solve_hidden(self, in_dim, out_dim, target_params, n_layers):
        best_h, best_diff = 1, float('inf')
        for h in range(1, 1024):
            diff = abs(self._count(in_dim, out_dim, h, n_layers) - target_params)
            if diff < best_diff:
                best_diff, best_h = diff, h
        return best_h

    def forward(self, x):
        return self.net(x)

class ClassicalCircuitReplacement(nn.Module):
    """Drop-in replacement for QuantumLayer — same in/out shape (NUM_QUBITS -> NUM_QUBITS),
    parameter-matched to the quantum circuit it's replacing."""
    def __init__(self, num_qubits, target_params, mode="mlp"):
        super().__init__()
        self.mode = mode
        if mode == "mlp":
            self.block = ParamMatchedMLP(num_qubits, num_qubits, target_params, n_layers=2)
        elif mode == "trig":
            # n_features chosen so 2*n_features (sin+cos) roughly matches target_params via the linear readout
            n_features = max(1, target_params // (2 * num_qubits))
            self.trig = TrigFeatureLayer(num_qubits, n_features, learnable=False)
            self.readout = nn.Linear(2 * n_features, num_qubits)
        else:
            raise ValueError(f"unknown mode {mode}")

    def forward(self, x):
        if self.mode == "mlp":
            return self.block(x)
        feats = self.trig(x)
        return self.readout(feats)


# ===========================================================================
# Classical LiGRU (was models/classical_ligru.py)
# ===========================================================================
# Classical benchmark LiGRU (no quantum layer) -- mirrors quantum_ligru.py's structure.
# ===========================================================================


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


# ===========================================================================
# Classical-feature LiGRU (was models/classical_feature_ligru.py)
# ===========================================================================
# Identical to the quantum LiGRU except QuantumLayer is replaced with a
# parameter-matched classical block (MLP or trig features). This isolates whether
# the *quantum circuit specifically* matters, or whether any nonlinear
# parametrized map of the same size gives the same benefit.
# ===========================================================================


CLASSICAL_FEATURE_TARGET_PARAMS = NUM_QUANTUM_LAYERS * NUM_QUBITS  # matches StronglyEntanglingLayers param count
CLASSICAL_FEATURE_MODE = "mlp"  # switch to "trig" for the trig-feature ablation variant


class ClassicalFeatureGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(ClassicalFeatureGRUCell, self).__init__()
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

        # only these two lines differ from quantum_ligru.py's ClassicalFeatureGRUCell
        self.update_q = ClassicalCircuitReplacement(NUM_QUBITS, CLASSICAL_FEATURE_TARGET_PARAMS, mode=CLASSICAL_FEATURE_MODE)
        self.candidate_q = ClassicalCircuitReplacement(NUM_QUBITS, CLASSICAL_FEATURE_TARGET_PARAMS, mode=CLASSICAL_FEATURE_MODE)

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


class ClassicalFeatureMultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(ClassicalFeatureMultiLayerGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([ClassicalFeatureGRUCell(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.layers.append(ClassicalFeatureGRUCell(hidden_dim, hidden_dim))

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


# ===========================================================================
# Quantum LiGRU (was models/quantum_ligru.py)
# ===========================================================================
# Quantum-hybrid LiGRU: each gate's nonlinearity is a 2-qubit circuit
# (AngleEmbedding + BasicEntanglerLayers).
# ===========================================================================


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(NUM_QUBITS))
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


class QuantumGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(QuantumGRUCell, self).__init__()
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


class QuantumMultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(QuantumMultiLayerGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(QuantumGRUCell(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(QuantumGRUCell(hidden_dim, hidden_dim))

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


# ===========================================================================
# Quantum LiGRU, frozen circuit (was models/quantum_ligru_frozen.py)
# ===========================================================================
# Ablation: the quantum layer keeps its random init but is never trained.
# ===========================================================================


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit_basic_entangler(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(NUM_QUBITS), rotation=qml.RX)
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


class FrozenQuantumLayer(nn.Module):
    def __init__(self, trainable: bool = True):
        super().__init__()
        weight_shapes = {"weights": (NUM_QUANTUM_LAYERS, NUM_QUBITS)}
        init_method = torch.nn.init.uniform_
        self.layer = qml.qnn.TorchLayer(circuit_basic_entangler, weight_shapes, init_method=init_method)

        with torch.no_grad():
            self.layer.weights.data = self.layer.weights.data * 2 * torch.pi

        if not trainable:
            # Freeze: keep the random init, but never update it during training
            self.layer.weights.requires_grad_(False)

    def forward(self, x):
        return self.layer(x)


class FrozenQuantumGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(FrozenQuantumGRUCell, self).__init__()
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

        self.update_q = FrozenQuantumLayer(trainable=False)  # ablation: freeze the quantum layer
        self.candidate_q = FrozenQuantumLayer(trainable=False)  # ablation: freeze the quantum layer

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


class FrozenQuantumMultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(FrozenQuantumMultiLayerGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(FrozenQuantumGRUCell(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(FrozenQuantumGRUCell(hidden_dim, hidden_dim))

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


# ===========================================================================
# Quantum LiGRU, shared circuit (was models/quantum_ligru_shared.py)
# ===========================================================================
# Quantum-hybrid LiGRU with explicit separate (or optionally shared) gate/candidate circuits.
# ===========================================================================


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit_shared(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(NUM_QUBITS), rotation=qml.RX)
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


class SharedQuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (NUM_QUANTUM_LAYERS, NUM_QUBITS)}
        init_method = torch.nn.init.uniform_
        self.layer = qml.qnn.TorchLayer(circuit_shared, weight_shapes, init_method=init_method)
        with torch.no_grad():
            self.layer.weights.data = self.layer.weights.data * 2 * torch.pi

    def forward(self, x):
        return self.layer(x)


class SharedQuantumGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, share_circuit=False) -> None:
        super(SharedQuantumGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.share_circuit = share_circuit

        self.update_whh, self.update_b = self.create_gate_parameters()
        self.candidate_whh, self.candidate_b = self.create_gate_parameters()

        self.x_proj_update = nn.Linear(self.input_dim, self.hidden_dim)
        self.x_proj_candidate = nn.Linear(self.input_dim, self.hidden_dim)

        # Bridge classical hidden_dim <-> num_qubits
        self.q_proj_update = nn.Linear(self.hidden_dim, NUM_QUBITS)
        self.q_proj_candidate = nn.Linear(self.hidden_dim, NUM_QUBITS)

        self.q_out_update = nn.Linear(NUM_QUBITS, self.hidden_dim)
        self.q_out_candidate = nn.Linear(NUM_QUBITS, self.hidden_dim)

        # Explicit separate circuits by default. Set share_circuit=True to
        # force gate and candidate paths through the SAME SharedQuantumLayer
        # (same weight tensor) as an ablation control.
        self.update_q = SharedQuantumLayer()
        if self.share_circuit:
            self.candidate_q = self.update_q
        else:
            self.candidate_q = SharedQuantumLayer()

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


class SharedQuantumMultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, share_circuit=False):
        super(SharedQuantumMultiLayerGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(SharedQuantumGRUCell(input_dim, hidden_dim, share_circuit=share_circuit))
        for _ in range(num_layers - 1):
            self.layers.append(SharedQuantumGRUCell(hidden_dim, hidden_dim, share_circuit=share_circuit))

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


# ===========================================================================
# Quantum LiGRU, no entanglement (was models/quantum_ligru_without_entanglement.py)
# ===========================================================================
# Ablation: the CNOT ring is removed, leaving only single-qubit rotations.
# ===========================================================================


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit_no_entanglement(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS))

    # weights shape for BasicEntanglerLayers: (num_layers, NUM_QUBITS)
    num_layers = weights.shape[0]
    for layer in range(num_layers):
        for wire in range(NUM_QUBITS):
            qml.RX(weights[layer, wire], wires=wire)
        # no CNOT ring here — entanglement removed

    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


class NoEntanglementQuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (NUM_QUANTUM_LAYERS, NUM_QUBITS, 3)}
        init_method = torch.nn.init.uniform_
        self.layer = qml.qnn.TorchLayer(circuit_no_entanglement, weight_shapes, init_method=init_method)
        with torch.no_grad():
            self.layer.weights.data = self.layer.weights.data * 2 * torch.pi

    def forward(self, x):
        return self.layer(x)


class NoEntanglementQuantumGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(NoEntanglementQuantumGRUCell, self).__init__()
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

        self.update_q = NoEntanglementQuantumLayer()
        self.candidate_q = NoEntanglementQuantumLayer()

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


class NoEntanglementQuantumMultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(NoEntanglementQuantumMultiLayerGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(NoEntanglementQuantumGRUCell(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(NoEntanglementQuantumGRUCell(hidden_dim, hidden_dim))

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


# ===========================================================================
# Quantum vs parameter-matched classical gate map (was models/quantum_ligru_linear_layer.py)
# ===========================================================================
# Quantum-hybrid LiGRU with a parameter-matched classical ablation baseline.
#
# Motivation
# ----------
# In the original code, `GateMapQuantumLayer` maps a 2-d vector of rotation angles to a
# 2-d vector of PauliZ expectation values using 6 trainable parameters (the
# StronglyEntanglingLayers weights, shape (1, 2, 3)). Swapping in *any* classical
# nonlinearity to compare against it is only a fair test if that classical
# nonlinearity has the same input/output dimensionality and the same parameter
# budget -- otherwise a performance gap could just be a capacity gap in
# disguise.
#
# `GateMapClassicalLayer` below is that matched baseline:
#   - input dim  = NUM_QUBITS (2), same as GateMapQuantumLayer
#   - output dim = NUM_QUBITS (2), same as GateMapQuantumLayer
#   - trainable parameters = 6 (a single Linear(2, 2): 2*2 weights + 2 bias),
#     identical to GateMapQuantumLayer's 6 rotation-angle weights
#   - tanh nonlinearity, so outputs land in (-1, 1) like PauliZ expectations do,
#     keeping the statistics fed into q_out_update / q_out_candidate comparable
#
# `GateMapGRUCell` / `GateMapMultiLayerGRU` take a `use_quantum` flag so you can build both
# variants from the same code path, and `compare_gate_maps()` tabulates
# parameter counts and per-call operation counts for both.
# ===========================================================================


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit_gate_map(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(NUM_QUBITS), rotation=qml.RX)
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def quantum_gate_count():
    """Number of quantum gates applied per input row (embedding + entangling layer).

    Uses qml.specs, the public API for circuit resource info. The exact key
    used to pull the gate count out of the returned dict has moved around a
    little across PennyLane versions, so this tries a couple of the common
    shapes and falls back to a manual analytic count (NUM_QUBITS embedding
    rotations + NUM_QUBITS*3 rotation angles + NUM_QUBITS ring-CNOTs per
    entangling layer) if neither matches your installed version.
    """
    dummy_input = torch.zeros(NUM_QUBITS)
    dummy_weights = torch.zeros(NUM_QUANTUM_LAYERS, NUM_QUBITS)
    try:
        specs = qml.specs(circuit_gate_map)(dummy_input, dummy_weights)
        resources = specs["resources"]
        if hasattr(resources, "num_gates"):
            return resources.num_gates
        if isinstance(resources, dict) and "num_gates" in resources:
            return resources["num_gates"]
    except Exception:
        pass
    # Analytic fallback: AngleEmbedding = NUM_QUBITS rotations; each
    # StronglyEntanglingLayers layer = NUM_QUBITS Rot gates (3 angles each,
    # counted as one gate) + NUM_QUBITS CNOTs in a ring.
    return NUM_QUBITS + NUM_QUANTUM_LAYERS * (NUM_QUBITS + NUM_QUBITS)


class GateMapQuantumLayer(nn.Module):
    """Quantum feature map: R^NUM_QUBITS -> R^NUM_QUBITS, 6 trainable parameters."""

    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (NUM_QUANTUM_LAYERS, NUM_QUBITS)}
        init_method = torch.nn.init.uniform_
        self.layer = qml.qnn.TorchLayer(circuit_gate_map, weight_shapes, init_method=init_method)
        with torch.no_grad():
            self.layer.weights.data = self.layer.weights.data * 2 * torch.pi

    def forward(self, x):
        return self.layer(x)


class GateMapClassicalLayer(nn.Module):
    """Classical nonlinear feature map: R^NUM_QUBITS -> R^NUM_QUBITS.

    Parameter-matched to GateMapQuantumLayer (see module docstring): a single
    Linear(NUM_QUBITS, NUM_QUBITS) followed by tanh has NUM_QUBITS**2 +
    NUM_QUBITS = 6 trainable parameters when NUM_QUBITS == 2, identical to
    GateMapQuantumLayer's 6 rotation-angle weights, with the same input/output
    dimensionality and a comparable bounded output range.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(NUM_QUBITS, NUM_QUBITS)
        # Match GateMapQuantumLayer's init scale/spirit: a symmetric random init
        # rather than PyTorch's default Kaiming init, so neither map starts
        # with a built-in advantage from initialization alone.
        nn.init.uniform_(self.linear.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.linear.bias, a=-1.0, b=1.0)

    def forward(self, x):
        return torch.tanh(self.linear(x))

    def flop_count(self, batch_size=1):
        """Multiply-add operations per forward call (per input row, times batch)."""
        in_f, out_f = self.linear.in_features, self.linear.out_features
        return batch_size * (in_f * out_f + out_f)  # matmul + bias add


def make_gate_map(use_quantum: bool):
    return GateMapQuantumLayer() if use_quantum else GateMapClassicalLayer()


class GateMapGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_quantum=True) -> None:
        super(GateMapGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_quantum = use_quantum

        self.update_whh, self.update_b = self.create_gate_parameters()
        self.candidate_whh, self.candidate_b = self.create_gate_parameters()

        self.x_proj_update = nn.Linear(self.input_dim, self.hidden_dim)
        self.x_proj_candidate = nn.Linear(self.input_dim, self.hidden_dim)

        # Bridge classical hidden_dim <-> num_qubits
        self.q_proj_update = nn.Linear(self.hidden_dim, NUM_QUBITS)
        self.q_proj_candidate = nn.Linear(self.hidden_dim, NUM_QUBITS)

        self.q_out_update = nn.Linear(NUM_QUBITS, self.hidden_dim)
        self.q_out_candidate = nn.Linear(NUM_QUBITS, self.hidden_dim)

        # This is the only line that changes between the quantum model and
        # the classical ablation baseline.
        self.update_q = make_gate_map(use_quantum)
        self.candidate_q = make_gate_map(use_quantum)

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


class GateMapMultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, use_quantum=True):
        super(GateMapMultiLayerGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_quantum = use_quantum

        self.layers = nn.ModuleList()
        self.layers.append(GateMapGRUCell(input_dim, hidden_dim, use_quantum=use_quantum))
        for _ in range(num_layers - 1):
            self.layers.append(GateMapGRUCell(hidden_dim, hidden_dim, use_quantum=use_quantum))

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


# ---------------------------------------------------------------------------
# Ablation comparison table
# ---------------------------------------------------------------------------

def _count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def compare_gate_maps(hidden_dim=32, num_layers=2, dropout=0.1):
    """Build the quantum and classical variants and print a comparison table.

    Reports, side by side:
      - trainable parameters in the gate-map module alone (should be equal:
        this is the whole point of the ablation)
      - trainable parameters in the full GateMapMultiLayerGRU (should also be equal,
        since every other layer is architecturally identical between variants)
      - input/output dimensionality of the gate map (equal by construction)
      - a per-input-row operation count for each gate map

    The operation counts are reported in different units (quantum gates vs.
    classical multiply-adds) and are included for transparency about
    structural cost, NOT as a wall-clock or FLOP-equivalent comparison --
    don't read the two numbers in that row as "X times more/less compute".
    """
    quantum_gate = GateMapQuantumLayer()
    classical_gate = GateMapClassicalLayer()

    quantum_model = GateMapMultiLayerGRU(
        input_dim=hidden_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout, use_quantum=True,
    )
    classical_model = GateMapMultiLayerGRU(
        input_dim=hidden_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout, use_quantum=False,
    )

    rows = [
        ("Gate-map input dim", NUM_QUBITS, NUM_QUBITS),
        ("Gate-map output dim", NUM_QUBITS, NUM_QUBITS),
        ("Gate-map trainable params", _count_trainable(quantum_gate), _count_trainable(classical_gate)),
        ("Full model trainable params", _count_trainable(quantum_model), _count_trainable(classical_model)),
        ("Ops per input row (units differ)", quantum_gate_count(), classical_gate.flop_count(batch_size=1)),
    ]

    header = f"{'Metric':34s} | {'Quantum':>12s} | {'Classical':>12s}"
    print(header)
    print("-" * len(header))
    for name, q_val, c_val in rows:
        print(f"{name:34s} | {q_val:12} | {c_val:12}")

    print(
        "\nNote: the 'ops per input row' units differ by construction -- the "
        "quantum count is the number of quantum gates applied per row "
        "(AngleEmbedding + StronglyEntanglingLayers), the classical count is "
        "multiply-add operations per row for the Linear(2,2). They are shown "
        "together for transparency about structural cost, not as an "
        "apples-to-apples runtime or compute-equivalence claim."
    )

    return {"quantum_model": quantum_model, "classical_model": classical_model, "rows": rows}


if __name__ == "__main__":
    compare_gate_maps()
