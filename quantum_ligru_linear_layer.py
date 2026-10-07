"""Quantum-hybrid LiGRU with a parameter-matched classical ablation baseline.

Motivation
----------
In the original code, `QuantumLayer` maps a 2-d vector of rotation angles to a
2-d vector of PauliZ expectation values using 6 trainable parameters (the
StronglyEntanglingLayers weights, shape (1, 2, 3)). Swapping in *any* classical
nonlinearity to compare against it is only a fair test if that classical
nonlinearity has the same input/output dimensionality and the same parameter
budget -- otherwise a performance gap could just be a capacity gap in
disguise.

`ClassicalLayer` below is that matched baseline:
  - input dim  = NUM_QUBITS (2), same as QuantumLayer
  - output dim = NUM_QUBITS (2), same as QuantumLayer
  - trainable parameters = 6 (a single Linear(2, 2): 2*2 weights + 2 bias),
    identical to QuantumLayer's 6 rotation-angle weights
  - tanh nonlinearity, so outputs land in (-1, 1) like PauliZ expectations do,
    keeping the statistics fed into q_out_update / q_out_candidate comparable

`GRUCell` / `MultiLayerGRU` take a `use_quantum` flag so you can build both
variants from the same code path, and `compare_gate_maps()` tabulates
parameter counts and per-call operation counts for both.
"""
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
    qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
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
    dummy_weights = torch.zeros(NUM_QUANTUM_LAYERS, NUM_QUBITS, 3)
    try:
        specs = qml.specs(circuit)(dummy_input, dummy_weights)
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


class QuantumLayer(nn.Module):
    """Quantum feature map: R^NUM_QUBITS -> R^NUM_QUBITS, 6 trainable parameters."""

    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (NUM_QUANTUM_LAYERS, NUM_QUBITS, 3)}
        init_method = torch.nn.init.uniform_
        self.layer = qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)
        with torch.no_grad():
            self.layer.weights.data = self.layer.weights.data * 2 * torch.pi

    def forward(self, x):
        return self.layer(x)


class ClassicalLayer(nn.Module):
    """Classical nonlinear feature map: R^NUM_QUBITS -> R^NUM_QUBITS.

    Parameter-matched to QuantumLayer (see module docstring): a single
    Linear(NUM_QUBITS, NUM_QUBITS) followed by tanh has NUM_QUBITS**2 +
    NUM_QUBITS = 6 trainable parameters when NUM_QUBITS == 2, identical to
    QuantumLayer's 6 rotation-angle weights, with the same input/output
    dimensionality and a comparable bounded output range.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(NUM_QUBITS, NUM_QUBITS)
        # Match QuantumLayer's init scale/spirit: a symmetric random init
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
    return QuantumLayer() if use_quantum else ClassicalLayer()


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_quantum=True) -> None:
        super(GRUCell, self).__init__()
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


class MultiLayerGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, use_quantum=True):
        super(MultiLayerGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_quantum = use_quantum

        self.layers = nn.ModuleList()
        self.layers.append(GRUCell(input_dim, hidden_dim, use_quantum=use_quantum))
        for _ in range(num_layers - 1):
            self.layers.append(GRUCell(hidden_dim, hidden_dim, use_quantum=use_quantum))

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
      - trainable parameters in the full MultiLayerGRU (should also be equal,
        since every other layer is architecturally identical between variants)
      - input/output dimensionality of the gate map (equal by construction)
      - a per-input-row operation count for each gate map

    The operation counts are reported in different units (quantum gates vs.
    classical multiply-adds) and are included for transparency about
    structural cost, NOT as a wall-clock or FLOP-equivalent comparison --
    don't read the two numbers in that row as "X times more/less compute".
    """
    quantum_gate = QuantumLayer()
    classical_gate = ClassicalLayer()

    quantum_model = MultiLayerGRU(
        input_dim=hidden_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout, use_quantum=True,
    )
    classical_model = MultiLayerGRU(
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