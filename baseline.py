"""
Parameter-matched classical baselines for ablation against quantum LiGRU circuits.
"""
import torch
import torch.nn as nn


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