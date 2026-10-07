"""Every experiment in the original notebook = the same pipeline with different
hyperparameters. This table is the single source of truth for those
differences (previously scattered as hard-coded values inside 4 giant cells).
"""
import torch.optim as optim

from models.classical_ligru import ClassicalMultiLayerGRU
from models.quantum_ligru import MultiLayerGRU as QuantumMultiLayerGRU
from common.egarch import mle, noise_from_path, get_conditional_volatility
from models.quantum_ligru_frozen import MultiLayerGRU as QuantumMultiLayerGRUFrozen
from models.classical_feature_ligru import MultiLayerGRU as ClassicalFeatureMultiLayerGRU
from models.quantum_ligru_shared import MultiLayerGRU as QuantumMultiLayerGRUShared
from models.quantum_ligru_linear_layer import MultiLayerGRU as QuantumMultiLayerGRULayer

CONFIGS = {
    "egarch_quantum_ligru": {
        "title_prefix": "EGARCH-quantum LiGRU",
        "out_prefix": "Quantum_EGARCH-LiGRU",
        "model_cls": QuantumMultiLayerGRU,
        "use_egarch_vol": True,
        "num_seeds": 7,
        "hidden_dim": 12,
        "num_layers": 2,
        "dropout": 0.4,
        "batch_size": 128,
        "optimizer_cls": optim.RMSprop,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "egarch_classical_ligru": {
        "title_prefix": "EGARCH-classical LiGRU",
        "out_prefix": "Classical_EGARCH-LiGRU",
        "model_cls": ClassicalMultiLayerGRU,
        "use_egarch_vol": True,
        "num_seeds": 7,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "standalone_ligru": {
        "title_prefix": "Classical LiGRU",
        "out_prefix": "Classical_LiGRU",
        "model_cls": ClassicalMultiLayerGRU,
        "use_egarch_vol": False,  # raw returns only, input_dim=1
        "num_seeds": 12,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "quantum_ligru": {
        "title_prefix": "Quantum LiGRU",
        "out_prefix": "Quantum_LiGRU",
        "model_cls": QuantumMultiLayerGRU,
        "use_egarch_vol": False,  # raw returns only, input_dim=1
        "num_seeds": 12,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "quantum_ligru_no_entanglement": {
        "title_prefix": "Quantum LiGRU (no entanglement)",
        "out_prefix": "Quantum_LiGRU_No_Entanglement",
        "model_cls": QuantumMultiLayerGRU,
        "use_egarch_vol": False,  # raw returns only, input_dim=1
        "num_seeds": 12,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.4,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "quantum_ligru_frozen_circuit": {
        "title_prefix": "Quantum LiGRU (frozen circuit)",
        "out_prefix": "Quantum_LiGRU_Frozen_Circuit",
        "model_cls": QuantumMultiLayerGRUFrozen,
        "use_egarch_vol": False,  # raw returns only, input_dim=1
        "num_seeds": 12,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.4,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "quantum_ligru_shared": {
        "title_prefix": "Quantum LiGRU (shared circuit)",
        "out_prefix": "Quantum_LiGRU_shared_circuit",
        "model_cls": QuantumMultiLayerGRUShared,
        "use_egarch_vol": False,  # raw returns only, input_dim=1
        "num_seeds": 12,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,
        "optimizer_cls": optim.Adam,
        "optimizer_kwargs": {"lr": 0.001},
    },
    "quantum_ligru_layer": {
            "title_prefix": "Quantum LiGRU Layer",
            "out_prefix": "Quantum_LiGRU_Layer",
            "model_cls": QuantumMultiLayerGRULayer,
            "use_egarch_vol": False,  
            "num_seeds": 12,
            "hidden_dim": 12,
            "num_layers": 3,
            "dropout": 0.3,
            "batch_size": 64,
            "optimizer_cls": optim.Adam,
            "optimizer_kwargs": {"lr": 0.002},
    },
}

CONFIGS["quantum_ligru_baseline_mlp"] = {
    "title_prefix": "Classical Baseline (param-matched MLP)",
    "out_prefix": "Baseline_MLP_LiGRU",
    "model_cls": ClassicalFeatureMultiLayerGRU,
    "use_egarch_vol": False,
    "num_seeds": 12,
    "hidden_dim": 4,
    "num_layers": 2,
    "dropout": 0.4,
    "batch_size": 32,
    "optimizer_cls": optim.Adam,
    "optimizer_kwargs": {"lr": 0.001},
}