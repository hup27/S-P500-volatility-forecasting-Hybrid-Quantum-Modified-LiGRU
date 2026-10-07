"""Run the classical param-matched MLP baseline ablation."""
from experiments.configs import CONFIGS
from experiments.run_experiment import run

if __name__ == "__main__":
    run(CONFIGS["quantum_ligru_baseline_mlp"])