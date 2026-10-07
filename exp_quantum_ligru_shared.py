"""Run the 'quantum_ligru' experiment. All logic lives in common/ and models/ —
this file only picks the config."""
from experiments.configs import CONFIGS
from experiments.run_experiment import run

if __name__ == "__main__":
    run(CONFIGS["quantum_ligru_shared"])
