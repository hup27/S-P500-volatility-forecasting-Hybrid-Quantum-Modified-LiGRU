"""Run the 'egarch_classical_ligru' experiment. All logic lives in common/ and models/ —
this file only picks the config."""
from experiments.configs import CONFIGS
from experiments.run_experiment import run

if __name__ == "__main__":
    run(CONFIGS["egarch_classical_ligru"])
