import torch
from gflownet.config import Config, init_empty
from gflownet.tasks.quantum_circuit import QuantumCircuitTrainer

def main():
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.desc = "quantum_circuit_gflownet"
    config.log_dir = "./logs/quantum_circuit_run"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.num_workers = 0  # For easier debugging
    config.print_every = 1
    config.algo.num_from_policy = 64
    config.validate_every = 100
    config.num_final_gen_steps = 5
    config.num_training_steps = 10000
    config.pickle_mp_messages = True
    config.overwrite_existing_exp = True
    
    # Quantum circuit specific settings
    config.task.quantum_circuit.num_qubits = 2  # Start with 2 qubits
    config.task.quantum_circuit.gates = ["H", "X", "Y", "Z"]  # Basic gate set
    
    # Temperature conditioning
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [2.0]  # Fixed temperature
    config.cond.temperature.num_thermometer_dim = 1

    # Training settings
    config.algo.sampling_tau = 0.95
    config.algo.train_random_action_prob = 0.01
    config.algo.tb.Z_learning_rate = 1e-3

    # Create and run trainer
    trainer = QuantumCircuitTrainer(config)
    trainer.run()

if __name__ == "__main__":
    main() 