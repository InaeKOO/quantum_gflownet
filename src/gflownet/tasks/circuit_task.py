import socket
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.circuit_building_env import CircuitBuildingEnv, QuantumCircuit, QuantumGateType
from gflownet.models.circuit_transformer import CircuitTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.misc import get_worker_device
from gflownet.utils.transforms import to_logreward


class QuantumCircuitTask(GFNTask):
    """Sets up a task for generating quantum circuits with specific properties.
    
    The reward is computed based on the circuit's ability to perform a target quantum operation
    or achieve a specific quantum state. This is a placeholder for actual quantum circuit evaluation.
    """

    def __init__(
        self,
        cfg: Config,
        wrap_model: Optional[Callable[[nn.Module], nn.Module]] = None,
    ) -> None:
        self._wrap_model = wrap_model if wrap_model is not None else (lambda x: x)
        self.cfg = cfg
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        self.env_ctx = CircuitBuildingEnv(
            max_qubits=cfg.task.circuit.max_qubits,
            max_gates=cfg.task.circuit.max_gates
        )
        self.models = self._load_task_models()

    def _load_task_models(self):
        model = CircuitTransformerGFN(
            env_ctx=self.env_ctx,
            cfg=self.cfg,
            num_graph_out=1,
            do_bck=True
        )
        model.to(get_worker_device())
        model = self._wrap_model(model)
        return {"circuit": model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def compute_reward_from_graph(self, graphs: List[Data]) -> Tensor:
        """Compute reward for quantum circuits.
        
        This is a placeholder for actual quantum circuit evaluation.
        In practice, this would evaluate the circuit's ability to perform
        a target quantum operation or achieve a specific quantum state.
        """
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(get_worker_device())
        
        # Placeholder reward computation
        # In practice, this would involve:
        # 1. Converting the graph to a quantum circuit
        # 2. Simulating the circuit
        # 3. Computing fidelity with target state/operation
        preds = torch.ones(len(graphs), device=get_worker_device())
        return preds.clip(1e-4, 100).reshape((-1,))

    def compute_obj_properties(self, circuits: List[QuantumCircuit]) -> Tuple[ObjectProperties, Tensor]:
        """Compute properties of quantum circuits."""
        graphs = [circuit for circuit in circuits]
        is_valid = torch.tensor([True for _ in graphs]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid

        preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))
        assert len(preds) == is_valid.sum()
        return ObjectProperties(preds), is_valid


# Example target circuits for training
TARGET_CIRCUITS = [
    # Example 1: Bell state preparation
    {
        "qubits": 2,
        "gates": [
            (QuantumGateType.H, [0], {}),
            (QuantumGateType.CNOT, [0, 1], {})
        ]
    },
    # Example 2: GHZ state preparation
    {
        "qubits": 3,
        "gates": [
            (QuantumGateType.H, [0], {}),
            (QuantumGateType.CNOT, [0, 1], {}),
            (QuantumGateType.CNOT, [1, 2], {})
        ]
    },
    # Example 3: Quantum Fourier Transform (2 qubits)
    {
        "qubits": 2,
        "gates": [
            (QuantumGateType.H, [0], {}),
            (QuantumGateType.S, [1], {}),
            (QuantumGateType.CNOT, [0, 1], {}),
            (QuantumGateType.H, [1], {})
        ]
    }
]


class QuantumCircuitDataset(Dataset):
    """Dataset of example quantum circuits for training."""

    def __init__(self, circuits) -> None:
        super().__init__()
        self.props: ObjectProperties
        self.circuits: List[QuantumCircuit] = []
        self.circuit_specs = circuits

    def setup(self, task: QuantumCircuitTask, ctx: CircuitBuildingEnv) -> None:
        self.circuits = []
        for spec in self.circuit_specs:
            circuit = QuantumCircuit()
            # Add qubits
            for i in range(spec["qubits"]):
                circuit.add_qubit(i)
            # Add gates
            for gate_type, qubits, params in spec["gates"]:
                circuit.add_gate(gate_type, qubits, params)
            self.circuits.append(circuit)
        
        self.props = task.compute_obj_properties(self.circuits)[0]

    def __len__(self):
        return len(self.circuits)

    def __getitem__(self, index):
        return self.circuits[index], self.props[index]


class QuantumCircuitTrainer(StandardOnlineTrainer):
    task: QuantumCircuitTask
    training_data: QuantumCircuitDataset

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.num_from_policy = 64
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 20  # Maximum number of qubits
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_num_from_policy = 64
        cfg.num_validation_gen_steps = 10
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_sample_p_b = True

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

    def setup_task(self):
        self.task = QuantumCircuitTask(
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

    def setup_data(self):
        super().setup_data()
        self.training_data = QuantumCircuitDataset(TARGET_CIRCUITS)

    def setup_env_context(self):
        self.ctx = CircuitBuildingEnv(
            max_qubits=self.cfg.algo.max_nodes,
            max_gates=100  # Maximum number of gates allowed
        )

    def setup(self):
        super().setup()
        self.training_data.setup(self.task, self.ctx)


def main():
    """Example of how this model can be run."""
    import datetime

    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = f"./logs/debug_run_quantum_circuit_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True
    config.num_training_steps = 1_00
    config.validate_every = 20
    config.num_final_gen_steps = 10
    config.num_workers = 1
    config.opt.lr_decay = 20_000
    config.algo.sampling_tau = 0.99
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0, 64.0]

    trial = QuantumCircuitTrainer(config)
    trial.run()


if __name__ == "__main__":
    main() 