import socket
from typing import Dict, List, Tuple

import math
import torch
import numpy as np
from torch import Tensor

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.circuit_building_env import AutoregressiveCircuitBuildingContext, CircuitBuildingEnv
from gflownet.models.circuit_transformer import CircuitTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward

def random_unitary(num_qubits: int, dtype: torch.dtype = torch.complex128,) -> torch.Tensor:
    d = 2 ** num_qubits
    real = torch.randn(d, d, dtype=dtype)
    imag = torch.randn(d, d, dtype=dtype)
    Z = real + 1j * imag
    Q, R = torch.linalg.qr(Z)
    diag_R = torch.diagonal(R)
    phase = diag_R / torch.abs(diag_R)      
    Q = Q * phase.unsqueeze(0)           
    return Q

def calculate_fidelity(circuit_str: str, target: torch.Tensor) -> float:
    """
    circuit_str: 
      Qubit 0: H X Z
      Qubit 1: Y Z
      ...
    target: [d,d] tensor (d=2**num_qubits)

    return: float, fidelity = |Tr(U† target)| / d
    """
    lines = [line.strip() for line in circuit_str.strip().splitlines() if line.strip()]
    qubit_seqs: List[List[str]] = []
    for line in lines:
        # "Qubit i: G1 G2 G3"
        _, rhs = line.split(":", 1)
        gates = rhs.strip().split()
        qubit_seqs.append(gates)

    num_qubits = len(qubit_seqs)
    d = 2 ** num_qubits

    U = torch.eye(d, dtype=target.dtype, device=target.device)

    SQ = {
        'H': (1/math.sqrt(2)) * torch.tensor([[1,  1],
                                              [1, -1]], 
                                             dtype=target.dtype, device=target.device),
        'X': torch.tensor([[0, 1],
                           [1, 0]],
                          dtype=target.dtype, device=target.device),
        'Y': torch.tensor([[0, -1j],
                           [1j, 0]],
                          dtype=target.dtype, device=target.device),
        'Z': torch.tensor([[1,  0],
                           [0, -1]],
                          dtype=target.dtype, device=target.device),
    }

    for q_idx, seq in enumerate(qubit_seqs):
        for g in seq:
            if g in SQ:
                G = SQ[g]
                op = None
                for i in range(num_qubits):
                    mat = G if i == q_idx else torch.eye(2, dtype=target.dtype, device=target.device)
                    op = mat if op is None else torch.kron(op, mat)
            elif g == 'C':
                # TODO: CNOT, CZ 등 다중 큐빗 게이트 구현 필요
                raise NotImplementedError("Gate 'C' (multi-qubit) is not implemented yet.")
            else:
                raise ValueError(f"Unknown gate symbol: {g}")

            U = op @ U

    fid = torch.abs(torch.trace(U.conj().T @ target)) / d
    return fid.item()

class ToyCircuitTask(GFNTask):
    """Sets up a task where the reward is based on the presence of specific gate patterns in the quantum circuit.
    The reward is normalized to be in [0,1]."""

    def __init__(
        self,
        matrix: torch.Tensor,
        num_qubits: int,  # Number of qubits in the circuit
        cfg: Config,
    ) -> None:
        self.num_qubits = num_qubits
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(obj_props)))

    def compute_obj_properties(self, objs: List[str]) -> Tuple[ObjectProperties, Tensor]:
        # Convert circuit string representation to tensor of rewards
        rs = torch.tensor([calculate_fidelity(circuit,self.matrix) for circuit in objs]).float()
        return ObjectProperties(rs[:, None]), torch.ones(len(objs), dtype=torch.bool)


class ToyCircuitTrainer(StandardOnlineTrainer):
    task: ToyCircuitTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.num_validation_gen_steps = 1
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.num_from_policy = 64
        cfg.model.num_emb = 64
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 10
        cfg.algo.max_len = 10
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-2
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

    def setup_model(self):
        self.model = CircuitTransformerGFN(
            self.ctx,
            self.cfg,
        )

    def setup_task(self):
        # Define matrix to look for in the circuits
        matrix = random_unitary(self.cfg.num_qubits)
        # For example, "HX" means Hadamard followed by X gate on the same qubit
        self.task = ToyCircuitTask(
            matrix = matrix,
            num_qubits=self.cfg.num_qubits,  # Using 2 qubits for this toy example
            cfg=self.cfg,
        )

    def setup_env_context(self):
        # Define available gates and number of qubits
        self.env = CircuitBuildingEnv(num_qubits=self.cfg.num_qubits)  # 2-qubit circuits
        self.ctx = AutoregressiveCircuitBuildingContext(
            gates=self.cfg.gates,
            num_qubits=self.cfg.num_qubits,
            num_cond_dim=self.task.num_cond_dim,
        )

    def setup_algo(self):
        super().setup_algo()
        # Enable autoregressive processing for the transformer
        self.algo.model_is_autoregressive = True


def main():
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.log_dir = "./logs/debug_run_toy_circuit"
    config.device = "cuda"
    config.overwrite_existing_exp = True
    config.num_training_steps = 2_000
    config.checkpoint_every = 200
    config.num_workers = 4
    config.num_qubits = 1
    config.gates = ["H", "X", "Y", "Z"]  # Basic quantum gates
    config.print_every = 1
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [2.0]
    config.cond.temperature.num_thermometer_dim = 1
    config.algo.train_random_action_prob = 0.05

    trial = ToyCircuitTrainer(config)
    trial.run()


if __name__ == "__main__":
    main() 