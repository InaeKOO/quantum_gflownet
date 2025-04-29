import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Tuple
from torch import Tensor

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward
from gflownet.envs.circuit_building_env import CircuitBuildingEnv, AutoregressiveCircuitBuildingContext
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.models.circuit_transformer import CircuitTransformerGFN

import math

def calculate_fidelity(circuit_str: str, target: torch.Tensor) -> float:
    """
    circuit_str: e.g. "Qubit 0: H X Z\nQubit 1: Y Z"
    target: [d,d] tensor (d=2**num_qubits) already on correct device

    returns fidelity = |Tr(U^† target)| / d
    """
    # Always use the same device as target
    device = target.device
    dtype = target.dtype

    # Parse circuit
    lines = [l.strip() for l in circuit_str.splitlines() if l.strip()]
    seqs: List[List[str]] = [l.split(":", 1)[1].split() for l in lines]
    n = len(seqs)
    d = 1 << n

    # Pre-define gates on device
    SQ = {
        'H': (1/math.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device),
        'X': torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device),
        'Y': torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device),
        'Z': torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device),
    }

    # Initialize U
    U = torch.eye(d, dtype=dtype, device=device)

    # Build U matrix
    for q_idx, gates in enumerate(seqs):
        for g in gates:
            if g not in SQ:
                raise ValueError(f"Unknown gate: {g}")
            G = SQ[g]
            op = None
            for i in range(n):
                m = G if i == q_idx else torch.eye(2, dtype=dtype, device=device)
                op = m if op is None else torch.kron(op, m)
            U = op @ U

    # Compute fidelity entirely on correct device
    prod = U.conj().transpose(-2, -1) @ target
    fid = torch.abs(torch.einsum('ii->', prod)) / d
    return fid.item()

class QuantumCircuitTask(GFNTask):
    """Sets up a task where the reward is computed using the fidelity between 
    the constructed quantum circuit and a target unitary matrix.
    """
    def __init__(
        self,
        matrix: torch.Tensor,
        num_qubits: int,
        cfg: Config,
        wrap_model: Optional[Callable[[nn.Module], nn.Module]] = None,
    ) -> None:
        super().__init__()
        # Move target unitary to the correct device
        self.device = torch.device(cfg.device)
        self.matrix = matrix.to(self.device)
        self.num_qubits = num_qubits

        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(obj_props)))

    def compute_obj_properties(self, objs: List[str]) -> Tuple[ObjectProperties, Tensor]:
        rs = torch.tensor([calculate_fidelity(c, self.matrix) for c in objs]).float()
        return ObjectProperties(rs[:, None]), torch.ones(len(objs), dtype=torch.bool)

class QuantumCircuitTrainer(StandardOnlineTrainer):
    task: QuantumCircuitTask

    def set_default_hps(self, cfg: Config):
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
        self.model = CircuitTransformerGFN(self.ctx, self.cfg)

    def setup_task(self):
        matrix = self.generate_target_unitary()
        print("Target: ", matrix)
        self.task = QuantumCircuitTask(
            matrix=matrix,
            num_qubits=self.cfg.task.quantum_circuit.num_qubits,
            cfg=self.cfg,
        )

    def setup_env_context(self):
        self.env = CircuitBuildingEnv(num_qubits=self.cfg.task.quantum_circuit.num_qubits)
        self.ctx = AutoregressiveCircuitBuildingContext(
            gates=self.cfg.task.quantum_circuit.gates,
            num_qubits=self.cfg.task.quantum_circuit.num_qubits,
            num_cond_dim=self.task.num_cond_dim,
        )

    def generate_target_unitary(self) -> torch.Tensor:
        """Generate a random target unitary matrix"""
        num_qubits = self.cfg.task.quantum_circuit.num_qubits
        d = 2 ** num_qubits
        device = torch.device(self.cfg.device)
        dtype = torch.complex128
        
        real = torch.randn(d, d, dtype=dtype, device=device).real
        imag = torch.randn(d, d, dtype=dtype, device=device).real
        Z = torch.complex(real, imag)
        Q, R = torch.linalg.qr(Z)
        diag_R = torch.diagonal(R)
        phase = diag_R / torch.abs(diag_R)
        Q = Q * phase.unsqueeze(0)
        return Q 