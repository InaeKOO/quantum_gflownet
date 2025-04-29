import torch
import torch.multiprocessing as mp

import socket
from typing import Dict, List, Tuple
import math

import numpy as np
from torch import Tensor
from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.circuit_building_env import AutoregressiveCircuitBuildingContext, CircuitBuildingEnv
from gflownet.models.circuit_transformer import CircuitTransformerGFN, generate_square_subsequent_mask
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward


def random_unitary(
        num_qubits: int,
        dtype: torch.dtype = torch.complex128,
        device: torch.device | str = "cpu",
) -> torch.Tensor:
    d = 2 ** num_qubits
    real = torch.randn(d, d, dtype=dtype, device=device).real
    imag = torch.randn(d, d, dtype=dtype, device=device).real
    Z = torch.complex(real, imag)
    Q, R = torch.linalg.qr(Z)
    diag_R = torch.diagonal(R)
    phase = diag_R / torch.abs(diag_R)
    Q = Q * phase.unsqueeze(0)
    return Q

def string_to_gate():
    pass

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


class ToyCircuitTask(GFNTask):
    def __init__(
        self,
        matrix: torch.Tensor,
        num_qubits: int,
        cfg: Config,
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
        self.model = CircuitTransformerGFN(self.ctx, self.cfg)

    def setup_task(self):
        matrix = random_unitary(
            num_qubits=self.cfg.task.toy_circuit.num_qubits,
            device=self.device
        )
        print("Target: ", matrix)
        self.task = ToyCircuitTask(
            matrix=matrix,
            num_qubits=self.cfg.task.toy_circuit.num_qubits,
            cfg=self.cfg,
        )

    def setup_env_context(self):
        self.env = CircuitBuildingEnv(num_qubits=self.cfg.task.toy_circuit.num_qubits)
        self.ctx = AutoregressiveCircuitBuildingContext(
            gates=self.cfg.task.toy_circuit.gates,
            num_qubits=self.cfg.task.toy_circuit.num_qubits,
            num_cond_dim=self.task.num_cond_dim,
        )

    def setup_algo(self):
        super().setup_algo()
        self.algo.model_is_autoregressive = True

def test_shapes():
    # 1) 빈 Config를 초기화하고 필수 필드 설정
    cfg = init_empty(Config())
    cfg.log_dir = "./logs/debug_run_toy_circuit_test"
    cfg.overwrite_existing_exp = True
    cfg.device = "cpu"
    cfg.task.toy_circuit.num_qubits = 1
    cfg.task.toy_circuit.gates = ["H", "X", "Y", "Z"]
    
    # 2) Trainer를 통해 default hps 및 모델/컨텍스트 설정
    trainer = ToyCircuitTrainer(cfg)
    trainer.set_default_hps(cfg)
    trainer.setup_env_context()
    trainer.setup_task()
    trainer.setup_model()
    
    model = trainer.model
    ctx   = trainer.ctx
    task  = trainer.task
    
    # 3) 더미 배치 생성
    batch_size = 4
    seq_len    = cfg.algo.max_len     # ToyCircuitTrainer에서 설정된 max_len
    emb_dim    = cfg.model.num_emb    # default hps의 embedding 차원
    device     = torch.device(cfg.device)
    
    # 모델에 들어갈 x: (batch_size, seq_len, emb_dim)
    dummy_x = torch.randn(batch_size, seq_len, emb_dim, device=device)
    # padding mask: (batch_size, seq_len)
    dummy_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    # conditional info
    dummy_ci = task.sample_conditional_information(batch_size, train_it=0)
    
    # 4) 차원 출력
    print(f"x.shape               = {tuple(dummy_x.shape)}")
    print(f"padding_mask.shape    = {tuple(dummy_padding_mask.shape)}")
    attn_mask = generate_square_subsequent_mask(seq_len).to(device)
    print(f"attn_mask.shape       = {tuple(attn_mask.shape)}")
    
    # 5) TransformerEncoder 직접 호출해보기
    #    CircuitTransformerGFN 내부의 encoder만 테스트
    encoder = model.encoder
    out = encoder(
        dummy_x,
        mask=attn_mask,
        src_key_padding_mask=dummy_padding_mask
    )
    print(f"encoder output shape  = {tuple(out.shape)}")


def main():
    config = init_empty(Config())
    config.log_dir = "./logs/debug_run_toy_circuit"
    config.device = "cuda"
    config.overwrite_existing_exp = True
    config.num_training_steps = 2000
    config.checkpoint_every = 200
    config.num_workers = 0
    config.task.toy_circuit.num_qubits = 1
    config.task.toy_circuit.gates = ["H", "X", "Y", "Z"]
    config.print_every = 1
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [2.0]
    config.cond.temperature.num_thermometer_dim = 1
    config.algo.train_random_action_prob = 0.05

    trial = ToyCircuitTrainer(config)
    trial.run()

if __name__ == "__main__":
    test_shapes()
