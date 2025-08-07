import torch
import torch.multiprocessing as mp

import socket
import os
from typing import Dict, List, Tuple
import math
import inspect
from datetime import datetime

import numpy as np
from torch import Tensor
from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.circuit_building_env import AutoregressiveCircuitBuildingContext, CircuitBuildingEnv
from gflownet.models.circuit_transformer import CircuitTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward
from gflownet.utils.circuit import sequence_to_matrices, total_matrix

# Import additional components for gate length prediction
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ============= Gate Length Predictor Components (copied from unitary_num.py) =============

def unitary_to_tensor(U: torch.Tensor) -> torch.Tensor:
    """
    U: [B, N, N] complex tensor or numpy array
    return: [B, 2, N, N] float tensor (real/imag)
    """
    if not torch.is_tensor(U):
        U = torch.from_numpy(U)
    if not torch.is_complex(U):
        raise ValueError("U must be complex dtype")
    x = torch.stack([U.real, U.imag], dim=1).float()
    return x

@dataclass
class Unitary_encoder_config:
    cond_emb_size: int
    model_features: list
    num_heads: int
    transformer_depths: list
    dropout: float

class PositionalEncoding2D(nn.Module):
    """Simple 2D positional encoding (placeholder - you may need to implement based on genQC)"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x):
        # Simple implementation - just return input for now
        # You might need to implement proper 2D positional encoding
        return x

class DownBlock2D(nn.Module):
    """Simple downsampling block"""
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SpatialTransformerSelfAttn(nn.Module):
    """Simplified spatial transformer with self-attention"""
    def __init__(self, channels, num_heads=8, depth=4, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
            for _ in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(depth)])
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # [B, HW, C]
        
        for layer, norm in zip(self.layers, self.norms):
            x_res = x_flat
            x_flat = norm(x_flat)
            x_flat, _ = layer(x_flat, x_flat, x_flat)
            x_flat = x_flat + x_res
        
        x = x_flat.permute(0, 2, 1).view(B, C, H, W)
        return x

class Config_Model(nn.Module):
    """Base config model class"""
    def __init__(self):
        super().__init__()

class Unitary_encoder(Config_Model):
    """Encoder for unitary conditions."""
    def __init__(self, cond_emb_size, model_features=None, num_heads=8, transformer_depths=(4, 4), dropout=0.1):
        super().__init__()
        self.cond_emb_size = cond_emb_size

        if model_features is None:
            in_ch   = 2
            mid_ch1 = cond_emb_size // 4
            mid_ch2 = cond_emb_size // 2
            out_ch  = cond_emb_size
            model_features = [in_ch, mid_ch1, mid_ch2, out_ch]
        else:
            assert len(model_features) == 4
            in_ch, mid_ch1, mid_ch2, out_ch = model_features

        self.params_config = Unitary_encoder_config(cond_emb_size, model_features, num_heads, list(transformer_depths), dropout)

        self.conv_in = nn.Conv2d(in_ch, mid_ch1, kernel_size=1, stride=1, padding=0)
        self.pos_enc = PositionalEncoding2D(d_model=mid_ch1)

        self.down1 = DownBlock2D(mid_ch1, mid_ch2, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        assert len(transformer_depths) == 2
        self.spatialTransformer1 = SpatialTransformerSelfAttn(
            mid_ch1, num_heads=num_heads, depth=transformer_depths[0], dropout=dropout
        )
        self.spatialTransformer2 = SpatialTransformerSelfAttn(
            mid_ch2, num_heads=num_heads, depth=transformer_depths[1], dropout=dropout
        )

        self.head = nn.Conv2d(mid_ch2, out_ch, kernel_size=1, stride=1, padding=0)

        self._init_weights()

    def _init_weights(self):
        self.head.weight.data.zero_()

    def forward(self, x):
        # x: [B, 2, 2^n, 2^n]
        b, *_ = x.shape

        x = self.conv_in(x)
        x = self.pos_enc(x)

        x = self.spatialTransformer1(x)
        x = self.down1(x)

        x = self.spatialTransformer2(x)

        x = self.head(x)
        x = torch.reshape(x, (b, self.cond_emb_size, -1))
        x = torch.permute(x, (0, 2, 1))  # [B, seq, ch]
        return x

class GateLenHead(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=None, dropout=0.1):
        super().__init__()
        if hidden is None:
            hidden = in_dim // 2
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, z_seq):
        # z_seq: [B, seq, in_dim]
        z = z_seq.mean(dim=1)  # GAP
        return self.fc(z)      # [B, n_classes]

class GateLenPredictor(nn.Module):
    def __init__(self, encoder: nn.Module, cond_emb_size: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head    = GateLenHead(cond_emb_size, n_classes)
    def forward(self, U):
        x = unitary_to_tensor(U)     # [B, 2, N, N]
        z_seq = self.encoder(x)      # [B, seq, cond_emb_size]
        logits = self.head(z_seq)    # [B, n_classes]
        return logits

# ============= End of copied components =============

# Global variable to hold the loaded gate length predictor model
_gate_len_predictor = None

def load_gate_len_predictor(model_path: str = "gatelen_predictor.pt", device: str = "cuda"):
    """Load the pre-trained gate length predictor model."""
    global _gate_len_predictor
    
    if _gate_len_predictor is None:
        # Default model path relative to genQC directory
        if model_path is None:
            genqc_dir = os.path.join(os.path.dirname(__file__), '../../../..', 'genQC')
            model_path = os.path.join(genqc_dir, "gatelen_predictor.pt")
        
        # Rebuild model with same architecture as training
        cond_emb_size = 256
        max_len = 12
        
        encoder = Unitary_encoder(cond_emb_size=cond_emb_size)
        _gate_len_predictor = GateLenPredictor(
            encoder, 
            cond_emb_size=cond_emb_size, 
            n_classes=max_len + 1
        ).to(device)
        
        # Load weights
        state = torch.load(model_path, map_location=device)
        _gate_len_predictor.load_state_dict(state)
        _gate_len_predictor.eval()
        
        print(f"Loaded gate length predictor from {model_path}")
    
    return _gate_len_predictor

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
    I = torch.eye(d, dtype=dtype, device=device)
    return total_matrix(sequence_to_matrices("WFNAPIG"))

def reward(x):
    #k = 41.821  # steepness
    #t = 0.8465  # threshold
    #s = 1 / (1 + np.exp(-k * (x - t)))
    b = math.log(0.1) / -0.25  # ≈ 9.21034
    return math.exp(b * (x-1))
    return np.clip(x * s, 0.0, 1.0)

def calculate_fidelity(circuit_str: str, target: torch.Tensor, device: str = "cuda") -> float:
    """
    Calculate reward based on how close the predicted gate length is to the target length.
    
    circuit_str: e.g. "HXZYZ" (gates sequence)
    target_length: desired gate length
    device: device to run the model on
    
    returns: reward based on gate length prediction accuracy
    """
    # Load the model if not already loaded
    model = load_gate_len_predictor(device=device)
    
    # Apply some basic constraints (same as before)
    if circuit_str.count('P') > 2 or circuit_str.count('Q') > 2 or circuit_str.count('R') > 2:
        return 0
    for i in range(1, len(circuit_str)):
        if circuit_str[i] == circuit_str[i-1]:
            return 0
    
    # Generate the unitary matrix from the circuit
    circuit_mat = total_matrix(sequence_to_matrices(circuit_str))
    circuit_mat = circuit_mat.to(device)
    
    # Add batch dimension for the model
    circuit_mat_batch = circuit_mat.unsqueeze(0)  # [1, d, d]
    
    with torch.no_grad():
        # Get predicted gate length
        logits = model(circuit_mat_batch)  # [1, n_classes]
        predicted_length = logits.argmax(dim=1).item()
    
    # Calculate reward based on how close the prediction is to target
    return -predicted_length


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
        self.matrix = matrix
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
        cfg.algo.max_nodes = 7
        cfg.algo.max_len = 7
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


def main():
    config = init_empty(Config())
    config.log_dir = f"./logs/debug_run_toy_circuit_{datetime.now().strftime('%Y%m%d_%H%M')}"
    config.device = "cuda"
    config.overwrite_existing_exp = True
    config.num_training_steps = 100000
    config.checkpoint_every = 200
    config.num_workers = 0
    config.task.toy_circuit.num_qubits = 3
    config.task.toy_circuit.gates = "ABDFGIKLNPQRSTUVWX"
    config.print_every = 1
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [2.0]
    config.cond.temperature.num_thermometer_dim = 1
    config.algo.train_random_action_prob = 0.05
    
    trial = ToyCircuitTrainer(config)
    trial.run()

if __name__ == "__main__":
    main()