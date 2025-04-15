import socket
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.circuit_building_env import AutoregressiveCircuitBuildingContext, CircuitBuildingEnv
from gflownet.models.circuit_transformer import CircuitTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward


class ToyCircuitTask(GFNTask):
    """Sets up a task where the reward is based on the presence of specific gate patterns in the quantum circuit.
    The reward is normalized to be in [0,1]."""

    def __init__(
        self,
        patterns: List[str],  # List of gate patterns to look for
        num_qubits: int,  # Number of qubits in the circuit
        cfg: Config,
    ) -> None:
        self.patterns = patterns
        self.num_qubits = num_qubits
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        self.norm = cfg.algo.max_len / min(map(len, patterns))

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(obj_props)))

    def compute_obj_properties(self, objs: List[str]) -> Tuple[ObjectProperties, Tensor]:
        # Convert circuit string representation to tensor of rewards
        rs = torch.tensor([sum([circuit.count(p) for p in self.patterns]) for circuit in objs]).float() / self.norm
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
        # Define patterns to look for in the circuits
        # For example, "HX" means Hadamard followed by X gate on the same qubit
        patterns = ["HX", "XH", "HH"]
        self.task = ToyCircuitTask(
            patterns=patterns,
            num_qubits=2,  # Using 2 qubits for this toy example
            cfg=self.cfg,
        )

    def setup_env_context(self):
        # Define available gates and number of qubits
        gates = ["H", "X", "Y", "Z", "C"]  # Basic quantum gates
        self.env = CircuitBuildingEnv(num_qubits=2)  # 2-qubit circuits
        self.ctx = AutoregressiveCircuitBuildingContext(
            gates=gates,
            num_qubits=2,
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
    config.print_every = 1
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [2.0]
    config.cond.temperature.num_thermometer_dim = 1
    config.algo.train_random_action_prob = 0.05

    trial = ToyCircuitTrainer(config)
    trial.run()


if __name__ == "__main__":
    main() 