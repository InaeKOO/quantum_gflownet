import pathlib
import torch
import numpy as np
from typing import Any, Dict, List
from torch import Tensor
from torch.utils.data import DataLoader

from gflownet.algo.envelope_q_learning import EnvelopeQLearning
from gflownet.algo.multiobjective_reinforce import MultiObjectiveReinforce
from gflownet.config import Config, init_empty
from gflownet.data.data_source import DataSource
from gflownet.tasks.circuit_task import QuantumCircuitTask
from gflownet.utils.conditioning import MultiObjectiveWeightedPreferences
from gflownet.utils.multiobjective_hooks import MultiObjectiveStatsHook, TopKHook
from gflownet.utils.sqlite_log import SQLiteLogHook
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet import ObjectProperties


class QuantumCircuitTrainer(StandardOnlineTrainer):
    task: QuantumCircuitTask

    def set_default_hps(self, cfg: Config):
        cfg.pickle_mp_messages = False
        cfg.num_workers = 4
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10

        cfg.algo.num_from_policy = 64
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_num_from_policy = 64
        cfg.num_validation_gen_steps = 10
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_sample_p_b = True

        cfg.replay.use = False
        cfg.replay.capacity = 10000
        cfg.replay.warmup = 1000

    def setup_task(self):
        self.task = QuantumCircuitTask(
            cfg=self.cfg,
            target_unitary=self.cfg.task.qcs.target_unitary,
            num_qubits=self.cfg.task.qcs.num_qubits,
        )

    def setup(self):
        self.sampling_hooks = []
        self.valid_sampling_hooks = []
        self.to_terminate = []
        super().setup()
        if hasattr(self.cfg.task.qcs, "log_topk") and self.cfg.task.qcs.log_topk:
            self._top_k_hook = TopKHook(10, self.cfg.task.qcs.n_valid_repeats, self.cfg.task.qcs.n_valid)
            self.valid_sampling_hooks.append(self._top_k_hook)

            self.sampling_hooks.append(
                MultiObjectiveStatsHook(
                    256,
                    self.cfg.log_dir,
                    compute_igd=False,
                    compute_pc_entropy=False,
                    compute_focus_accuracy=False,
                    focus_cosim=0.0,
                )
            )
            self.to_terminate.append(self.sampling_hooks[-1].terminate)

    def build_validation_data_loader(self) -> DataLoader:
        model = self._wrap_for_mp(self.model)

        valid_cond_vectors = np.ones((self.cfg.task.qcs.n_valid, 1))
        valid_cond_vectors = torch.tensor(valid_cond_vectors, dtype=torch.float32)
        test_data = RepeatedCondInfoDataset(valid_cond_vectors, repeat=self.cfg.task.qcs.n_valid_repeats)

        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True)
        src.do_conditionals_dataset_in_order(test_data, self.cfg.algo.valid_num_from_dataset, model)

        if self.cfg.log_dir:
            src.add_sampling_hook(SQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "valid"), self.ctx))
        for hook in self.valid_sampling_hooks:
            src.add_sampling_hook(hook)

        return self._make_data_loader(src)

    def train_batch(self, batch, epoch_idx: int, batch_idx: int, train_it: int) -> Dict[str, Any]:
        # Optionally log rewards for debugging
        with torch.no_grad():
            circuits = batch.action_data  # assumed to be list of tensors
            rewards = self.task.compute_reward_from_tensor(circuits)
            print(f"[Step {train_it}] Avg Fidelity Reward: {rewards.mean().item():.4f}")
        return super().train_batch(batch, epoch_idx, batch_idx, train_it)


class RepeatedCondInfoDataset:
    def __init__(self, cond_info_vectors, repeat):
        self.cond_info_vectors = torch.as_tensor(cond_info_vectors).float()
        self.repeat = repeat

    def __len__(self):
        return len(self.cond_info_vectors) * self.repeat

    def __getitem__(self, idx):
        return self.cond_info_vectors[int(idx // self.repeat)]

def random_unitary(d: int) -> torch.Tensor:
    Z = torch.randn(d, d, dtype=torch.cfloat)
    Z += 1j * torch.randn(d, d)
    Q, R = torch.linalg.qr(Z)
    diag_R = torch.diagonal(R)
    phase = diag_R / torch.abs(diag_R)
    Q = Q * phase.unsqueeze(0)  # broadcasting 맞춤
    return Q

def main():
    config = init_empty(Config())
    config.desc = "debug_qcs"
    config.log_dir = "./logs/debug_run_qcs"
    config.device = "cuda"
    config.overwrite_existing_exp = True
    config.num_training_steps = 100
    config.validate_every = 10
    config.num_final_gen_steps = 10
    config.num_workers = 1
    config.algo.sampling_tau = 0.99
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0, 64.0]

    config.task.qcs = lambda: None
    config.task.qcs.target_unitary = random_unitary(4)
    config.task.qcs.num_qubits = 2
    config.task.qcs.n_valid = 15
    config.task.qcs.n_valid_repeats = 2
    config.task.qcs.log_topk = True

    trial = QuantumCircuitTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
