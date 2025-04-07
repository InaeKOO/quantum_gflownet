import torch
from typing import Callable, Dict, List, Optional, Tuple
from torch import Tensor

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward


class QuantumCircuitTask(GFNTask):
    """
    Task definition for quantum circuit synthesis using GFlowNet.
    Replaces molecule-based reward with fidelity between generated and target unitary.
    """
    def __init__(
        self,
        cfg: Config,
        wrap_model: Optional[Callable] = None,
        target_unitary: Optional[Tensor] = None,
        num_qubits: int = 2,
    ) -> None:
        self.cfg = cfg
        self._wrap_model = wrap_model if wrap_model is not None else (lambda x: x)
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

        # Custom quantum circuit fields
        self.target_unitary = target_unitary  # Must be set externally if not passed here
        self.num_qubits = num_qubits

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def compute_reward_from_tensor(self, circuits: List[Tensor]) -> Tensor:
        """
        Args:
            circuits: List of (n+3, n, l) shaped circuit tensors
        Returns:
            reward tensor of shape (B,)
        """
        rewards = []
        for c in circuits:
            actions = self.circuit_tensor_to_actions(c)
            gate_seq = [self.gate_from_action(a) for a in actions]
            U = torch.eye(2 ** self.num_qubits, dtype=torch.cfloat, device=self.target_unitary.device)
            for g in gate_seq:
                U = g @ U
            d = U.shape[0]
            fidelity = torch.abs(torch.trace(U.conj().T @ self.target_unitary)) / d
            rewards.append(fidelity ** 2)

        return torch.tensor(rewards, dtype=torch.float32, device="cpu").clamp(min=1e-4, max=1.0)

    def compute_obj_properties(self, circuits: List[Tensor]) -> Tuple[ObjectProperties, Tensor]:
        rewards = self.compute_reward_from_tensor(circuits)
        return ObjectProperties(rewards.unsqueeze(1)), torch.ones(len(circuits), dtype=torch.bool)

    def circuit_tensor_to_actions(self, circuit_tensor: Tensor) -> List[Tensor]:
        n_plus_3, n, l = circuit_tensor.shape
        assert n_plus_3 >= 4, "tensor must include at least 4 features: gate_type, q1, q2, theta"
        actions = []
        for layer in range(l):
            for qubit in range(n):
                gate_type = circuit_tensor[0, qubit, layer]
                q1 = circuit_tensor[1, qubit, layer]
                q2 = circuit_tensor[2, qubit, layer]
                theta = circuit_tensor[3, qubit, layer]
                actions.append(torch.tensor([gate_type, q1, q2, theta], dtype=torch.float32))
        return actions

    def gate_from_action(self, action: Tensor) -> Tensor:
        # Replace this with your actual gate mapping logic
        gate_type, q1, q2, theta = action.int()
        # Dummy single-qubit RX gate example
        g = torch.tensor([
            [torch.cos(theta / 2), -1j * torch.sin(theta / 2)],
            [-1j * torch.sin(theta / 2), torch.cos(theta / 2)]
        ], dtype=torch.cfloat)
        return self.expand_single_qubit_gate(g, int(q1))

    def expand_single_qubit_gate(self, gate: Tensor, target: int) -> Tensor:
        I = torch.eye(2, dtype=torch.cfloat)
        ops = [I] * self.num_qubits
        ops[target] = gate
        U = ops[0]
        for i in range(1, self.num_qubits):
            U = torch.kron(U, ops[i])
        return U

if __name__ == "__main__":
    pass
