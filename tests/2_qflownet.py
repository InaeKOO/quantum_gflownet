import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tqdm

H = torch.tensor([[1 / np.sqrt(2), 1 / np.sqrt(2)],
                  [1 / np.sqrt(2), -1 / np.sqrt(2)]], dtype=torch.complex128)
X = torch.tensor([[0, 1],
                  [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j],
                  [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0],
                  [0, -1]], dtype=torch.complex128)
T = torch.tensor([[1, 0],
                  [0, np.exp(1j * np.pi / 4)]], dtype=torch.complex128)
S = torch.tensor([[1, 0],
                  [0, np.exp(1j * np.pi / 2)]], dtype=torch.complex128)

gates = {
    'H1': torch.kron(H, torch.eye(2, dtype=torch.complex128)),
    'H2': torch.kron(torch.eye(2, dtype=torch.complex128), H),
    'X1': torch.kron(X, torch.eye(2, dtype=torch.complex128)),
    'X2': torch.kron(torch.eye(2, dtype=torch.complex128), X),
    'Y1': torch.kron(Y, torch.eye(2, dtype=torch.complex128)),
    'Y2': torch.kron(torch.eye(2, dtype=torch.complex128), Y),
    'Z1': torch.kron(Z, torch.eye(2, dtype=torch.complex128)),
    'Z2': torch.kron(torch.eye(2, dtype=torch.complex128), Z),
    'T1': torch.kron(T, torch.eye(2, dtype=torch.complex128)),
    'T2': torch.kron(torch.eye(2, dtype=torch.complex128), T),
    'S1': torch.kron(S, torch.eye(2, dtype=torch.complex128)),
    'S2': torch.kron(torch.eye(2, dtype=torch.complex128), S),
    'CX1': torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.complex128),
    'CX2': torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.complex128),
}

sorted_keys = sorted(gates.keys())

def random_unitary(num_qubits: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    d = 2 ** num_qubits
    real = torch.randn(d, d, dtype=dtype)
    imag = torch.randn(d, d, dtype=dtype)
    Z = real + 1j * imag
    Q, R = torch.linalg.qr(Z)
    phase = torch.diagonal(R) / torch.abs(torch.diagonal(R))
    return Q * phase.unsqueeze(0)

def has_overlap(circuit: list[str]) -> bool:
    return 'CX1' in circuit and 'CX2' in circuit

def fidelity_reward(circuit: list[str], target: torch.Tensor) -> float:
    if has_overlap(circuit):
        return 0.0
    current = torch.eye(4, dtype=target.dtype, device=target.device)
    for gate_name in circuit:
        current = gates[gate_name] @ current
    fid = torch.abs(torch.trace(current.conj().T @ target)) / 4
    return fid.item()

def circuit_to_tensor(circuit: list[str]) -> torch.Tensor:
    return torch.tensor([k in circuit for k in sorted_keys], dtype=torch.float32)

class FlowModel(nn.Module):
    def __init__(self, num_hid: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(len(sorted_keys), num_hid),
            nn.LeakyReLU(),
            nn.Linear(num_hid, len(sorted_keys))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).exp() * (1 - x)

def circuit_parents(circuit):
   if not circuit:  # If circuit is empty, return empty lists
       return [], []
   parent_actions = []
   parent_states = []
   # Create parent state by removing the last gate
   i = len(circuit) - 1
   last = sorted_keys.index(circuit[i])
   parent_actions.append(last)
   parent_states.append(circuit[:i])
   while i - 1 >= 0:  
    before_last = sorted_keys.index(circuit[i-1])
    if last[-1] != before_last[-1]:
      parent_actions.append(before_last)  
      parent_states.append(circuit[:i-1]+circuit[i:])
      break
    else:
        i -= 1
   return parent_states, parent_actions

class TBModel(nn.Module):
    def __init__(self, num_hid: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(len(sorted_keys), num_hid),
            nn.LeakyReLU(),
            nn.Linear(num_hid, len(sorted_keys) * 2)
        )
        self.logZ = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor):
        logits = self.mlp(x)
        P_F = logits[..., :len(sorted_keys)] * (1 - x) + x * -100
        P_B = logits[..., len(sorted_keys):] * x + (1 - x) * -100
        return P_F, P_B

def main():
    F_sa = FlowModel(512)
    opt = torch.optim.Adam(F_sa.parameters(), 3e-4)
    target = random_unitary(2)
    losses = []
    minibatch_loss = 0
    update_freq = 4

    for episode in tqdm.tqdm(range(50000), ncols=40):
        state = []
        edge_flow_prediction = F_sa(circuit_to_tensor(state))

        for t in range(20):
            policy = edge_flow_prediction / edge_flow_prediction.sum()
            action = Categorical(probs=policy).sample()
            new_state = state + [sorted_keys[action]]

            parent_states, parent_actions = circuit_parents(new_state)
            px = torch.stack([circuit_to_tensor(p) for p in parent_states])
            pa = torch.tensor(parent_actions).long()
            parent_edge_flow_preds = F_sa(px)[torch.arange(len(parent_states)), pa]

            if t == 19:
                reward = fidelity_reward(new_state, target)
                edge_flow_prediction = torch.zeros(len(sorted_keys))
            else:
                reward = 0
                edge_flow_prediction = F_sa(circuit_to_tensor(new_state))

            flow_mismatch = (parent_edge_flow_preds.sum() - edge_flow_prediction.sum() - reward).pow(2)
            minibatch_loss += flow_mismatch
            state = new_state

        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0

    plt.figure(figsize=(10, 3))
    plt.plot(losses)
    plt.yscale('log')
    plt.show()

def main_tb():
    model = TBModel(512)
    opt = torch.optim.Adam(model.parameters(), 3e-4)
    target = random_unitary(2)
    tb_losses = []
    logZs = []
    minibatch_loss = 0
    update_freq = 2
    sampled_circuits = []

    for episode in tqdm.tqdm(range(50000), ncols=40):
        state = []
        P_F_s, P_B_s = model(circuit_to_tensor(state))
        total_P_F, total_P_B = 0, 0

        for t in range(10):
            cat = Categorical(logits=P_F_s)
            action = cat.sample()
            new_state = state + [sorted_keys[action]]
            total_P_F += cat.log_prob(action)

            if t == 9:
                reward = torch.tensor(fidelity_reward(new_state, target)).float()
            P_F_s, P_B_s = model(circuit_to_tensor(new_state))
            total_P_B += Categorical(logits=P_B_s).log_prob(action)

            state = new_state

        loss = (model.logZ + total_P_F - torch.log(reward).clip(min=-20) - total_P_B).pow(2)
        minibatch_loss += loss

        if episode % update_freq == 0:
            tb_losses.append(minibatch_loss.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
            logZs.append(model.logZ.item())
            sampled_circuits.append(state)

    f, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(tb_losses)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('loss')
    ax[1].plot(np.exp(logZs))
    ax[1].set_ylabel('estimated Z')
    plt.show()
    print(f"Final logZ: {model.logZ.exp()}")
    print(f"Number of circuits sampled: {len(sampled_circuits)}")
    print(f"Last 5 circuits: {sampled_circuits[-5:]}")
    print(f"Best circuit fidelity: {max(fidelity_reward(circuit, target) for circuit in sampled_circuits)}")


if __name__ == "__main__":
    main_tb()
