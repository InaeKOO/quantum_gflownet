import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tqdm
import os
import json
from datetime import datetime
import pathlib

I = torch.eye(2, dtype=torch.complex128)
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
CX_12 = torch.kron(torch.kron(torch.eye(2), torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128)), I) + \
          torch.kron(torch.kron(X, torch.tensor([[0, 0], [0, 1]], dtype=torch.complex128)), I)
CX_23 = torch.kron(I, torch.kron(torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128), I)) + \
          torch.kron(I, torch.kron(torch.tensor([[0, 0], [0, 1]], dtype=torch.complex128), X))
CX_31 = torch.kron(torch.kron(torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128), I), I) + \
          torch.kron(torch.kron(torch.tensor([[0, 0], [0, 1]], dtype=torch.complex128), X), I)

Toffoli = torch.tensor([
    [1,0,0,0,0,0,0,0],  # |000>→|000>
    [0,1,0,0,0,0,0,0],  # |001>→|001>
    [0,0,1,0,0,0,0,0],  # |010>→|010>
    [0,0,0,1,0,0,0,0],  # |011>→|011>
    [0,0,0,0,1,0,0,0],  # |100>→|100>
    [0,0,0,0,0,1,0,0],  # |101>→|101>
    [0,0,0,0,0,0,0,1],  # |110>→|111>
    [0,0,0,0,0,0,1,0],  # |111>→|110>
], dtype=torch.complex128)

gates = {
    'H1': torch.kron(H, torch.kron(I, I)),
    'H2': torch.kron(I, torch.kron(H, I)),
    'H3': torch.kron(I, torch.kron(I, H)),
    'X1': torch.kron(X, torch.kron(I, I)),
    'X2': torch.kron(I, torch.kron(X, I)),
    'X3': torch.kron(I, torch.kron(I, X)),
    'Y1': torch.kron(Y, torch.kron(I, I)),
    'Y2': torch.kron(I, torch.kron(Y, I)),
    'Y3': torch.kron(I, torch.kron(I, Y)),
    'Z1': torch.kron(Z, torch.kron(I, I)),
    'Z2': torch.kron(I, torch.kron(Z, I)),
    'Z3': torch.kron(I, torch.kron(I, Z)),
    'T1': torch.kron(T, torch.kron(I, I)),
    'T2': torch.kron(I, torch.kron(T, I)),
    'T3': torch.kron(I, torch.kron(I, T)),
    'S1': torch.kron(S, torch.kron(I, I)),
    'S2': torch.kron(I, torch.kron(S, I)),
    'S3': torch.kron(I, torch.kron(I, S)),
    'CX1': CX_12,
    'CX2': CX_23,
    'CX3': CX_31,
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
    if 'CX1' in circuit and 'CX2' in circuit and 'CX3' in circuit:
        return True
    if circuit.count('CX1') >= 2 or circuit.count('CX2') >= 2 or circuit.count('CX3') >= 2:
        return True
    if circuit.count('CX1') + circuit.count('CX2') + circuit.count('CX3') >= 3:
        return True
    return False

def fidelity_reward(circuit: list[str], target: torch.Tensor) -> float:
    if has_overlap(circuit):
        return 0.0
    current = torch.eye(8, dtype=target.dtype, device=target.device)
    for gate_name in circuit:
        current = gates[gate_name] @ current
    fid = torch.abs(torch.trace(current.conj().T @ target)) / 8
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
      j = i - 1
      while j - 1 >= 0:
        bb_last = sorted_keys.index(circuit[j-1])
        if bb_last[-1] != before_last[-1] and bb_last[-1] != last[-1]:
            parent_actions.append(bb_last)
            parent_states.append(circuit[:j-1]+circuit[j:])
            break
        else:
            j -= 1
      break
    else:
        i -= 1
   return parent_states, parent_actions

def complex_to_dict(complex_tensor):
    """Convert complex tensor to a dictionary of real and imaginary parts"""
    return {
        'real': complex_tensor.real.tolist(),
        'imag': complex_tensor.imag.tolist()
    }

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
    target = random_unitary(3)
    losses = []
    minibatch_loss = 0
    update_freq = 4

    for episode in tqdm.tqdm(range(50000), ncols=100):
        state = []
        edge_flow_prediction = F_sa(circuit_to_tensor(state))

        for t in range(3):
            policy = edge_flow_prediction / edge_flow_prediction.sum()
            action = Categorical(probs=policy).sample()
            new_state = state + [sorted_keys[action]]

            parent_states, parent_actions = circuit_parents(new_state)
            px = torch.stack([circuit_to_tensor(p) for p in parent_states])
            pa = torch.tensor(parent_actions).long()
            parent_edge_flow_preds = F_sa(px)[torch.arange(len(parent_states)), pa]

            if t == 2:
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
    # Training configuration
    num_episodes = 100000  # Increase number of episodes
    max_circuit_length = 50  # Maximum number of gates in a circuit
    hidden_size = 512
    learning_rate = 3e-4
    update_freq = 2

    # Create results directory with timestamp in the same directory as this script
    script_dir = pathlib.Path(__file__).parent.absolute()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(script_dir, f"results_3qubit_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Save training configuration
    config = {
        'num_episodes': num_episodes,
        'max_circuit_length': max_circuit_length,
        'hidden_size': hidden_size,
        'learning_rate': learning_rate,
        'update_freq': update_freq
    }
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    model = TBModel(hidden_size)
    opt = torch.optim.Adam(model.parameters(), learning_rate)
    target = random_unitary(3)
    #target = Toffoli
    fidelities = []
    tb_losses = []
    logZs = []
    minibatch_loss = 0
    sampled_circuits = []
    for episode in tqdm.tqdm(range(num_episodes), ncols=40):
        state = []
        P_F_s, P_B_s = model(circuit_to_tensor(state))
        total_P_F, total_P_B = 0, 0

        for t in range(max_circuit_length):
            cat = Categorical(logits=P_F_s)
            action = cat.sample()
            new_state = state + [sorted_keys[action]]
            total_P_F += cat.log_prob(action)

            if t == max_circuit_length - 1:
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
            fidelities.append(reward)
            sampled_circuits.append(state)

    # Save plots
    f, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(tb_losses)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('loss')
    ax[1].plot(np.exp(logZs))
    ax[1].set_ylabel('estimated Z')
    plt.savefig(os.path.join(results_dir, 'training_plots.png'))
    plt.close()

    # Save training logs
    training_logs = {
        'config': config,
        'losses': tb_losses,
        'logZs': logZs,
        'final_logZ': model.logZ.item(),
        'final_circuits': sampled_circuits[-100:],  # Save last 100 circuits
        'target_unitary': complex_to_dict(target)  # Save the target unitary in a serializable format
    }
    with open(os.path.join(results_dir, 'training_logs.json'), 'w') as f:
        json.dump(training_logs, f, indent=2)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'final_logZ': model.logZ.item(),
        'target_unitary': complex_to_dict(target),
        'config': config
    }, os.path.join(results_dir, 'final_model.pth'))

    print(f"Results saved in directory: {results_dir}")
    print(f"Training configuration: {config}")
    print(f"Final logZ: {model.logZ.exp()}")
    print(f"Number of circuits sampled: {len(sampled_circuits)}")
    print(f"Last 5 circuits: {sampled_circuits[-5:]}")
    print(f"Best circuit fidelity: {max(fidelity_reward(circuit, target) for circuit in sampled_circuits)}")
    plt.plot(fidelities)
    plt.xlabel('Length')
    plt.ylabel('Fidelity')
    plt.title('3-qubit matrix')
    plt.show()

def main_cl():
    # Training configuration
    num_episodes       = 10000   # episodes per try
    curriculum_start   = 3       # 시작 회로 길이
    curriculum_end     = 10      # (exclusive) 최대 길이+1
    hidden_size        = 512
    learning_rate      = 3e-4
    update_freq        = 2
    threshold_factor   = 1.1     # 이전 best_fid * 1.1 이상 되어야 level up
    count = 0

    model = TBModel(hidden_size)
    opt   = torch.optim.Adam(model.parameters(), learning_rate)
    target = random_unitary(3)   # 목표 유니터리
    len_fidelity = []
    last_level_best_fid = 1e-6  # level=3 기저값 (0→곱해도 0 되는 문제 방지)

    for level in range(curriculum_start, curriculum_end):
        target_threshold = last_level_best_fid * threshold_factor
        print(f"\n=== Level {level}: target fidelity ≥ {target_threshold:.4f} ===")

        # 이 레벨에서 threshold를 넘을 때까지 반복
        while True:
            best_fid = 0.0
            minibatch_loss = 0.0

            for episode in tqdm.tqdm(range(num_episodes), ncols=60):
                state = []
                total_P_F, total_P_B = 0.0, 0.0
                P_F_s, P_B_s = model(circuit_to_tensor(state))

                for t in range(level):
                    cat = Categorical(logits=P_F_s)
                    action = cat.sample()
                    new_state = state + [sorted_keys[action]]
                    total_P_F += cat.log_prob(action)

                    if t == level - 1:
                        reward = fidelity_reward(new_state, target)
                        best_fid = max(best_fid, reward)
                    P_F_s, P_B_s = model(circuit_to_tensor(new_state))
                    total_P_B += Categorical(logits=P_B_s).log_prob(action)
                    state = new_state

                loss = (model.logZ + total_P_F - torch.log(torch.tensor(reward).clamp(min=1e-6)) - total_P_B).pow(2)
                minibatch_loss += loss

                if episode % update_freq == 0:
                    minibatch_loss.backward()
                    opt.step()
                    opt.zero_grad()
                    minibatch_loss = 0.0

            print(f" Level {level} tried → best_fid = {best_fid:.4f}")
            if(len(len_fidelity) > level - 3):
                len_fidelity[level-3] = best_fid
            else: len_fidelity.append(best_fid)

            # threshold 넘었으면 다음 level 로 이동
            if best_fid >= target_threshold:
                print(f" ▶ Level {level} clear!")
                last_level_best_fid = best_fid
                break
            elif count > 5:
                print(f" ▶ Level {level} pass.")
                count = 0
                break
            else:
                print(f" ✗ ({best_fid:.4f} < {target_threshold:.4f}) → Level {level} retry.")
                count += 1

    print("Curriculum Ended!")
    plt.plot(len_fidelity)
    plt.xlabel('Length')
    plt.ylabel('Fidelity')
    plt.title('3-qubit matrix')
    plt.show()

if __name__ == "__main__":
    main_tb()