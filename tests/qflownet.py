import matplotlib.pyplot as pp
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tqdm

gates = {
    'H': torch.tensor([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]], dtype=torch.complex128),
    'X': torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    'Y': torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    'Z': torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
    'T': torch.tensor([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=torch.complex128),
    'S': torch.tensor([[1, 0], [0, np.exp(1j * np.pi / 2)]], dtype=torch.complex128)
}

sorted_keys = sorted(gates.keys())

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

def fidelity_reward(circuit_str: str, target: torch.Tensor) -> float:
    current = torch.eye(2, dtype=target.dtype, device=target.device)
    for i in circuit_str:
        if i in gates:
            current = gates[i] @ current
        else:
            raise ValueError(f"Unknown gate: {i}")
    fid = torch.abs(torch.trace(current.conj().T @ target)) / 2
    return fid.item()

def circuit_to_tensor(circuit):
  return torch.tensor([i in circuit for i in sorted_keys]).float()

class FlowModel(nn.Module):
  def __init__(self, num_hid):
    super().__init__()
    # We encoded the current state as binary vector, for each patch the associated
    # dimension is either 0 or 1 depending on the absence or precense of that patch.
    # Therefore the input dimension is 6 for the 6 patches.
    self.mlp = nn.Sequential(nn.Linear(6, num_hid), nn.LeakyReLU(),
                             # We also output 6 numbers, since there are up to
                             # 6 possible actions (and thus child states), but we
                             # will mask those outputs for patches that are
                             # already there.
                             nn.Linear(num_hid, 6))
  def forward(self, x):
    # We take the exponential to get positive numbers, since flows must be positive,
    # and multiply by (1 - x) to give 0 flow to actions we know we can't take
    # (in this case, x[i] is 1 if a feature is already there, so we know we
    # can't add it again).
    F = self.mlp(x).exp() * (1 - x)
    return F
  
def circuit_parents(circuit):
   if not circuit:  # If circuit is empty, return empty lists
       return [], []
   parent_actions = []
   parent_states = []
   # Create parent state by removing the last gate
   parent_state = circuit[:-1]
   # Get the action (index of the last gate in sorted_keys)
   action = sorted_keys.index(circuit[-1])
   parent_states.append(parent_state)
   parent_actions.append(action)
   return parent_states, parent_actions

class TBModel(nn.Module):
  def __init__(self, num_hid):
    super().__init__()
    # The input dimension is 6 for the 6 patches.
    self.mlp = nn.Sequential(nn.Linear(6, num_hid), nn.LeakyReLU(),
                             # We now output 12 numbers, 6 for P_F and 6 for P_B
                             nn.Linear(num_hid, 12))
    # log Z is just a single number
    self.logZ = nn.Parameter(torch.ones(1))

  def forward(self, x):
    logits = self.mlp(x)
    # Slice the logits, and mask invalid actions (since we're predicting
    # log-values), we use -100 since exp(-100) is tiny, but we don't want -inf)
    P_F = logits[..., :6] * (1 - x) + x * -100
    P_B = logits[..., 6:] * x + (1 - x) * -100
    return P_F, P_B

def main():
        # Instantiate model and optimizer
    F_sa = FlowModel(512)
    opt = torch.optim.Adam(F_sa.parameters(), 3e-4)
    target = random_unitary(1)
    # Let's keep track of the losses and the faces we sample
    losses = []
    sampled_circuits = []
    # To not complicate the code, I'll just accumulate losses here and take a
    # gradient step every `update_freq` episode.
    minibatch_loss = 0
    update_freq = 4
    for episode in tqdm.tqdm(range(50000), ncols=40):
    # Each episode starts with an "empty state"
        state = []
        # Predict F(s, a)
        edge_flow_prediction = F_sa(circuit_to_tensor(state))
        for t in range(3):
            # The policy is just normalizing, and gives us the probability of each action
            policy = edge_flow_prediction / edge_flow_prediction.sum()
            # Sample the action
            action = Categorical(probs=policy).sample()
            # "Go" to the next state
            new_state = state + [sorted_keys[action]]

            # Now we want to compute the loss, we'll first enumerate the parents
            parent_states, parent_actions = circuit_parents(new_state)
            # And compute the edge flows F(s, a) of each parent
            px = torch.stack([circuit_to_tensor(p) for p in parent_states])
            pa = torch.tensor(parent_actions).long()
            parent_edge_flow_preds = F_sa(px)[torch.arange(len(parent_states)), pa]
            # Now we need to compute the reward and F(s, a) of the current state,
            # which is currently `new_state`
            if t == 2:
            # If we've built a complete face, we're done, so the reward is > 0
            # (unless the face is invalid)
                reward = fidelity_reward(new_state, target)
                # and since there are no children to this state F(s,a) = 0 \forall a
                edge_flow_prediction = torch.zeros(6)
            else:
                # Otherwise we keep going, and compute F(s, a)
                reward = 0
                edge_flow_prediction = F_sa(circuit_to_tensor(new_state))

            # The loss as per the equation above
            flow_mismatch = (parent_edge_flow_preds.sum() - edge_flow_prediction.sum() - reward).pow(2)
            minibatch_loss += flow_mismatch  # Accumulate
            # Continue iterating
            state = new_state

    # We're done with the episode, add the face to the list, and if we are at an
    # update episode, take a gradient step.
        sampled_circuits.append(state)
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
    pp.figure(figsize=(10,3))
    pp.plot(losses)
    pp.yscale('log')
    pp.show()

def main_tb():
   # Instantiate model and optimizer
    model = TBModel(512)
    opt = torch.optim.Adam(model.parameters(),  3e-4)
    target = random_unitary(1)
    # Let's keep track of the losses and the faces we sample
    tb_losses = []
    tb_sampled_faces = []
    # To not complicate the code, I'll just accumulate losses here and take a
    # gradient step every `update_freq` episode.
    minibatch_loss = 0
    update_freq = 2

    logZs = []
    for episode in tqdm.tqdm(range(50000), ncols=40):
    # Each episode starts with an "empty state"
        state = []
        # Predict P_F, P_B
        P_F_s, P_B_s = model(circuit_to_tensor(state))
        total_P_F = 0
        total_P_B = 0
        for t in range(3):
            # Here P_F is logits, so we want the Categorical to compute the softmax for us
            cat = Categorical(logits=P_F_s)
            action = cat.sample()
            # "Go" to the next state
            new_state = state + [sorted_keys[action]]
            # Accumulate the P_F sum
            total_P_F += cat.log_prob(action)

            if t == 2:
                # If we've built a complete face, we're done, so the reward is > 0
                # (unless the face is invalid)
                reward = torch.tensor(fidelity_reward(new_state, target)).float()
            # We recompute P_F and P_B for new_state
            P_F_s, P_B_s = model(circuit_to_tensor(new_state))
            # Here we accumulate P_B, going backwards from `new_state`. We're also just
            # going to use opposite semantics for the backward policy. I.e., for P_F action
            # `i` just added the face part `i`, for P_B we'll assume action `i` removes
            # face part `i`, this way we can just keep the same indices.
            total_P_B += Categorical(logits=P_B_s).log_prob(action)

            # Continue iterating
            state = new_state

        # We're done with the trajectory, let's compute its loss. Since the reward can
        # sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
        loss = (model.logZ + total_P_F - torch.log(reward).clip(-20) - total_P_B).pow(2)
        minibatch_loss += loss

        # Add the face to the list, and if we are at an
        # update episode, take a gradient step.
        tb_sampled_faces.append(state)
        if episode % update_freq == 0:
            tb_losses.append(minibatch_loss.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0
            logZs.append(model.logZ.item())
    f, ax = pp.subplots(2, 1, figsize=(10,6))
    pp.sca(ax[0])
    pp.plot(tb_losses)
    pp.yscale('log')
    pp.ylabel('loss')
    pp.sca(ax[1])
    pp.plot(np.exp(logZs))
    pp.ylabel('estimated Z')
    pp.show()
    print(model.logZ.exp())

if __name__ == "__main__":
    main_tb() #1-qubit problem