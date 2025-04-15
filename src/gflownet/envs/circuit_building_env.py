from copy import deepcopy
from typing import Any, List, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data


from gflownet.envs.graph_building_env import (
    ActionIndex,
    Graph,
    GraphAction,
    GraphActionType,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)

class QuantumCircuit(Graph):
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        # Each qubit is represented as a sequence of gates
        self.qubit_sequences: List[List[Any]] = [[] for _ in range(num_qubits)]

    def __repr__(self):
        return "\n".join(f"Qubit {i}: {seq}" for i, seq in enumerate(self.qubit_sequences))

    @property
    def nodes(self):
        return self.qubit_sequences

class CircuitBuildingEnv(GraphBuildingEnv):
    """This class extends GraphBuildingEnv to generate quantum circuits as sequences of gates on multiple qubits."""

    def __init__(self, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits

    def new(self):
        return QuantumCircuit(self.num_qubits)

    def step(self, g: Graph, a: GraphAction):
        c: QuantumCircuit = deepcopy(g)  # type: ignore
        if a.action == GraphActionType.AddNode:
            # The value contains both the gate type and the qubit index
            qubit_idx = a.value
            gate_type = a.source
            c.qubit_sequences[qubit_idx].append(gate_type)
        return c

    def count_backward_transitions(self, g: Graph, check_idempotent: bool = False):
        # Count the total number of gates across all qubits
        c: QuantumCircuit = g  # type: ignore
        return sum(len(seq) for seq in c.qubit_sequences)

    def parents(self, g: Graph):
        c: QuantumCircuit = deepcopy(g)  # type: ignore
        parents = []
        for qubit_idx, seq in enumerate(c.qubit_sequences):
            if not len(seq):
                continue
            gate = seq.pop()
            parents.append((GraphAction(GraphActionType.AddNode, value=(gate, qubit_idx)), c))
        return parents

    def reverse(self, g: Graph, ga: GraphAction):
        return GraphAction(GraphActionType.Stop)

class CircuitBatch:
    def __init__(self, circuits: List[torch.Tensor], pad: int):
        self.circuits = circuits
        # Each circuit is a tensor of shape (num_qubits, max_seq_len)
        self.x = pad_sequence(circuits, batch_first=False, padding_value=pad)
        self.mask = self.x.eq(pad).T
        self.lens = torch.tensor([len(i) for i in circuits], dtype=torch.long)
        self.logit_idx = self.x.ne(pad).T.flatten().nonzero().flatten()
        self.num_graphs = self.lens.sum().item()

    def to(self, device):
        for name in dir(self):
            x = getattr(self, name)
            if isinstance(x, torch.Tensor):
                setattr(self, name, x.to(device))
        return self

class AutoregressiveCircuitBuildingContext(GraphBuildingEnvContext):
    """This context generates quantum circuits by adding gates to qubits in an autoregressive fashion."""

    def __init__(self, gates: Sequence[str], num_qubits: int, num_cond_dim=0):
        self.gates = gates
        self.num_qubits = num_qubits
        self.action_type_order = [GraphActionType.Stop, GraphActionType.AddNode]

        # Each gate can be applied to any qubit
        self.num_tokens = len(gates) + 2  # Gates + BOS + PAD
        self.bos_token = len(gates)
        self.pad_token = len(gates) + 1
        self.num_actions = len(gates) * num_qubits + 1  # (Gates × Qubits) + Stop
        self.num_cond_dim = num_cond_dim

    def ActionIndex_to_GraphAction(self, g: Data, aidx: ActionIndex, fwd: bool = True) -> GraphAction:
        t = self.action_type_order[aidx.action_type]
        if t is GraphActionType.Stop:
            return GraphAction(t)
        elif t is GraphActionType.AddNode:
            # Convert the action index to (gate_type, qubit_idx)
            gate_idx = aidx.col_idx % len(self.gates)
            qubit_idx = aidx.col_idx // len(self.gates)
            return GraphAction(t, value=(gate_idx, qubit_idx))
        raise ValueError(aidx)

    def GraphAction_to_ActionIndex(self, g: Data, action: GraphAction) -> ActionIndex:
        if action.action is GraphActionType.Stop:
            col = 0
            type_idx = self.action_type_order.index(action.action)
        elif action.action is GraphActionType.AddNode:
            qubit_idx = action.value
            gate_idx = action.source
            col = qubit_idx * len(self.gates) + gate_idx
            type_idx = self.action_type_order.index(action.action)
        else:
            raise ValueError(action)
        return ActionIndex(action_type=type_idx, row_idx=0, col_idx=int(col))

    def graph_to_Data(self, g: Graph):
        c: QuantumCircuit = g  # type: ignore
        # Convert each qubit sequence to a tensor and stack them
        sequences = []
        for seq in c.qubit_sequences:
            sequences.append(torch.tensor([self.bos_token] + seq, dtype=torch.long))
        return torch.stack(sequences)

    def collate(self, graphs: List[Data]):
        return CircuitBatch(graphs, pad=self.pad_token)

    def is_sane(self, g: Graph) -> bool:
        return True

    def graph_to_obj(self, g: Graph):
        c: QuantumCircuit = g  # type: ignore
        return "\n".join(f"Qubit {i}: {' '.join(self.gates[int(g)] for g in seq)}" 
                        for i, seq in enumerate(c.qubit_sequences))

    def object_to_log_repr(self, g: Graph):
        return self.graph_to_obj(g)