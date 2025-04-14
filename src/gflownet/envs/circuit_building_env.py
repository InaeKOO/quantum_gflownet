import copy
import enum
import json
import re
from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Callable

import networkx as nx
import numpy as np
import torch
import torch_geometric.data as gd
from networkx.algorithms.isomorphism import is_isomorphic
from torch_scatter import scatter, scatter_max

from .graph_building_env import Graph, GraphAction, GraphActionType, GraphBuildingEnv

class QuantumGateType(enum.Enum):
    """Types of quantum gates that can be applied to qubits"""
    H = enum.auto()  # Hadamard
    X = enum.auto()  # Pauli X
    Y = enum.auto()  # Pauli Y
    Z = enum.auto()  # Pauli Z
    CNOT = enum.auto()  # Controlled NOT
    SWAP = enum.auto()  # SWAP
    T = enum.auto()  # T gate
    S = enum.auto()  # S gate
    RX = enum.auto()  # Rotation around X axis
    RY = enum.auto()  # Rotation around Y axis
    RZ = enum.auto()  # Rotation around Z axis

class QuantumCircuit(Graph):
    """A specialized graph for representing quantum circuits.
    Nodes represent qubits and edges represent quantum gates.
    """
    def __init__(self):
        super().__init__()
        self.gate_count = 0  # Track number of gates for unique IDs
        self.qubit_count = 0  # Track number of qubits

    def add_qubit(self, qubit_id: int = None) -> int:
        """Add a qubit to the circuit.
        
        Args:
            qubit_id: Optional ID for the qubit. If None, will use next available ID.
            
        Returns:
            The ID of the added qubit.
        """
        if qubit_id is None:
            qubit_id = self.qubit_count
        self.add_node(qubit_id, type="qubit")
        self.qubit_count = max(self.qubit_count, qubit_id + 1)
        return qubit_id

    def add_gate(self, gate_type: QuantumGateType, qubits: List[int], params: Dict[str, Any] = None) -> int:
        """Add a quantum gate to the circuit.
        
        Args:
            gate_type: Type of quantum gate to add
            qubits: List of qubit IDs the gate acts on
            params: Optional parameters for the gate (e.g. rotation angles)
            
        Returns:
            The ID of the added gate.
        """
        gate_id = self.gate_count
        self.gate_count += 1
        
        # Add edges from gate to qubits
        for qubit in qubits:
            self.add_edge(gate_id, qubit, type="gate", gate_type=gate_type, params=params or {})
            
        return gate_id

    def get_gate_qubits(self, gate_id: int) -> List[int]:
        """Get the qubits a gate acts on."""
        return [n for n in self.neighbors(gate_id)]

    def get_qubit_gates(self, qubit_id: int) -> List[int]:
        """Get the gates acting on a qubit."""
        return [n for n in self.neighbors(qubit_id)]

class CircuitBuildingEnv(GraphBuildingEnv):
    """Environment for building quantum circuits using graph-based actions."""
    
    def __init__(self, max_qubits: int = 10, max_gates: int = 100):
        """Initialize the quantum circuit building environment.
        
        Args:
            max_qubits: Maximum number of qubits allowed in the circuit
            max_gates: Maximum number of gates allowed in the circuit
        """
        super().__init__()
        self.max_qubits = max_qubits
        self.max_gates = max_gates
        
        # Required attributes for the transformer model
        self.num_node_dim = 1  # Dimension of node features (qubit index)
        self.num_edge_dim = 2  # Dimension of edge features (gate type and parameters)
        self.num_cond_dim = 1  # Dimension of conditioning information
        self.num_new_node_values = 1  # Number of possible new node values
        self.num_node_attr_logits = 1  # Number of node attribute logits
        self.num_edge_attr_logits = 2  # Number of edge attribute logits
        self.num_node_attrs = 1  # Number of node attributes
        self.num_edge_attrs = 2  # Number of edge attributes
        self.edges_are_duplicated = False  # Whether edges are duplicated in the graph
        self.edges_are_unordered = True  # Whether edge order matters
        self.action_type_order = [
            GraphActionType.Stop,
            GraphActionType.AddQubit,
            GraphActionType.AddGate,
            GraphActionType.SetGateParam,
            GraphActionType.RemoveGate,
            GraphActionType.RemoveQubit
        ]
        self.bck_action_type_order = [
            GraphActionType.RemoveGate,
            GraphActionType.RemoveQubit
        ]

    def new(self) -> QuantumCircuit:
        """Create a new empty quantum circuit."""
        return QuantumCircuit()

    def step(self, g: QuantumCircuit, action: GraphAction) -> QuantumCircuit:
        """Apply an action to the quantum circuit.
        
        Args:
            g: Current quantum circuit
            action: Action to apply
            
        Returns:
            Modified quantum circuit
        """
        gp = g.copy()
        
        if action.action is GraphActionType.AddQubit:
            if g.qubit_count >= self.max_qubits:
                raise ValueError(f"Cannot add more than {self.max_qubits} qubits")
            gp.add_qubit(action.source)
            
        elif action.action is GraphActionType.AddGate:
            if g.gate_count >= self.max_gates:
                raise ValueError(f"Cannot add more than {self.max_gates} gates")
            gate_type = QuantumGateType(action.value)
            qubits = [action.source]
            if action.target is not None:
                qubits.append(action.target)
            gp.add_gate(gate_type, qubits, action.attr)
            
        elif action.action is GraphActionType.SetGateParam:
            if not g.has_edge(action.source, action.target):
                raise ValueError(f"No gate between qubits {action.source} and {action.target}")
            gp.edges[(action.source, action.target)]["params"][action.attr] = action.value
            
        elif action.action is GraphActionType.RemoveGate:
            if not g.has_edge(action.source, action.target):
                raise ValueError(f"No gate between qubits {action.source} and {action.target}")
            gp.remove_edge(action.source, action.target)
            
        elif action.action is GraphActionType.RemoveQubit:
            if not g.has_node(action.source):
                raise ValueError(f"No qubit with ID {action.source}")
            # Remove all gates connected to this qubit
            for gate in g.get_qubit_gates(action.source):
                gp.remove_edge(gate, action.source)
            gp.remove_node(action.source)
            
        else:
            # Fall back to base class actions
            return super().step(g, action)
            
        return gp

    def parents(self, g: QuantumCircuit) -> List[Tuple[GraphAction, QuantumCircuit]]:
        """Get possible parent states and actions that lead to the current state.
        
        Args:
            g: Current quantum circuit
            
        Returns:
            List of (action, parent circuit) pairs
        """
        parents = []
        
        # Handle qubit removal
        for qubit in g.nodes:
            if g.nodes[qubit]["type"] == "qubit":
                new_g = g.copy()
                new_g.remove_node(qubit)
                parents.append((GraphAction(GraphActionType.RemoveQubit, source=qubit), new_g))
                
        # Handle gate removal
        for gate in g.nodes:
            if g.nodes[gate]["type"] == "gate":
                qubits = g.get_gate_qubits(gate)
                new_g = g.copy()
                new_g.remove_node(gate)
                parents.append((GraphAction(GraphActionType.RemoveGate, source=gate), new_g))
                
        # Handle gate parameter changes
        for u, v, data in g.edges(data=True):
            if data["type"] == "gate":
                for param, value in data["params"].items():
                    new_g = g.copy()
                    new_g.edges[(u, v)]["params"][param] = None
                    parents.append((GraphAction(GraphActionType.SetGateParam, source=u, target=v, 
                                             attr=param, value=value), new_g))
                    
        return parents

    def is_sane(self, g: QuantumCircuit) -> bool:
        """Check if a quantum circuit is valid.
        
        Args:
            g: Quantum circuit to check
            
        Returns:
            True if the circuit is valid
        """
        # Check qubit count
        if g.qubit_count > self.max_qubits:
            return False
            
        # Check gate count
        if g.gate_count > self.max_gates:
            return False
            
        # Check all gates have valid qubit connections
        for gate in g.nodes:
            if g.nodes[gate]["type"] == "gate":
                qubits = g.get_gate_qubits(gate)
                if not qubits:
                    return False
                    
        return True 

    def has_n(self, g: Optional[QuantumCircuit] = None, n: Optional[int] = None) -> Union[bool, Callable[[QuantumCircuit, int], bool]]:
        """Check if a graph has a specific node.
        
        Args:
            g: The quantum circuit graph (optional)
            n: The node ID to check (optional)
            
        Returns:
            If both g and n are provided, returns True if the node exists in the graph, False otherwise.
            If either g or n is not provided, returns a callable that takes (g, n) as arguments.
        """
        if g is not None and n is not None:
            return n in g.nodes
        return lambda g, n: n in g.nodes

class GraphActionType(enum.Enum):
    # Forward actions
    Stop = enum.auto()
    AddNode = enum.auto()
    AddEdge = enum.auto()
    SetNodeAttr = enum.auto()
    SetEdgeAttr = enum.auto()
    # Backward actions
    RemoveNode = enum.auto()
    RemoveEdge = enum.auto()
    RemoveNodeAttr = enum.auto()
    RemoveEdgeAttr = enum.auto()
    # Quantum circuit actions
    AddQubit = enum.auto()
    AddGate = enum.auto()
    SetGateParam = enum.auto()
    RemoveGate = enum.auto()
    RemoveQubit = enum.auto()

    @cached_property
    def cname(self):
        return self.name

    @cached_property
    def mask_name(self):
        return self.name

    @cached_property
    def is_backward(self):
        return self.name.startswith("Remove")

class ActionIndex(NamedTuple):
    """Represents an action index in the categorical distribution.
    
    Attributes:
        action_idx: Index of the action type in the distribution
        row_idx: Row index for the action (e.g. source node for an edge)
        col_idx: Column index for the action (e.g. target node for an edge)
    """
    action_idx: int
    row_idx: int
    col_idx: int

class GraphActionCategorical:
    """Represents a categorical distribution over graph actions."""
    def __init__(
        self,
        graphs: gd.Batch,
        raw_logits: List[torch.Tensor],
        keys: List[Union[str, None]],
        types: List[GraphActionType],
        deduplicate_edge_index=True,
        action_masks: List[torch.Tensor] = None,
        slice_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
        self.graphs = graphs
        self.raw_logits = raw_logits
        self.keys = keys
        self.types = types
        self.deduplicate_edge_index = deduplicate_edge_index
        self.action_masks = action_masks
        self.slice_dict = slice_dict

    @property
    def logits(self):
        return self.raw_logits

    @logits.setter
    def logits(self, new_raw_logits):
        self.raw_logits = new_raw_logits

    @property
    def action_masks(self):
        return self._action_masks

    @action_masks.setter
    def action_masks(self, new_action_masks):
        self._action_masks = new_action_masks
        self._apply_action_masks()

    def _apply_action_masks(self):
        if self._action_masks is not None:
            for i, mask in enumerate(self._action_masks):
                if mask is not None:
                    self.raw_logits[i] = self._mask(self.raw_logits[i], mask)

    def _mask(self, x, m):
        return x.masked_fill(~m, -1e9)

    def detach(self):
        return GraphActionCategorical(
            self.graphs,
            [i.detach() for i in self.raw_logits],
            self.keys,
            self.types,
            self.deduplicate_edge_index,
            self.action_masks,
            self.slice_dict,
        )

    def to(self, device):
        return GraphActionCategorical(
            self.graphs.to(device),
            [i.to(device) for i in self.raw_logits],
            self.keys,
            self.types,
            self.deduplicate_edge_index,
            [i.to(device) if i is not None else None for i in self.action_masks],
            {k: v.to(device) for k, v in self.slice_dict.items()} if self.slice_dict is not None else None,
        )

    def log_n_actions(self):
        """Compute the log number of actions for each graph in the batch."""
        n_actions = []
        for i, t in enumerate(self.types):
            if t is GraphActionType.Stop:
                n_actions.append(torch.ones(self.graphs.num_graphs, device=self.raw_logits[i].device))
            else:
                key = self.keys[i]
                if key is None:
                    n_actions.append(torch.ones(self.graphs.num_graphs, device=self.raw_logits[i].device))
                else:
                    if key == "non_edge_index":
                        n_actions.append(
                            scatter(
                                torch.ones(self.graphs[key].shape[1], device=self.raw_logits[i].device),
                                self.graphs.batch[self.graphs[key][0]],
                                dim=0,
                                reduce="sum",
                            )
                        )
                    else:
                        n_actions.append(
                            scatter(
                                torch.ones(self.graphs[key].shape[0], device=self.raw_logits[i].device),
                                self.graphs.batch,
                                dim=0,
                                reduce="sum",
                            )
                        )
        return torch.stack(n_actions, 1).log()

    def _compute_batchwise_max(
        self,
        x: List[torch.Tensor],
        detach: bool = True,
        batch: Optional[List[torch.Tensor]] = None,
        reduce_columns: bool = True,
    ):
        """Compute the maximum value for each graph in the batch."""
        if batch is None:
            batch = [None] * len(x)
        maxes = []
        for i, (t, xi, b) in enumerate(zip(self.types, x, batch)):
            if t is GraphActionType.Stop:
                maxes.append(xi)
            else:
                key = self.keys[i]
                if key is None:
                    maxes.append(xi)
                else:
                    if key == "non_edge_index":
                        if b is None:
                            b = self.graphs.batch[self.graphs[key][0]]
                        maxes.append(scatter_max(xi, b, dim=0)[0])
                    else:
                        if b is None:
                            b = self.graphs.batch
                        maxes.append(scatter_max(xi, b, dim=0)[0])
        if reduce_columns:
            return torch.stack(maxes, 1).max(1)[0]
        return torch.stack(maxes, 1)

    def logsoftmax(self):
        """Compute the log softmax of the logits."""
        maxes = self._compute_batchwise_max(self.raw_logits, detach=True)
        return [xi - maxes.unsqueeze(1) for xi in self.raw_logits]

    def logsumexp(self, x=None):
        """Compute the log sum exp of the logits."""
        if x is None:
            x = self.raw_logits
        maxes = self._compute_batchwise_max(x, detach=True)
        return torch.logsumexp(torch.stack([xi - maxes.unsqueeze(1) for xi in x], 0), 0) + maxes

    def sample(self) -> List[ActionIndex]:
        """Sample actions from the categorical distribution."""
        logits = self.logsoftmax()
        actions = []
        for i, (t, xi) in enumerate(zip(self.types, logits)):
            if t is GraphActionType.Stop:
                actions.append(ActionIndex(i, 0, 0))
            else:
                key = self.keys[i]
                if key is None:
                    actions.append(ActionIndex(i, 0, 0))
                else:
                    if key == "non_edge_index":
                        idx = torch.multinomial(torch.exp(xi), 1).squeeze()
                        actions.append(ActionIndex(i, self.graphs[key][0][idx], self.graphs[key][1][idx]))
                    else:
                        idx = torch.multinomial(torch.exp(xi), 1).squeeze()
                        actions.append(ActionIndex(i, idx, 0))
        return actions

    def argmax(
        self,
        x: List[torch.Tensor],
        batch: List[torch.Tensor] = None,
        dim_size: int = None,
    ) -> List[ActionIndex]:
        """Compute the argmax of the logits."""
        if batch is None:
            batch = [None] * len(x)
        actions = []
        for i, (t, xi, b) in enumerate(zip(self.types, x, batch)):
            if t is GraphActionType.Stop:
                actions.append(ActionIndex(i, 0, 0))
            else:
                key = self.keys[i]
                if key is None:
                    actions.append(ActionIndex(i, 0, 0))
                else:
                    if key == "non_edge_index":
                        if b is None:
                            b = self.graphs.batch[self.graphs[key][0]]
                        if dim_size is None:
                            dim_size = self.graphs.num_graphs
                        idx = scatter_max(xi, b, dim=0, dim_size=dim_size)[1]
                        actions.append(ActionIndex(i, self.graphs[key][0][idx], self.graphs[key][1][idx]))
                    else:
                        if b is None:
                            b = self.graphs.batch
                        if dim_size is None:
                            dim_size = self.graphs.num_graphs
                        idx = scatter_max(xi, b, dim=0, dim_size=dim_size)[1]
                        actions.append(ActionIndex(i, idx, 0))
        return actions

    def log_prob(self, actions: List[ActionIndex], logprobs: torch.Tensor = None, batch: torch.Tensor = None):
        """Compute the log probability of the actions."""
        if logprobs is None:
            logprobs = self.logsoftmax()
        if batch is None:
            batch = self.graphs.batch
        probs = []
        for i, (t, a) in enumerate(zip(self.types, actions)):
            if t is GraphActionType.Stop:
                probs.append(logprobs[i][0])
            else:
                key = self.keys[i]
                if key is None:
                    probs.append(logprobs[i][0])
                else:
                    if key == "non_edge_index":
                        idx = (self.graphs[key][0] == a.row_idx) & (self.graphs[key][1] == a.col_idx)
                        probs.append(logprobs[i][idx])
                    else:
                        probs.append(logprobs[i][a.row_idx])
        return torch.stack(probs, 1).sum(1)

    def entropy(self, logprobs=None):
        """Compute the entropy of the categorical distribution."""
        if logprobs is None:
            logprobs = self.logsoftmax()
        return -torch.sum(torch.exp(logprobs) * logprobs, 1)


def action_type_to_mask(t: GraphActionType, gbatch: gd.Batch, assert_mask_exists: bool = False):
    """Convert an action type to a mask for the batch of graphs."""
    if t is GraphActionType.Stop:
        return torch.ones(gbatch.num_graphs, 1, device=gbatch.x.device)
    elif t is GraphActionType.AddNode:
        return torch.ones(gbatch.num_graphs, 1, device=gbatch.x.device)
    elif t is GraphActionType.AddEdge:
        return torch.ones(gbatch.non_edge_index.shape[1], device=gbatch.x.device)
    elif t is GraphActionType.SetNodeAttr:
        return torch.ones(gbatch.x.shape[0], device=gbatch.x.device)
    elif t is GraphActionType.SetEdgeAttr:
        return torch.ones(gbatch.edge_index.shape[1], device=gbatch.x.device)
    elif t is GraphActionType.RemoveNode:
        return torch.ones(gbatch.x.shape[0], device=gbatch.x.device)
    elif t is GraphActionType.RemoveEdge:
        return torch.ones(gbatch.edge_index.shape[1], device=gbatch.x.device)
    elif t is GraphActionType.RemoveNodeAttr:
        return torch.ones(gbatch.x.shape[0], device=gbatch.x.device)
    elif t is GraphActionType.RemoveEdgeAttr:
        return torch.ones(gbatch.edge_index.shape[1], device=gbatch.x.device)
    elif t is GraphActionType.AddQubit:
        return torch.ones(gbatch.num_graphs, 1, device=gbatch.x.device)
    elif t is GraphActionType.AddGate:
        return torch.ones(gbatch.non_edge_index.shape[1], device=gbatch.x.device)
    elif t is GraphActionType.SetGateParam:
        return torch.ones(gbatch.edge_index.shape[1], device=gbatch.x.device)
    elif t is GraphActionType.RemoveGate:
        return torch.ones(gbatch.edge_index.shape[1], device=gbatch.x.device)
    elif t is GraphActionType.RemoveQubit:
        return torch.ones(gbatch.x.shape[0], device=gbatch.x.device)
    else:
        if assert_mask_exists:
            raise ValueError(f"Unknown action type {t}")
        return None 