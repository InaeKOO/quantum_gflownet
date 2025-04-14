from itertools import chain
from typing import Dict, Optional, Union, Callable

import torch
import torch.nn as nn
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.utils import add_self_loops

from gflownet.config import Config
from gflownet.envs.circuit_building_env import (
    CircuitBuildingEnv,
    GraphActionCategorical,
    GraphActionType,
    action_type_to_mask,
    QuantumGateType
)


def mlp(n_in, n_hid, n_out, n_layer, act=nn.LeakyReLU):
    """Creates a fully-connected network with no activation after the last layer."""
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(*sum([[nn.Linear(n[i], n[i + 1]), act()] for i in range(n_layer + 1)], [])[:-1])


class CircuitTransformer(nn.Module):
    """A transformer model specialized for quantum circuits.
    
    This model takes in:
    - Node features (qubits)
    - Edge features (quantum gates)
    - Graph features (circuit-level information)
    """

    def __init__(
        self, x_dim, e_dim, g_dim, num_emb=64, num_layers=3, num_heads=2, num_noise=0, ln_type="pre", concat=True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_noise = num_noise
        assert ln_type in ["pre", "post"]
        self.ln_type = ln_type

        # Input feature processing
        self.x2h = mlp(x_dim + num_noise, num_emb, num_emb, 2)
        self.e2h = mlp(e_dim, num_emb, num_emb, 2)
        self.c2h = mlp(max(1, g_dim), num_emb, num_emb, 2)
        
        # Transformer layers
        n_att = num_emb * num_heads if concat else num_emb
        self.graph2emb = nn.ModuleList(
            sum(
                [
                    [
                        gnn.GENConv(num_emb, num_emb, num_layers=1, aggr="add", norm=None),
                        gnn.TransformerConv(num_emb * 2, n_att // num_heads, edge_dim=num_emb, heads=num_heads),
                        nn.Linear(n_att, num_emb),
                        gnn.LayerNorm(num_emb, affine=False),
                        mlp(num_emb, num_emb * 4, num_emb, 1),
                        gnn.LayerNorm(num_emb, affine=False),
                        nn.Linear(num_emb, num_emb * 2),
                    ]
                    for i in range(self.num_layers)
                ],
                [],
            )
        )

    def forward(self, g: gd.Batch, cond: Optional[torch.Tensor]):
        """Forward pass for quantum circuit processing.
        
        Args:
            g: Batch of quantum circuits
            cond: Optional conditioning information
            
        Returns:
            node_embeddings: Per-qubit embeddings
            graph_embeddings: Per-circuit embeddings
        """
        if self.num_noise > 0:
            x = torch.cat([g.x, torch.rand(g.x.shape[0], self.num_noise, device=g.x.device)], 1)
        else:
            x = g.x
            
        # Process input features
        o = self.x2h(x)
        e = self.e2h(g.edge_attr)
        c = self.c2h(cond if cond is not None else torch.ones((g.num_graphs, 1), device=g.x.device))
        
        # Add conditioning nodes
        num_total_nodes = g.x.shape[0]
        u, v = torch.arange(num_total_nodes, device=o.device), g.batch + num_total_nodes
        aug_edge_index = torch.cat([g.edge_index, torch.stack([u, v]), torch.stack([v, u])], 1)
        
        # Add edge features for conditioning edges
        e_p = torch.zeros((num_total_nodes * 2, e.shape[1]), device=g.x.device)
        e_p[:, 0] = 1  # Bias term
        aug_e = torch.cat([e, e_p], 0)
        
        # Add self loops
        aug_edge_index, aug_e = add_self_loops(aug_edge_index, aug_e, "mean")
        aug_batch = torch.cat([g.batch, torch.arange(c.shape[0], device=o.device)], 0)
        
        # Process through transformer layers
        o = torch.cat([o, c], 0)
        for i in range(self.num_layers):
            gen, trans, linear, norm1, ff, norm2, cscale = self.graph2emb[i * 7 : (i + 1) * 7]
            cs = cscale(c[aug_batch])
            
            if self.ln_type == "post":
                agg = gen(o, aug_edge_index, aug_e)
                l_h = linear(trans(torch.cat([o, agg], 1), aug_edge_index, aug_e))
                scale, shift = cs[:, : l_h.shape[1]], cs[:, l_h.shape[1] :]
                o = norm1(o + l_h * scale + shift, aug_batch)
                o = norm2(o + ff(o), aug_batch)
            else:
                o_norm = norm1(o, aug_batch)
                agg = gen(o_norm, aug_edge_index, aug_e)
                l_h = linear(trans(torch.cat([o_norm, agg], 1), aug_edge_index, aug_e))
                scale, shift = cs[:, : l_h.shape[1]], cs[:, l_h.shape[1] :]
                o = o + l_h * scale + shift
                o = o + ff(norm2(o, aug_batch))

        # Split node and graph embeddings
        o_final = o[: -c.shape[0]]
        glob = torch.cat([gnn.global_mean_pool(o_final, g.batch), o[-c.shape[0] :]], 1)
        return o_final, glob


class CircuitTransformerGFN(nn.Module):
    """CircuitTransformer class for a GFlowNet which outputs a GraphActionCategorical.
    
    Specialized for quantum circuit actions.
    """

    # Map action types to graph parts
    _action_type_to_graph_part = {
        GraphActionType.Stop: "graph",
        GraphActionType.AddQubit: "node",
        GraphActionType.AddGate: "non_edge",
        GraphActionType.SetGateParam: "edge",
        GraphActionType.RemoveGate: "edge",
        GraphActionType.RemoveQubit: "node",
    }

    # Map graph parts to batch keys
    _graph_part_to_key = {
        "graph": None,
        "node": "x",
        "non_edge": "non_edge_index",
        "edge": "edge_index",
    }

    action_type_to_key = lambda action_type: CircuitTransformerGFN._graph_part_to_key.get(
        CircuitTransformerGFN._action_type_to_graph_part.get(action_type)
    )

    def __init__(
        self,
        env_ctx: CircuitBuildingEnv,
        cfg: Config,
        num_graph_out=1,
        do_bck=False,
    ):
        super().__init__()
        self.transf = CircuitTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=cfg.model.num_emb,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
            concat=cfg.model.graph_transformer.concat_heads,
        )
        
        self.env_ctx = env_ctx
        num_emb = cfg.model.num_emb
        num_final = num_emb
        num_glob_final = num_emb * 2
        num_edge_feat = num_emb if env_ctx.edges_are_unordered else num_emb * 2
        
        self.edges_are_duplicated = env_ctx.edges_are_duplicated
        self.edges_are_unordered = env_ctx.edges_are_unordered
        self.action_type_order = env_ctx.action_type_order

        # Define input/output dimensions for each action type
        self._action_type_to_num_inputs_outputs = {
            GraphActionType.Stop: (num_glob_final, 1),  # Stop action uses global features
            GraphActionType.AddQubit: (num_final, 1),
            GraphActionType.AddGate: (num_edge_feat, len(QuantumGateType)),
            GraphActionType.SetGateParam: (num_edge_feat, 1),  # For rotation angles
            GraphActionType.RemoveGate: (num_edge_feat, 1),
            GraphActionType.RemoveQubit: (num_final, 1),
            GraphActionType.AddNode: (num_final, 1),  # Base graph actions
            GraphActionType.AddEdge: (num_edge_feat, 1),
            GraphActionType.SetNodeAttr: (num_final, 1),
            GraphActionType.SetEdgeAttr: (num_edge_feat, 1),
            GraphActionType.RemoveNode: (num_final, 1),
            GraphActionType.RemoveEdge: (num_edge_feat, 1),
            GraphActionType.RemoveNodeAttr: (num_final, 1),
            GraphActionType.RemoveEdgeAttr: (num_edge_feat, 1),
        }

        self._action_type_to_key = {
            at: self._graph_part_to_key[self._action_type_to_graph_part[at]] 
            for at in self._action_type_to_graph_part
        }

        # Create MLPs for each action type
        mlps = {}
        for atype in chain(env_ctx.action_type_order, env_ctx.bck_action_type_order if do_bck else []):
            num_in, num_out = self._action_type_to_num_inputs_outputs[atype]
            mlps[atype.cname] = mlp(num_in, num_emb, num_out, cfg.model.num_layers)
        self.mlps = nn.ModuleDict(mlps)

        self.do_bck = do_bck
        if do_bck:
            self.bck_action_type_order = env_ctx.bck_action_type_order

        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, cfg.model.num_layers)
        self._logZ = mlp(max(1, env_ctx.num_cond_dim), num_emb * 2, 1, 2)

    def logZ(self, cond_info: Optional[torch.Tensor]):
        if cond_info is None:
            return self._logZ(torch.ones((1, 1), device=self._logZ[0].weight.device))
        return self._logZ(cond_info)

    def _make_cat(self, g: gd.Batch, emb: Dict[str, Tensor], action_types: list[GraphActionType]):
        # Handle the case where there are no nodes in the graph
        if not hasattr(g, 'x'):
            g.x = torch.zeros((0, 1), device=emb['graph'].device)
            
        return GraphActionCategorical(
            g,
            raw_logits=[self.mlps[t.cname](emb[self._action_type_to_graph_part[t]]) for t in action_types],
            keys=[self._action_type_to_key[t] for t in action_types],
            action_masks=[action_type_to_mask(t, g) for t in action_types],
            types=action_types,
        )

    def forward(self, g: gd.Batch, cond: Optional[torch.Tensor]):
        # Get embeddings
        node_embeddings, graph_embeddings = self.transf(g, cond)
        
        # Process non-edges (potential new gates)
        if hasattr(g, "non_edge_index"):
            ne_row, ne_col = g.non_edge_index
            if self.edges_are_unordered:
                non_edge_embeddings = node_embeddings[ne_row] + node_embeddings[ne_col]
            else:
                non_edge_embeddings = torch.cat([node_embeddings[ne_row], node_embeddings[ne_col]], 1)
        else:
            non_edge_embeddings = None
            
        # Process existing edges (gates)
        if self.edges_are_duplicated:
            e_row, e_col = g.edge_index[:, ::2]
        else:
            e_row, e_col = g.edge_index
            
        if self.edges_are_unordered:
            edge_embeddings = node_embeddings[e_row] + node_embeddings[e_col]
        else:
            edge_embeddings = torch.cat([node_embeddings[e_row], node_embeddings[e_col]], 1)

        # Prepare embeddings for action prediction
        emb = {
            "graph": graph_embeddings,
            "node": node_embeddings,
            "edge": edge_embeddings,
            "non_edge": non_edge_embeddings,
        }

        # Get graph-level output
        graph_out = self.emb2graph_out(graph_embeddings)
        
        # Get forward and backward action distributions
        fwd_cat = self._make_cat(g, emb, self.action_type_order)
        if self.do_bck:
            bck_cat = self._make_cat(g, emb, self.bck_action_type_order)
            return fwd_cat, bck_cat, graph_out
        return fwd_cat, graph_out 