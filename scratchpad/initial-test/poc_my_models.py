import torch
import torch.nn as nn

# --- Imports from our project ---
from panq_functions import GNNDecoder # From the Astra codebase

# ==============================================================================
# === FAKE MAMBA BLOCK FOR MACOS DEVELOPMENT ===
# ==============================================================================

# class Mamba(nn.Module):
#     def __init__(self, d_model, d_state, d_conv, expand):
#         super().__init__()
#         self.d_model = d_model
#         # This layer ensures the output has the correct dimension.
#         self.dummy_layer = nn.Linear(d_model, d_model)

#     def forward(self, x):
#         # This placeholder just passes the data through a linear layer.
#         # The real Mamba would do complex sequence processing here.
#         return self.dummy_layer(x)

from mamba_ssm import Mamba2

# ==============================================================================

class MGHD(nn.Module):
    """
    Mamba-Graph Hybrid Decoder (MGHD).
    This model combines a Mamba SSM for processing a sequence of check nodes
    and a GNN for spatial message passing.
    """

    def __init__(self, gnn_params, mamba_params):
        super().__init__()
        
        # --- Part 1: Initial Feature Processing ---
        self.input_embedding = nn.Linear(gnn_params['n_node_inputs'], mamba_params['d_model'])
        
        # --- Part 2: The Sequence Processor (Mamba) ---
        self.mamba = Mamba2(
            d_model=mamba_params['d_model'],
            d_state=mamba_params['d_state'],
            d_conv=mamba_params['d_conv'],
            expand=mamba_params['expand']
        )
        
        # --- Part 3: The Spatial Processor (GNN) ---
        self.gnn = GNNDecoder(**gnn_params)
        
        # --- Part 4: The Interface Layer (FIXED) ---
        # Project from Mamba's output to GNN's expected n_node_inputs (not n_node_features)
        self.projection = nn.Linear(mamba_params['d_model'], gnn_params['n_node_inputs'])

    def forward(self, node_inputs, src_ids, dst_ids):
        """
        Defines the full data flow through the hybrid model.
        """
        num_check_nodes = self.gnn.dist**2 - 1
        num_qubit_nodes = self.gnn.dist**2
        nodes_per_graph = num_check_nodes + num_qubit_nodes
        batch_size = node_inputs.shape[0] // nodes_per_graph

        # --- Step A: Isolate and Reshape Data for Mamba ---
        indices = [i*nodes_per_graph + j for i in range(batch_size) for j in range(num_check_nodes)]
        check_node_inputs = node_inputs[indices]
        check_node_inputs = check_node_inputs.to(self.input_embedding.weight.dtype)
        embedded_check_inputs = self.input_embedding(check_node_inputs)
        mamba_input_sequence = embedded_check_inputs.view(batch_size, num_check_nodes, -1)

        # --- Step B: Sequential Processing ---
        mamba_output_sequence = self.mamba(mamba_input_sequence)
        temporal_features = mamba_output_sequence.reshape(-1, self.mamba.d_model)

        # --- Step C: Prepare Inputs for the GNN (FIXED) ---
        projected_check_features = self.projection(temporal_features)
        
        # Create tensor with correct dimensions (n_node_inputs, not n_node_features)
        gnn_input_features = torch.zeros(node_inputs.shape[0], self.gnn.n_node_inputs, device=node_inputs.device)
        
        # Place projected features back with correct dtype
        gnn_input_features[indices] = projected_check_features.to(gnn_input_features.dtype)

        # --- Step D: Spatial Processing ---
        gnn_outputs = self.gnn(gnn_input_features, src_ids, dst_ids)
        return gnn_outputs
