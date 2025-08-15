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

from mamba_ssm import Mamba

# ==============================================================================

class ChannelSE(nn.Module):
    """Squeeze-and-Excitation for channel attention"""
    def __init__(self, channels: int, reduction: int = 4, use_hsigmoid: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)
        self.act = nn.SiLU()
        self.gate = (lambda t: torch.clamp((t + 3) / 6, 0, 1)) if use_hsigmoid else torch.sigmoid
        
    def forward(self, x):  # x: [B,T,C]
        s = x.mean(dim=1)  # [B,C] - global average pooling over time
        a = self.fc2(self.act(self.fc1(s)))  # [B,C]
        w = self.gate(a).unsqueeze(1)        # [B,1,C]
        return x * w

class FiLM(nn.Module):
    """Feature-wise Linear Modulation for hardware priors"""
    def __init__(self, c_in: int, h_in: int, hidden: int = 32):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(h_in, hidden), 
            nn.SiLU(), 
            nn.Linear(hidden, c_in)
        )
        self.beta = nn.Sequential(
            nn.Linear(h_in, hidden), 
            nn.SiLU(), 
            nn.Linear(hidden, c_in)
        )
        
    def forward(self, x, h):  # x: [B,T,C], h: [B,H]
        g = self.gamma(h).unsqueeze(1)  # [B,1,C]
        b = self.beta(h).unsqueeze(1)   # [B,1,C]
        return g * x + b

class MGHD(nn.Module):
    """
    Mamba-Graph Hybrid Decoder (MGHD).
    This model combines a Mamba SSM for processing a sequence of check nodes
    and a GNN for spatial message passing.
    """

    def __init__(self, gnn_params, mamba_params):
        super().__init__()
        
        # Store mamba_params for later use
        self.mamba_params = mamba_params
        C = mamba_params['d_model']
        
        # Extract attention configuration
        self.attn_mech = mamba_params.get('attention_mechanism', 'none')
        
        # --- Part 1: Initial Feature Processing ---
        self.input_embedding = nn.Linear(gnn_params['n_node_inputs'], C)
        
        # --- Part 2: The Sequence Processor (Mamba) ---
        self.mamba = Mamba(
            d_model=C,
            d_state=mamba_params['d_state'],
            d_conv=mamba_params['d_conv'],
            expand=mamba_params['expand']
        )
        
        # --- Part 2b: Attention Mechanism (Lightweight) ---
        if self.attn_mech == 'channel_attention':
            self.se = ChannelSE(C, reduction=mamba_params.get('se_reduction', 4))
        elif self.attn_mech == 'cross_attention':
            # Placeholder for hardware priors - in practice, get from error characterization
            self.prior_dim = mamba_params.get('film_hidden_in', 16)  # optional explicit in-size
            self.film = FiLM(C, h_in=self.prior_dim, hidden=mamba_params.get('film_hidden', 32))
            # Until real priors are threaded in, keep a buffer as placeholder
            self.register_buffer('hardware_priors', torch.zeros(1, self.prior_dim))
        
        # --- Part 3: The Spatial Processor (GNN) ---
        self.gnn = GNNDecoder(**gnn_params)
        
        # --- Part 4: The Interface Layer (FIXED) ---
        # Project from Mamba's output to GNN's expected n_node_inputs (not n_node_features)
        self.projection = nn.Linear(C, gnn_params['n_node_inputs'])

    def forward(self, node_inputs, src_ids, dst_ids, priors=None):
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
        
        # --- Step B2: Apply Attention (if configured) ---
        if self.attn_mech == 'channel_attention':
            mamba_output_sequence = self.se(mamba_output_sequence)
        elif self.attn_mech == 'cross_attention':
            # Expand hardware priors to match batch size
            if priors is None:
                priors = self.hardware_priors.expand(batch_size, -1)
            mamba_output_sequence = self.film(mamba_output_sequence, priors)
        # 'none' case: no additional processing
        
        temporal_features = mamba_output_sequence.reshape(-1, self.mamba_params['d_model'])

        # --- Step C: Prepare Inputs for the GNN (FIXED) ---
        projected_check_features = self.projection(temporal_features)
        
        # Create tensor with correct dimensions (n_node_inputs, not n_node_features)
        gnn_input_features = torch.zeros(node_inputs.shape[0], self.gnn.n_node_inputs, device=node_inputs.device)
        
        # Place projected features back with correct dtype
        gnn_input_features[indices] = projected_check_features.to(gnn_input_features.dtype)

        # --- Step D: Spatial Processing ---
        gnn_outputs = self.gnn(gnn_input_features, src_ids, dst_ids)
        return gnn_outputs
