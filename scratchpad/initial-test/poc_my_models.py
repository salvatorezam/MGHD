import torch
import torch.nn as nn
import numpy as np

# --- Imports from our project ---
from panq_functions import GNNDecoder, surface_code_edges # From the Astra codebase
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
        
        # Optional stabilization: LayerNorm after Mamba (+attention)
        self.post_mamba_ln = bool(mamba_params.get('post_mamba_ln', False))
        if self.post_mamba_ln:
            self.mamba_ln = nn.LayerNorm(C)
        
        # --- Part 3: The Spatial Processor (GNN) ---
        self.gnn = GNNDecoder(**gnn_params)
        
        # --- Part 4: The Interface Layer (FIXED) ---
        # Project from Mamba's output to GNN's expected n_node_inputs (not n_node_features)
        self.projection = nn.Linear(C, gnn_params['n_node_inputs'])
        
        # Store dimensions for inference
        self.dist = gnn_params['dist']
        self.n_node_inputs = gnn_params['n_node_inputs']
        self.n_node_outputs = gnn_params['n_node_outputs']
        
        # Code configuration for authoritative indices
        self.code_config = {
            'type': 'surface',  # Default to surface code
            'distance': self.dist,
            'layout': 'planar'  # Default to planar, can be set to 'rotated'
        }
        
        # Precompute static indices for inference (will be created on first use)
        self.register_buffer('_check_node_indices', None, persistent=False)
        self.register_buffer('_src_ids', None, persistent=False)
        self.register_buffer('_dst_ids', None, persistent=False)
        self.register_buffer('_edge_index', None, persistent=False)
        # Optional authoritative Hx/Hz overrides (to ensure exact parity with external samplers)
        self.register_buffer('_auth_Hx', None, persistent=False)
        self.register_buffer('_auth_Hz', None, persistent=False)
    
    def set_rotated_layout(self):
        """Configure the model to use rotated surface code layout."""
        self.code_config['layout'] = 'rotated'
        # Clear cached indices to force rebuilding with new layout
        self._check_node_indices = None
        self._src_ids = None
        self._dst_ids = None
        self._edge_index = None
        
        # For rotated d=3, set n_node_outputs to 2 (binary classification per qubit)
        if self.dist == 3:
            # For per-qubit binary classification use 2 logits per data qubit
            self.gnn.n_node_outputs = 2
            # Update the final layer to output 2 logits - preserve device
            if hasattr(self.gnn, 'final_digits') and self.gnn.final_digits is not None:
                current_device = next(self.gnn.final_digits.parameters()).device
                self.gnn.final_digits = nn.Linear(self.gnn.n_node_features, 2).to(current_device)

    def set_authoritative_mats(self, Hx: np.ndarray, Hz: np.ndarray, device: str | torch.device = None):
        """Optionally override code matrices used to build graph indices.
        Ensures MGHD graph aligns exactly with external H matrices (e.g., from canonical pack/LUT).
        """
        dev = device if device is not None else next(self.parameters()).device
        Hx_t = torch.from_numpy(np.asarray(Hx, dtype=np.uint8)).to(dev)
        Hz_t = torch.from_numpy(np.asarray(Hz, dtype=np.uint8)).to(dev)
        # Register/replace buffers
        self.register_buffer('_auth_Hx', Hx_t, persistent=False)
        self.register_buffer('_auth_Hz', Hz_t, persistent=False)
        # Clear cached indices to rebuild with new matrices
        self._check_node_indices = None
        self._src_ids = None
        self._dst_ids = None
        self._edge_index = None

    def _build_authoritative_indices(self, device):
        """
        Build authoritative gather indices from actual code matrices.
        Creates proper bipartite graph structure from Hx/Hz matrices.
        """
        from panqec.codes import surface_2d

        code_type = self.code_config['type']
        distance = self.code_config['distance']
        layout = self.code_config.get('layout', 'planar')

        if code_type == 'surface':
            # Prefer explicitly provided authoritative H matrices (from pack/LUT)
            if (self._auth_Hx is not None) and (self._auth_Hz is not None):
                Hx = self._auth_Hx.detach().to(device).to(torch.uint8).cpu().numpy()
                Hz = self._auth_Hz.detach().to(device).to(torch.uint8).cpu().numpy()
            else:
                # Build from panqec for generality
                if layout == 'rotated':
                    code = surface_2d.RotatedPlanar2DCode(distance)
                else:
                    code = surface_2d.Planar2DCode(distance)
                # Extract stabilizer matrices
                Hx = code.Hx.toarray()  # X stabilizers
                Hz = code.Hz.toarray()  # Z stabilizers
            
            # Canonical ordering for rotated layouts is Z checks first, then X checks
            num_z_checks = Hz.shape[0]
            num_x_checks = Hx.shape[0]
            num_check_nodes = num_z_checks + num_x_checks
            
            # Number of data qubits
            num_data_qubits = Hx.shape[1]  # Should equal Hz.shape[1]
            
            # Total nodes: check nodes + data qubits
            total_nodes = num_check_nodes + num_data_qubits
            
            # Build edge indices from Hz (Z) then Hx (X) matrices (directed),
            # and add reverse edges to mirror message passing in training utilities
            src_ids = []
            dst_ids = []
            
            # Z stabilizers occupy check indices [0, num_z_checks)
            for i in range(num_z_checks):
                for j in range(num_data_qubits):
                    if Hz[i, j] == 1:
                        src_ids.append(i)
                        dst_ids.append(num_check_nodes + j)
            # X stabilizers occupy check indices [num_z_checks, num_z_checks+num_x_checks)
            for i in range(num_x_checks):
                for j in range(num_data_qubits):
                    if Hx[i, j] == 1:
                        src_ids.append(num_z_checks + i)
                        dst_ids.append(num_check_nodes + j)
            
            # Add reverse edges (data -> check)
            src_ids_rev = [num_check_nodes + j for _i in range(num_z_checks) for j in range(num_data_qubits) if Hz[_i, j] == 1]
            dst_ids_rev = [i for i in range(num_z_checks) for j in range(num_data_qubits) if Hz[i, j] == 1]
            src_ids_rev += [num_check_nodes + j for _i in range(num_x_checks) for j in range(num_data_qubits) if Hx[_i, j] == 1]
            dst_ids_rev += [num_z_checks + i for i in range(num_x_checks) for j in range(num_data_qubits) if Hx[i, j] == 1]

            src_ids_all = src_ids + src_ids_rev
            dst_ids_all = dst_ids + dst_ids_rev

            # Convert to tensors
            src_ids = torch.tensor(src_ids_all, device=device, dtype=torch.long)
            dst_ids = torch.tensor(dst_ids_all, device=device, dtype=torch.long)
            
            # Create check node indices
            check_indices = torch.arange(num_check_nodes, device=device, dtype=torch.long)
            
            # Assert bounds checking
            assert src_ids.max() < total_nodes, f"src_ids max {src_ids.max()} >= total_nodes {total_nodes}"
            assert dst_ids.max() < total_nodes, f"dst_ids max {dst_ids.max()} >= total_nodes {total_nodes}"
            assert len(src_ids) > 0, "No edges found in code graph"

            # Build node_feature_index: map each node to a syndrome feature index (only checks map, qubits = -1)
            node_feature_index = torch.full((total_nodes, 1), fill_value=-1, device=device, dtype=torch.long)
            node_feature_index[:num_check_nodes, 0] = torch.arange(num_check_nodes, device=device, dtype=torch.long)

            # Edge index convenience buffer [2, num_edges]
            edge_index = torch.stack([src_ids, dst_ids], dim=0)

            return (
                check_indices,
                src_ids,
                dst_ids,
                edge_index,
                node_feature_index,
                num_check_nodes,
                num_data_qubits,
            )
            
        elif code_type == 'bb':
            # Build from BB code matrices (using existing utilities)
            from bb_panq_functions import bb_code
            code = bb_code(distance)

            Hx = code.hx.astype(int)
            Hz = code.hz.astype(int)

            # Canonical ordering: Z then X
            num_z_checks = Hz.shape[0]
            num_x_checks = Hx.shape[0]
            num_check_nodes = num_z_checks + num_x_checks
            num_data_qubits = Hx.shape[1]
            total_nodes = num_check_nodes + num_data_qubits

            src_ids = []
            dst_ids = []
            # Z checks first
            for i in range(num_z_checks):
                for j in range(num_data_qubits):
                    if Hz[i, j] == 1:
                        src_ids.append(i)
                        dst_ids.append(num_check_nodes + j)
            # X checks next
            for i in range(num_x_checks):
                for j in range(num_data_qubits):
                    if Hx[i, j] == 1:
                        src_ids.append(num_z_checks + i)
                        dst_ids.append(num_check_nodes + j)

            # Add reverse edges
            src_ids_rev = [num_check_nodes + j for _i in range(num_z_checks) for j in range(num_data_qubits) if Hz[_i, j] == 1]
            dst_ids_rev = [i for i in range(num_z_checks) for j in range(num_data_qubits) if Hz[i, j] == 1]
            src_ids_rev += [num_check_nodes + j for _i in range(num_x_checks) for j in range(num_data_qubits) if Hx[_i, j] == 1]
            dst_ids_rev += [num_z_checks + i for i in range(num_x_checks) for j in range(num_data_qubits) if Hx[i, j] == 1]

            src_ids_all = src_ids + src_ids_rev
            dst_ids_all = dst_ids + dst_ids_rev

            src_ids = torch.tensor(src_ids_all, device=device, dtype=torch.long)
            dst_ids = torch.tensor(dst_ids_all, device=device, dtype=torch.long)

            check_indices = torch.arange(num_check_nodes, device=device, dtype=torch.long)

            assert src_ids.max() < total_nodes, f"src_ids max {src_ids.max()} >= total_nodes {total_nodes}"
            assert dst_ids.max() < total_nodes, f"dst_ids max {dst_ids.max()} >= total_nodes {total_nodes}"
            assert len(src_ids) > 0, "No edges found in code graph (BB)"

            node_feature_index = torch.full((total_nodes, 1), fill_value=-1, device=device, dtype=torch.long)
            node_feature_index[:num_check_nodes, 0] = torch.arange(num_check_nodes, device=device, dtype=torch.long)
            edge_index = torch.stack([src_ids, dst_ids], dim=0)

            return (
                check_indices,
                src_ids,
                dst_ids,
                edge_index,
                node_feature_index,
                num_check_nodes,
                num_data_qubits,
            )
            
        else:
            raise ValueError(f"Unknown code type: {code_type}")

    def set_output_size_from_metadata(self, meta: dict):
        """
        Adjust output head ONLY when appropriate.
        For rotated layouts we enforce a binary head (2 logits per data qubit).
        This prevents accidental reversion to a 9-logit multi-class head.
        """
        if not isinstance(meta, dict):
            return

        # If meta explicitly says rotated layout, enforce binary head
        if meta.get('surface_layout') == 'rotated':
            if getattr(self.gnn, 'n_node_outputs', None) != 2:
                device = next(self.parameters()).device
                self.gnn.n_node_outputs = 2
                # Recreate final layer to 2 logits (binary per qubit)
                if hasattr(self.gnn, 'final_digits'):
                    self.gnn.final_digits = nn.Linear(self.gnn.n_node_features, 2).to(device)
                print("[MGHD] Enforced binary head (2 logits/qubit) for rotated layout from metadata")
            return

        # For non-rotated layouts we currently keep the binary head as well.
        # If in future we support multi-class heads, gate it explicitly here by layout name.
        return

    def _ensure_static_indices(self, device):
        """Create static index tensors for inference if they don't exist"""
        if self._check_node_indices is None:
            # Build authoritative indices from actual code matrices
            (
                check_indices,
                src_ids,
                dst_ids,
                edge_index,
                node_feature_index,
                num_check_nodes,
                num_data_qubits,
            ) = self._build_authoritative_indices(device)
            
            # Register the authoritative indices as buffers
            self.register_buffer('_check_node_indices', check_indices, persistent=False)
            self.register_buffer('_src_ids', src_ids, persistent=False)
            self.register_buffer('_dst_ids', dst_ids, persistent=False)
            self.register_buffer('_edge_index', edge_index, persistent=False)
            self.register_buffer('_node_feature_index', node_feature_index, persistent=False)
            
            # Store dimensions for later use
            self._num_check_nodes = num_check_nodes
            self._num_data_qubits = num_data_qubits

    def decode_one(self, packed_syndrome: torch.ByteTensor, *, device: str="cuda", temporal_T: int=1, metadata: dict = None) -> torch.ByteTensor:
        """
        Fast inference for a single syndrome (batch=1).
        
        Args:
            packed_syndrome: Either [N_bytes] packed or [N_syn] unpacked syndrome
            device: Target device
            temporal_T: Number of temporal steps (default=1 for single-shot)
            
        Returns:
            Correction decisions as uint8 tensor [1, N_bits]
        """
        # Ensure static indices are created
        self._ensure_static_indices(device)
        
        # Unit check: assert indices are non-empty and within bounds
        assert self._src_ids.numel() > 0, "Source indices are empty"
        assert self._dst_ids.numel() > 0, "Destination indices are empty"
        total_nodes_bound = self._num_check_nodes + self._num_data_qubits
        assert int(self._src_ids.max().item()) < total_nodes_bound, (
            f"src_ids max {int(self._src_ids.max().item())} >= total nodes {total_nodes_bound}"
        )
        assert int(self._dst_ids.max().item()) < total_nodes_bound, (
            f"dst_ids max {int(self._dst_ids.max().item())} >= total nodes {total_nodes_bound}"
        )
        
        # Optional metadata shape check
        if metadata and isinstance(metadata, dict) and 'N_bits' in metadata:
            if hasattr(self.gnn, 'n_node_outputs') and self.gnn.n_node_outputs != int(metadata['N_bits']):
                raise ValueError(f"Model head outputs {self.gnn.n_node_outputs} != metadata N_bits {metadata['N_bits']}")

        # Handle packed vs unpacked input
        N_syn = self._num_check_nodes  # Number of syndrome bits from authoritative indices
        N_bytes = (N_syn + 7) // 8  # Ceiling division
        
        if packed_syndrome.dim() == 1 and packed_syndrome.dtype == torch.uint8:
            if packed_syndrome.shape[0] == N_bytes:
                # Packed input - need to unpack
                unpacked = torch.zeros(1, N_syn, dtype=torch.float32, device=device)
                for i in range(N_syn):
                    byte_idx = i // 8
                    bit_idx = i % 8
                    unpacked[0, i] = (packed_syndrome[byte_idx] >> bit_idx) & 1
                syndrome = unpacked
            elif packed_syndrome.shape[0] == N_syn:
                # Unpacked input with uint8 dtype
                syndrome = packed_syndrome.to(dtype=torch.float32, device=device)
                syndrome = syndrome.unsqueeze(0)  # Add batch dimension
            else:
                raise ValueError(f"Expected {N_bytes} bytes for packed or {N_syn} bits for unpacked syndrome, got {packed_syndrome.shape[0]}")
        else:
            # Assume already unpacked with different dtype
            syndrome = packed_syndrome.to(dtype=torch.float32, device=device)
            if syndrome.dim() == 1:
                syndrome = syndrome.unsqueeze(0)  # Add batch dimension
        
        # Ensure we have the right shape [1, N_syn]
        if syndrome.shape != (1, self._num_check_nodes):
            raise ValueError(f"Expected syndrome shape [1, {self._num_check_nodes}], got {syndrome.shape}")
        
        # Create full node inputs tensor [1, nodes_per_graph, n_node_inputs]
        num_check_nodes = self._num_check_nodes
        num_qubit_nodes = self._num_data_qubits
        nodes_per_graph = num_check_nodes + num_qubit_nodes
        
        # Initialize with zeros
        node_inputs = torch.zeros(1, nodes_per_graph, self.n_node_inputs, device=device, dtype=torch.float32)
        
        # Place syndrome in check node positions (first num_check_nodes positions)
        node_inputs[0, :num_check_nodes, 0] = syndrome[0, :num_check_nodes]
        
        # Reshape for model forward pass [batch*n_nodes, n_node_inputs]
        node_inputs_flat = node_inputs.view(-1, self.n_node_inputs)
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = self.forward(node_inputs_flat, self._src_ids, self._dst_ids)
        
        # Extract final iteration outputs and qubit node decisions
        final_outputs = outputs[-1]  # [batch*n_nodes, n_node_outputs]
        
        # Extract qubit node outputs (last num_qubit_nodes positions)
        qubit_outputs = final_outputs[num_check_nodes:num_check_nodes + num_qubit_nodes]
        
        # Convert to binary decisions (argmax for classification)
        decisions = torch.argmax(qubit_outputs, dim=1, keepdim=True)  # [num_qubit_nodes, 1]
        
        # Reshape to [1, N_bits] and convert to uint8
        decisions = decisions.view(1, -1).to(torch.uint8)
        
        return decisions

    def export_onnx_int8_ready(self, path: str, N_syn: int, N_bits: int):
        """
        Export model to ONNX format ready for int8 quantization.
        
        Args:
            path: Output ONNX file path
            N_syn: Number of syndrome bits
            N_bits: Number of output bits
        """
        device = next(self.parameters()).device
        
        # Create dummy inputs for tracing
        # Ensure static indices are created to get proper dimensions
        self._ensure_static_indices(device)
        num_check_nodes = self._num_check_nodes
        num_qubit_nodes = self._num_data_qubits
        nodes_per_graph = num_check_nodes + num_qubit_nodes
        

        
        # Create dummy inputs
        dummy_node_inputs = torch.randn(1, nodes_per_graph, self.n_node_inputs, device=device, dtype=torch.float32)
        dummy_node_inputs_flat = dummy_node_inputs.view(-1, self.n_node_inputs)
        
        # Set model to eval mode
        self.eval()
        
        # Export to ONNX
        torch.onnx.export(
            self,
            (dummy_node_inputs_flat, self._src_ids, self._dst_ids),
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['node_inputs', 'src_ids', 'dst_ids'],
            output_names=['outputs'],
            dynamic_axes={
                'node_inputs': {0: 'batch_nodes'},
                'outputs': {1: 'batch_nodes'}
            }
        )
        
        print(f"Model exported to {path}")
        print(f"Input shape: [batch_nodes, {self.n_node_inputs}]")
        print(f"Output shape: [{self.gnn.n_iters}, batch_nodes, {self.n_node_outputs}]")

    def forward(self, node_inputs, src_ids, dst_ids, priors=None):
        """
        Defines the full data flow through the hybrid model.
        """
        # Use authoritative sizes derived from Hx/Hz (supports rotated d=3: 8+9=17)
        self._ensure_static_indices(node_inputs.device)
        num_check_nodes = self._num_check_nodes
        num_qubit_nodes = self._num_data_qubits
        nodes_per_graph = num_check_nodes + num_qubit_nodes
        batch_size = node_inputs.shape[0] // nodes_per_graph

        # --- Step A: Isolate and Reshape Data for Mamba (vectorized) ---
        # node_inputs: [B*n_nodes, n_node_inputs] -> [B, n_nodes, n_node_inputs]
        xin = node_inputs.view(batch_size, nodes_per_graph, self.n_node_inputs)
        # Slice first num_check_nodes positions (the check nodes), then flatten back
        check_node_inputs = xin[:, :num_check_nodes, :].reshape(-1, self.n_node_inputs)
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
        if self.post_mamba_ln:
            mamba_output_sequence = self.mamba_ln(mamba_output_sequence)
        
        temporal_features = mamba_output_sequence.reshape(-1, self.mamba_params['d_model'])

        # --- Step C: Prepare Inputs for the GNN (FIXED) ---
        projected_check_features = self.projection(temporal_features)
        
        # Create tensor with correct dimensions (n_node_inputs, not n_node_features)
        gnn_input_features = torch.zeros(node_inputs.shape[0], self.gnn.n_node_inputs, device=node_inputs.device)
        
        # Place projected features back with correct dtype (vectorized)
        # projected_check_features is [B*num_check_nodes, n_node_inputs]
        # Reshape to [B, num_check_nodes, n_node_inputs], then place in gnn_input
        projected_reshaped = projected_check_features.view(batch_size, num_check_nodes, self.gnn.n_node_inputs)
        gnn_input_reshaped = gnn_input_features.view(batch_size, nodes_per_graph, self.gnn.n_node_inputs)
        gnn_input_reshaped[:, :num_check_nodes, :] = projected_reshaped.to(gnn_input_features.dtype)
        gnn_input_features = gnn_input_reshaped.view(-1, self.gnn.n_node_inputs)

        # --- Step D: Spatial Processing ---
        gnn_outputs = self.gnn(gnn_input_features, src_ids, dst_ids)
        return gnn_outputs
