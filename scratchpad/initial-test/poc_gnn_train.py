from panqec.codes import surface_2d
from panqec.error_models import PauliErrorModel
from panqec.decoders import MatchingDecoder, BeliefPropagationOSDDecoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import time
import os
import sys
import csv
from datetime import datetime

# Create output directory
output_dir = "/u/home/kulp/MGHD/scratchpad/initial-test/Plots and Data"
os.makedirs(output_dir, exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train GNN decoder with optional CUDA-Q backend')
parser.add_argument('--backend', choices=['panqec', 'cudaq'], default='panqec',
                    help='Backend for syndrome generation (default: panqec)')
parser.add_argument('--cudaq-mode', choices=['foundation', 'student'], default='foundation',
                    help='CUDA-Q mode: foundation (device-agnostic) or student (device-specific)')
parser.add_argument('--T-rounds', type=int, default=1,
                    help='Number of syndrome extraction rounds for CUDA-Q (default: 1)')
parser.add_argument('--bitpack', action='store_true',
                    help='Store syndrome data as bit-packed uint8 (CUDA-Q only)')
parser.add_argument('--d', type=int, default=3,
                    help='Surface code distance (default: 3)')
parser.add_argument('--surface-layout', choices=['rotated', 'planar'], default='rotated',
                    help='Surface code layout type (default: rotated)')
parser.add_argument('--teacher-syndromes', type=str, default=None, help='NPZ with teacher input syndromes')
parser.add_argument('--teacher-labels-relay', type=str, default=None, help='NPZ with relay hard_labels')
parser.add_argument('--teacher-labels-mwpm', type=str, default=None, help='NPZ with MWPM hard_labels')
parser.add_argument('--teacher-labels-mwpf', type=str, default=None, help='NPZ with MWPF hard_labels')
parser.add_argument('--teacher-labels-ensemble', type=str, default=None, help='NPZ with ensemble hard_labels')
parser.add_argument('--teacher', choices=['none', 'mwpm', 'relay', 'mwpf', 'ensemble'], default='none',
                    help='Teacher for generating labels: none, mwpm, relay, mwpf, or ensemble (default: none)')
parser.add_argument('--epochs', type=int, default=1,
                    help='Number of training epochs (default: 1)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Training batch size (default: 128)')
parser.add_argument('--steps-per-epoch', type=int, default=0,
                    help='Max training steps per epoch; 0 means process all batches')

# Try to parse args, but provide defaults if running in notebook/interactive mode
try:
    args = parser.parse_args()
    d = args.d
    backend = args.backend
    cudaq_mode = args.cudaq_mode
    T_rounds = args.T_rounds
    bitpack = args.bitpack
    surface_layout = args.surface_layout
    teacher_synd = args.teacher_syndromes
    teacher_relay = args.teacher_labels_relay
    teacher_mwpm = args.teacher_labels_mwpm
    teacher = args.teacher
    epochs = args.epochs
    batch_size = args.batch_size
    steps_per_epoch = args.steps_per_epoch
except SystemExit:
    # Fallback for interactive/notebook usage
    class Args:
        def __init__(self):
            self.backend = 'panqec'
            self.cudaq_mode = 'foundation'
            self.T_rounds = 1
            self.bitpack = False
            self.d = 3
            self.surface_layout = 'rotated'
            self.teacher_syndromes = None
            self.teacher_labels_relay = None
            self.teacher_labels_mwpm = None
            self.teacher_labels_mwpf = None
            self.teacher_labels_ensemble = None
            self.teacher = 'none'
            self.epochs = 1
            self.batch_size = 128
            self.steps_per_epoch = 0
    args = Args()
    d = args.d
    backend = args.backend
    cudaq_mode = args.cudaq_mode
    T_rounds = args.T_rounds
    bitpack = args.bitpack
    surface_layout = args.surface_layout
    teacher_synd = args.teacher_syndromes
    teacher_relay = args.teacher_labels_relay
    teacher_mwpm = args.teacher_labels_mwpm
    teacher = args.teacher
    epochs = args.epochs
    batch_size = args.batch_size
    steps_per_epoch = args.steps_per_epoch

# Generate timestamp for this run
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = f"MGHD_vs_Baseline_d{d}_{backend}_{run_timestamp}"

print(f"Configuration:")
print(f"  Distance: {d}")
print(f"  Backend: {backend}")
print(f"  Surface layout: {surface_layout}")
print(f"  Teacher: {teacher}")
if backend == 'cudaq':
    print(f"  CUDA-Q Mode: {cudaq_mode}")
    print(f"  T-rounds: {T_rounds}")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Steps per epoch: {steps_per_epoch if steps_per_epoch > 0 else 'all'}")

# Setup CUDA-Q configuration if needed
cudaq_cfg = None
if backend == 'cudaq':
    from cudaq_backend.circuits import make_surface_layout_d3_avoid_bad_edges
    
    if d == 3:
        layout = make_surface_layout_d3_avoid_bad_edges()
    else:
        # For other distances, use a generic layout (simplified)
        layout = {
            'data': list(range(d*d)),
            'ancilla_x': list(range(d*d, d*d + (d*d-1)//2)),
            'ancilla_z': list(range(d*d + (d*d-1)//2, d*d + d*d-1)),
            'cz_layers': [],
            'prx_layers': [],
            'total_qubits': 2*d*d - 1,
            'distance': d
        }
    
    cudaq_cfg = {
        "layout": layout,
        "T": T_rounds,
        "rng": np.random.default_rng(1234),
        "bitpack": bitpack
    }
    print(f"  Layout qubits: {layout['total_qubits']}")

# Continue with original parameter setup...

# import tools
from panq_functions import GNNDecoder, collate, fraction_of_solved_puzzles, compute_accuracy, logical_error_rate, \
    surface_code_edges, generate_syndrome_error_volume, adapt_trainset, ler_loss, save_model, load_model
from ldpc.mod2 import *

if torch.cuda.is_available():
    device = torch.device('cuda')
    use_amp = True # to use automatic mixed precision
    amp_data_type = torch.float16
else:
    device = torch.device('cpu')
    use_amp = True
    '''float16 is not supported for cpu use bfloat16 instead'''
    amp_data_type = torch.bfloat16

# Import MGHD class from poc_my_models.py
from poc_my_models import MGHD

"""
Parameters
"""
d = 3
error_model_name = "DP"

if (error_model_name == "X"):
    error_model = PauliErrorModel(1, 0.0, 0)
elif (error_model_name == "Z"):
    error_model = PauliErrorModel(0, 0.0, 1)
elif (error_model_name == "XZ"):
    error_model = PauliErrorModel(0.5, 0.0, 0.5)
elif (error_model_name == "DP"):
    error_model = PauliErrorModel(0.34, 0.32, 0.34)

# list of hyperparameters (ALL 23 parameters optimized with comprehensive attention mechanism investigation)
n_node_inputs = 4
n_node_outputs = 4
n_iters = 7  # Optimized: 7 (was 8)
n_node_features = 128  # Optimized: 128 (was 256)
n_edge_features = 384  # Optimized: 384 (was 256)
len_test_set = 5000  # Back to smaller test set for faster, optimistic tracking during training
test_err_rate = 0.05
len_train_set = 20000 # original len_test_set * 10
max_train_err_rate = 0.15

# ============ TeacherDataset for supervised distillation ============
class TeacherDataset(torch.utils.data.Dataset):
    def __init__(self, syndromes_npz: str, relay_npz: str = None, mwpm_npz: str = None):
        data = np.load(syndromes_npz)
        if 'syndromes' not in data:
            raise ValueError("Teacher syndromes NPZ missing 'syndromes'")
        self.synd = data['syndromes'].astype(np.uint8)
        self.meta = None
        if 'metadata_json' in data:
            try:
                import json
                self.meta = json.loads(str(data['metadata_json'].item()))
            except Exception:
                self.meta = None
        self.labels_relay = None
        self.labels_mwpm = None
        if relay_npz:
            z = np.load(relay_npz)
            self.labels_relay = z['hard_labels'].astype(np.uint8)
        if mwpm_npz:
            z = np.load(mwpm_npz)
            arr = z['hard_labels_mwpm']
            if arr.size > 0:
                self.labels_mwpm = arr.astype(np.uint8)
        N = self.synd.shape[0]
        if self.labels_relay is not None and self.labels_relay.shape[0] != N:
            raise ValueError("Relay labels length mismatch with syndromes")
        if self.labels_mwpm is not None and self.labels_mwpm.shape[0] != N:
            raise ValueError("MWPM labels length mismatch with syndromes")

    def __len__(self):
        return self.synd.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.synd[idx])  # [N_syn] uint8
        y_relay = torch.from_numpy(self.labels_relay[idx]) if self.labels_relay is not None else None
        y_mwpm = torch.from_numpy(self.labels_mwpm[idx]) if self.labels_mwpm is not None else None
        return x, y_relay, y_mwpm

# ---- Curriculum for training error rate p (center near 0.05 over time) ----
def sample_training_p(epoch, total_epochs):
    third = max(1, total_epochs // 3)
    if epoch < third:
        lo, hi = 0.04, 0.10
    elif epoch < 2 * third:
        lo, hi = 0.035, 0.075
    else:
        lo, hi = 0.03, 0.06
    return float(np.random.uniform(lo, hi))

# ---- Stability & early-stop config ----
eval_runs = 1           # Back to single eval run for faster, optimistic tracking
final_eval_runs = 5     # number of times to eval at the very end (best checkpoint)
early_stop_patience = 8 # epochs without improvement before stopping
early_stop_min_delta = 0.0  # required improvement in LER to reset patience

lr = 6.839647835588333e-05  # LOCKED: Trial B winner - optimal LR from investigation
weight_decay = 0.00010979214543158697  # Optimized: precise value from investigation
msg_net_size = 96  # Optimized: 96 (was 128)
msg_net_dropout_p = 0.04561710273200902  # Optimized: precise value from investigation
gru_dropout_p = 0.08904846656472562  # Optimized: precise value from investigation

# Advanced training hyperparameters discovered in optimization
lr_schedule = "constant"  # Optimized: constant scheduling
warmup_steps = 28  # Optimized: 28 warmup steps
label_smoothing = 0.14  # LOCKED: Trial B winner - optimal smoothing for 0.0548 LER
gradient_clip = 4.039566817780428  # Optimized: precise gradient clipping value
residual_connections = 1  # Optimized: enable residual connections
noise_injection = 0.005446402602129624  # Optimized: precise noise injection
accumulation_steps = 1  # Optimized: gradient accumulation steps

# Define the Mamba parameters (optimized with Optuna + attention mechanism)
mamba_params = {
    'd_model': 192,  # Optimized: 192 (was 256)
    'd_state': 64,   # Optimized: 64 (was 55)
    'd_conv': 2,     # Optimized: 2 (confirmed)
    'expand': 3,     # Optimized: 3 (was 4)
    # NEW: Channel attention (SE) with optimal reduction factor
    'attention_mechanism': 'channel_attention',
    'se_reduction': 4,  # Optimal setting from 0.0532 LER achievement
    'mamba_layers': 1  # Keep single layer for fair comparison
}

print("n_iters: ", n_iters, "n_node_outputs: ", n_node_outputs, "n_node_features: ", n_node_features,
      "n_edge_features: ", n_edge_features)
print("msg_net_size: ", msg_net_size, "msg_net_dropout_p: ", msg_net_dropout_p, "gru_dropout_p: ", gru_dropout_p)
print("learning rate: ", lr, "weight decay: ", weight_decay, "len train set: ", len_train_set, 'max train error rate: ',
      max_train_err_rate, "len test set: ", len_test_set, "test error rate: ", test_err_rate)
print("MGHD: d_model:", mamba_params['d_model'], "d_state:", mamba_params['d_state'], 
      "attention:", mamba_params['attention_mechanism'], "se_reduction:", mamba_params['se_reduction'])

"""
Create the Surface code
"""
dist = d # can be different in case of using lower distance trained decoder for larger distance
print('trained', d, '\t retrain', dist)

code = surface_2d.RotatedPlanar2DCode(dist)

src, tgt = surface_code_edges(code)
src_tensor = torch.LongTensor(src)
tgt_tensor = torch.LongTensor(tgt)
GNNDecoder.surface_code_edges = (src_tensor, tgt_tensor)

hxperp = torch.FloatTensor(nullspace(code.Hx.toarray())).to(device)
hzperp = torch.FloatTensor(nullspace(code.Hz.toarray())).to(device)
GNNDecoder.hxperp = hxperp
GNNDecoder.hzperp = hzperp
GNNDecoder.device = device

# ---------------------------
print("Initializing models for comparison...")
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name()}")

# 1. Create the Baseline GNN Model
gnn_params = {
    'dist': dist, 'n_node_inputs': n_node_inputs, 'n_node_outputs': n_node_outputs,
    'n_iters': n_iters, 'n_node_features': n_node_features, 'n_edge_features': n_edge_features,
    'msg_net_size': msg_net_size, 'msg_net_dropout_p': msg_net_dropout_p, 'gru_dropout_p': gru_dropout_p
}

gnn_baseline = GNNDecoder(**gnn_params).to(device)
optimizer_baseline = optim.AdamW(gnn_baseline.parameters(), lr=lr, weight_decay=weight_decay)
scheduler_baseline = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_baseline, mode='min', factor=0.5, patience=3
)
print(f"Baseline GNN parameters: {sum(p.numel() for p in gnn_baseline.parameters())}")

# 2. Create your new Hybrid MGHD Model
mghd_model = MGHD(gnn_params=gnn_params, mamba_params=mamba_params).to(device)

# If teacher metadata provided, adjust MGHD head to match N_bits (e.g., 9 for rotated)
if teacher_synd is not None:
    try:
        import json as _json
        _nz = np.load(teacher_synd)
        if 'metadata_json' in _nz:
            _meta = _json.loads(str(_nz['metadata_json'].item()))
            mghd_model.set_output_size_from_metadata(_meta)
            print(f"[Trainer] Adjusted MGHD head per metadata N_bits={_meta.get('N_bits')} (Surface layout: {_meta.get('surface_layout')})")
        else:
            _meta = None
    except Exception as _e:
        print(f"[Trainer] Warning: failed to read teacher metadata: {_e}")
        _meta = None
else:
    _meta = None

# If teacher relay labels provided, assert width equals model head width
def _check_label_width(npz_path: str, label_key: str, expected: int):
    if npz_path is None:
        return
    try:
        _z = np.load(npz_path)
        if label_key not in _z:
            return
        _arr = _z[label_key]
        if _arr.size == 0:
            return
        if _arr.shape[1] != expected:
            print(f"[ERROR] Teacher labels width {_arr.shape[1]} != model head {expected}")
            sys.exit(2)
    except Exception as _e:
        print(f"[Trainer] Warning: could not validate teacher labels '{npz_path}': {_e}")

_head_w = int(getattr(mghd_model.gnn, 'n_node_outputs', n_node_outputs))
_check_label_width(teacher_relay, 'hard_labels', _head_w)
_check_label_width(teacher_mwpm, 'hard_labels_mwpm', _head_w)

# Fail-fast: If rotated surface layout and no metadata, ensure expected N_bits=9
if surface_layout == 'rotated' and _meta is None and (teacher_relay is not None or teacher_mwpm is not None):
    expected_rotated_bits = 9
    if _head_w != expected_rotated_bits:
        print(f"[ERROR] Rotated surface code expects {expected_rotated_bits} bits but model head has {_head_w}")
        sys.exit(2)

# ---- EMA for MGHD (eval-only) ----
ema_decay = 0.999
ema_state = {k: v.detach().clone() for k, v in mghd_model.state_dict().items()}

def ema_update(model):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema_state[k].mul_(ema_decay).add_(v.detach().clone(), alpha=1.0 - ema_decay)

def load_ema(model):
    model.load_state_dict(ema_state, strict=True)

optimizer_mghd = optim.AdamW(mghd_model.parameters(), lr=lr, weight_decay=weight_decay)
# For optimal results, use constant learning rate (lr_schedule = "constant")
# scheduler_mghd = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_mghd, mode='min', factor=0.5, patience=3
# )
scheduler_mghd = torch.optim.lr_scheduler.LambdaLR(optimizer_mghd, lr_lambda=lambda epoch: 1.0)

# Warmup scheduler for first 28 steps as optimized
def warmup_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

warmup_scheduler_mghd = torch.optim.lr_scheduler.LambdaLR(optimizer_mghd, lr_lambda=warmup_lambda)
print(f"Hybrid MGHD parameters: {sum(p.numel() for p in mghd_model.parameters())}")

# Generate the test data
testset = adapt_trainset(
    generate_syndrome_error_volume(code, error_model=error_model, p=test_err_rate, batch_size=len_test_set,
                                   for_training=False, backend=backend, cudaq_mode=cudaq_mode, cudaq_cfg=cudaq_cfg), 
    code, num_classes=n_node_inputs, for_training=False)
testloader = DataLoader(testset, batch_size=512, collate_fn=collate, shuffle=False)

print(f"Dataset size: {len_train_set}")
print(f"Test set size: {len(testset)}")

"""
Train
"""
""" automatic mixed precision """
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

# Use CLI-controlled training duration and batch size
epochs = int(epochs)
batch_size = int(batch_size)
# 0 means use all batches in the DataLoader
steps_per_epoch = int(steps_per_epoch)

# Keep constant LR (no cosine annealing) - this was key to ~0.06 LER sweet spot

# IMPROVED: Add label smoothing and gradient clipping for better optimization
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)  # Optimized: 0.142 from investigation
gradient_clip_value = gradient_clip  # Optimized: 4.04 from investigation

start_time = time.time()
size = 2 * GNNDecoder.dist ** 2 - 1
error_index = GNNDecoder.dist ** 2 - 1

# Initialize lists to store metrics for plotting
baseline_losses = []
baseline_lers = []
baseline_lerx = []
baseline_lerz = []
baseline_frac_solved = []

mghd_losses = []
mghd_lers = []
mghd_lerx = []
mghd_lerz = []
mghd_frac_solved = []

epoch_times = []

# Early stopping + Best checkpoint
best_mghd_ler = float('inf')
best_mghd_state = None
best_epoch = 0
current_global_step = 0

def evaluate_model_avg(model, loader, code, runs=1):
    """Return (lerx_mean, lerz_mean, lertot_mean) across 'runs' repeated evaluations."""
    lerx_vals, lerz_vals, lertot_vals = [], [], []
    model.eval()
    with torch.no_grad():
        for _ in range(max(1, runs)):
            lerx, lerz, lertot = logical_error_rate(model, loader, code)
            lerx_vals.append(float(lerx))
            lerz_vals.append(float(lerz))
            lertot_vals.append(float(lertot))
    return float(np.mean(lerx_vals)), float(np.mean(lerz_vals)), float(np.mean(lertot_vals))

def evaluate_over_p_grid(model, code, ps=(0.03, 0.04, 0.05, 0.06, 0.08), runs=1, set_size=5000):
    results = []
    for p in ps:
        testset_p = adapt_trainset(
            generate_syndrome_error_volume(code, error_model=error_model, p=p, batch_size=set_size, 
                                           for_training=False, backend=backend, cudaq_mode=cudaq_mode, cudaq_cfg=cudaq_cfg),
            code, num_classes=n_node_inputs, for_training=False)
        testloader_p = DataLoader(testset_p, batch_size=512, collate_fn=collate, shuffle=False)
        lerx, lerz, lertot = evaluate_model_avg(model, testloader_p, code, runs=runs)
        results.append((float(p), float(lerx), float(lerz), float(lertot)))
    ps_arr = np.array([r[0] for r in results], dtype=float)
    lers = np.array([r[3] for r in results], dtype=float)
    auc = float(np.trapz(lers, ps_arr) / (ps_arr.max() - ps_arr.min()))
    return results, auc

# CSV Metrics Logger
csv_path = os.path.join(output_dir, f'{run_id}_training_metrics.csv')
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'epoch',
            'baseline_loss', 'baseline_ler', 'baseline_lerx', 'baseline_lerz', 'baseline_frac_solved',
            'mghd_loss', 'mghd_ler', 'mghd_lerx', 'mghd_lerz', 'mghd_frac_solved',
            'mghd_lr'
        ])
    print(f"Initialized CSV logging at {csv_path}")

# CSV for multi-p evaluation
pgrid_csv = os.path.join(output_dir, f'{run_id}_pgrid_metrics.csv')
if not os.path.exists(pgrid_csv):
    with open(pgrid_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'epoch', 'model', 'p', 'ler_total', 'ler_x', 'ler_z'])
    print(f"Initialized p-grid CSV at {pgrid_csv}")

# Modified training loop to compare both models
print("Starting training...")
print("epoch, baseline_loss, baseline_LER, mghd_loss, mghd_LER, train_time")
print("Curriculum sampling active: p ranges [0.04,0.10] -> [0.035,0.075] -> [0.03,0.06]")
print("Multi-p metrics written to pgrid_metrics.csv (per-epoch and final AUC)")
sys.stdout.flush()

# Early stopping tracking
epochs_without_improve = 0

for epoch in range(epochs):
    epoch_start_time = time.time()
    
    # Set both models to training mode
    gnn_baseline.train()
    mghd_model.train()

    epoch_loss_baseline = []
    epoch_loss_mghd = []

    # Progress indicator
    print(f"\nEpoch {epoch+1}/{epochs} - Training...")
    
    # Regenerate training data each epoch with curriculum p near 0.05
    p_train = sample_training_p(epoch, epochs)
    if teacher_synd is not None:
        # Use TeacherDataset for supervised distillation
        distil_ds = TeacherDataset(teacher_synd, relay_npz=teacher_relay, mwpm_npz=teacher_mwpm)
        # Re-validate head size from metadata if present
        if distil_ds.meta is not None:
            mghd_model.set_output_size_from_metadata(distil_ds.meta)
            _head_w = int(getattr(mghd_model.gnn, 'n_node_outputs', n_node_outputs))
            # Validate teacher widths
            if distil_ds.labels_relay is not None and distil_ds.labels_relay.shape[1] != _head_w:
                print(f"[ERROR] Relay labels width {distil_ds.labels_relay.shape[1]} != model head {_head_w}")
                sys.exit(2)
            if distil_ds.labels_mwpm is not None and distil_ds.labels_mwpm.shape[1] != _head_w:
                print(f"[ERROR] MWPM labels width {distil_ds.labels_mwpm.shape[1]} != model head {_head_w}")
                sys.exit(2)
        # Simple collate for distillation: produce node_inputs and teacher labels
        def _distil_collate(batch):
            xs = [torch.as_tensor(b[0], dtype=torch.uint8) for b in batch]
            ys_relay = [b[1] for b in batch] if batch[0][1] is not None else None
            ys_mwpm = [b[2] for b in batch] if batch[0][2] is not None else None
            X = torch.stack(xs, dim=0)  # [B, N_syn]
            # Build node_inputs for our GNN forward: pack into expected flat graph format
            # Use existing collate by adapting to expected format later in loop
            return X, (None if ys_relay is None else torch.stack(ys_relay, dim=0)), (None if ys_mwpm is None else torch.stack(ys_mwpm, dim=0))
        trainloader = DataLoader(distil_ds, batch_size=batch_size, shuffle=True, collate_fn=_distil_collate)
    else:
        trainset = adapt_trainset(
            generate_syndrome_error_volume(code, error_model, p=p_train, batch_size=len_train_set,
                                           backend=backend, cudaq_mode=cudaq_mode, cudaq_cfg=cudaq_cfg),
            code, num_classes=n_node_inputs)
        trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate, shuffle=True)
    print(f"  [Curriculum] p_train={p_train:.5f}; batches={len(trainloader)}")
    
    # Initialize gradient accumulation for MGHD
    optimizer_mghd.zero_grad()
    
    for i, batch in enumerate(trainloader):
        if teacher_synd is None:
            inputs, targets, src_ids, dst_ids = batch
            inputs, targets = inputs.to(device), targets.to(device)
            src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
        else:
            # Distillation path
            X_batch, Y_relay, Y_mwpm = batch
            inputs, targets, src_ids, dst_ids = X_batch, Y_relay, Y_mwpm, None

        # --- Train the Baseline GNN ---
        optimizer_baseline.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
            if teacher_synd is None:
                outputs_baseline = gnn_baseline(inputs, src_ids, dst_ids)
                loss_baseline = criterion(outputs_baseline[-1], targets)
            else:
                # Skip baseline training when distilling from external teacher labels
                loss_baseline = torch.tensor(0.0, device=device, dtype=amp_data_type)

        if teacher_synd is None:
            scaler.scale(loss_baseline).backward()
            scaler.step(optimizer_baseline)
            scaler.update()
            epoch_loss_baseline.append(loss_baseline.item())
        else:
            # No-op baseline path during distillation
            epoch_loss_baseline.append(0.0)

        # --- Train your Hybrid MGHD ---
        with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
            # Add noise injection for regularization (optimized: 0.00544) - only first 5 epochs
            effective_noise = noise_injection if epoch < 3 else 0.0
            if teacher_synd is None and effective_noise > 0 and inputs.dtype.is_floating_point:
                noise = torch.randn_like(inputs) * effective_noise
                noisy_inputs = inputs + noise
            else:
                noisy_inputs = inputs if teacher_synd is None else None

            if teacher_synd is None:
                outputs_mghd = mghd_model(noisy_inputs, src_ids, dst_ids)
                loss_mghd = criterion(outputs_mghd[-1], targets)
            else:
                # Distillation: build node_inputs for batch=1 graphs and compute logits, then BCE/CE to teacher labels
                X_batch, Y_relay, Y_mwpm = inputs, targets, src_ids  # mapped above
                # Build graph inputs per sample (batch of independent graphs)
                num_check_nodes = (dist**2 - 1) if surface_layout == 'planar' else 8
                num_qubit_nodes = (dist**2) if surface_layout == 'planar' else 9
                nodes_per_graph = num_check_nodes + num_qubit_nodes
                B = X_batch.shape[0]
                node_inputs = torch.zeros(B * nodes_per_graph, gnn_baseline.n_node_inputs, device=device, dtype=amp_data_type)
                # Place syndrome in first feature of check nodes per graph
                for b in range(B):
                    off = b * nodes_per_graph
                    xi = X_batch[b].to(device=device, dtype=torch.uint8)
                    node_inputs[off:off+num_check_nodes, 0] = xi.to(node_inputs.dtype)
                # Use training src/dst (single graph) tiled by batch size
                og_src, og_dst = GNNDecoder.surface_code_edges
                add = nodes_per_graph
                tiled_src = torch.cat([og_src + b*add for b in range(B)]).to(device)
                tiled_dst = torch.cat([og_dst + b*add for b in range(B)]).to(device)
                outputs_mghd = mghd_model(node_inputs, tiled_src, tiled_dst)
                logits = outputs_mghd[-1].view(B, nodes_per_graph, -1)[:, num_check_nodes:, :]  # [B, N_bits, C]
                # Hard labels -> class indices per bit
                loss_terms = []
                if Y_relay is not None:
                    y = Y_relay.to(device=device, dtype=torch.long)
                    loss_terms.append(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.view(-1)))
                if Y_mwpm is not None:
                    y = Y_mwpm.to(device=device, dtype=torch.long)
                    loss_terms.append(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.view(-1)))
                loss_mghd = sum(loss_terms) / max(1, len(loss_terms)) if loss_terms else torch.tensor(0.0, device=device, dtype=amp_data_type)
            
            # Gradient accumulation (optimized: 1 step)
            loss_mghd = loss_mghd / accumulation_steps

        scaler.scale(loss_mghd).backward()
        
        # Only step optimizer every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            # IMPROVED: Add gradient clipping for stability
            scaler.unscale_(optimizer_mghd)
            
            # Debug: Check gradient norms
            total_norm = torch.nn.utils.clip_grad_norm_(mghd_model.parameters(), gradient_clip_value)
            if i == 0:  # Print once per epoch
                print(f"    MGHD grad norm: {total_norm:.6f}")
            
            scaler.step(optimizer_mghd)
            scaler.update()
            # EMA disabled - was hurting performance
            # ema_update(mghd_model)
            optimizer_mghd.zero_grad()
            
            # Warmup for first few steps
            current_global_step += 1
            if current_global_step < warmup_steps:
                warmup_scheduler_mghd.step()
        
        epoch_loss_mghd.append(loss_mghd.item() * accumulation_steps)  # Scale back for logging
        
        # Progress indicator every 20 batches
        if (i + 1) % 20 == 0:
            print(f"  Batch {i+1}/{len(trainloader)} completed")
            sys.stdout.flush()

    # Honor a cap on the number of steps per epoch (0 = use all batches)
    if steps_per_epoch > 0 and (i + 1) >= steps_per_epoch:
        print(f"  Reached steps-per-epoch limit ({steps_per_epoch}); breaking.")
        break

    # Calculate average losses
    avg_loss_baseline = np.mean(epoch_loss_baseline)
    avg_loss_mghd = np.mean(epoch_loss_mghd)

    print(f"Epoch {epoch+1} - Evaluating models...")
    
    # --- After each epoch, evaluate both models ---
    gnn_baseline.eval()
    mghd_model.eval()

    with torch.no_grad():
        # Baseline eval
        lerx_baseline, lerz_baseline, ler_tot_baseline = evaluate_model_avg(gnn_baseline, testloader, code, runs=eval_runs)

        # MGHD eval with raw training weights (EMA disabled - was hurting performance)
        lerx_mghd, lerz_mghd, ler_tot_mghd = evaluate_model_avg(mghd_model, testloader, code, runs=eval_runs)
        
        # Calculate fraction solved
        frac_solved_baseline = fraction_of_solved_puzzles(gnn_baseline, testloader, code)
        frac_solved_mghd = fraction_of_solved_puzzles(mghd_model, testloader, code)

    # Update best checkpoint if MGHD improved
    if ler_tot_mghd + early_stop_min_delta < best_mghd_ler:
        best_mghd_ler = ler_tot_mghd
        # Save raw parameters with timestamped filename
        best_mghd_state = {k: v.cpu() for k, v in mghd_model.state_dict().items()}
        best_epoch = epoch + 1
        model_path = os.path.join(output_dir, f'{run_id}_mghd_best.pt')
        torch.save(best_mghd_state, model_path)
        epochs_without_improve = 0
    else:
        epochs_without_improve += 1

    # Per-epoch multi-p evaluation (fast: runs=1, set_size=5000)
    # Baseline
    baseline_pgrid, baseline_auc = evaluate_over_p_grid(gnn_baseline, code, runs=1, set_size=5000)
    with open(pgrid_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        for (pp, lx, lz, lt) in baseline_pgrid:
            writer.writerow([datetime.utcnow().isoformat(), epoch + 1, 'baseline', f"{pp:.5f}", f"{lt:.6f}", f"{lx:.6f}", f"{lz:.6f}"])
        writer.writerow([datetime.utcnow().isoformat(), epoch + 1, 'baseline', 'AUC', f"{baseline_auc:.6f}", '', ''])

    # MGHD (raw weights - EMA disabled)
    mghd_pgrid, mghd_auc = evaluate_over_p_grid(mghd_model, code, runs=1, set_size=5000)
    with open(pgrid_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        for (pp, lx, lz, lt) in mghd_pgrid:
            writer.writerow([datetime.utcnow().isoformat(), epoch + 1, 'mghd_raw', f"{pp:.5f}", f"{lt:.6f}", f"{lx:.6f}", f"{lz:.6f}"])
        writer.writerow([datetime.utcnow().isoformat(), epoch + 1, 'mghd_raw', 'AUC', f"{mghd_auc:.6f}", '', ''])

    # Check for early stopping
    if epochs_without_improve >= early_stop_patience:
        print(f"Early stopping triggered at epoch {epoch+1} "
              f"(no improvement for {early_stop_patience} epochs).")
        break

    # Store metrics for plotting
    baseline_losses.append(avg_loss_baseline)
    baseline_lers.append(ler_tot_baseline)
    baseline_lerx.append(lerx_baseline)
    baseline_lerz.append(lerz_baseline)
    baseline_frac_solved.append(frac_solved_baseline)
    
    mghd_losses.append(avg_loss_mghd)
    mghd_lers.append(ler_tot_mghd)
    mghd_lerx.append(lerx_mghd)
    mghd_lerz.append(lerz_mghd)
    mghd_frac_solved.append(frac_solved_mghd)

    # CSV logging
    current_lr_mghd = optimizer_mghd.param_groups[0]['lr']
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(), epoch + 1,
            f"{avg_loss_baseline:.6f}", f"{ler_tot_baseline:.6f}", f"{lerx_baseline:.6f}", f"{lerz_baseline:.6f}", f"{frac_solved_baseline:.6f}",
            f"{avg_loss_mghd:.6f}", f"{ler_tot_mghd:.6f}", f"{lerx_mghd:.6f}", f"{lerz_mghd:.6f}", f"{frac_solved_mghd:.6f}",
            f"{current_lr_mghd:.8f}"
        ])

    # Update learning rate schedulers
    scheduler_baseline.step(ler_tot_baseline)
    scheduler_mghd.step()  # LambdaLR keeps constant LR

    # Calculate epoch time
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)

    # Detailed output
    print(f"\n=== EPOCH {epoch+1} RESULTS ===")
    print(f"Training time: {epoch_time:.2f}s")
    print(f"Baseline GNN:")
    print(f"  - Train Loss: {avg_loss_baseline:.6f}")
    print(f"  - LER_X: {lerx_baseline:.6f}")
    print(f"  - LER_Z: {lerz_baseline:.6f}")
    print(f"  - LER_Total: {ler_tot_baseline:.6f}")
    print(f"  - Fraction Solved: {frac_solved_baseline:.6f}")
    
    print(f"Hybrid MGHD:")
    print(f"  - Train Loss: {avg_loss_mghd:.6f}")
    print(f"  - LER_X: {lerx_mghd:.6f}")
    print(f"  - LER_Z: {lerz_mghd:.6f}")
    print(f"  - LER_Total: {ler_tot_mghd:.6f}")
    print(f"  - Fraction Solved: {frac_solved_mghd:.6f}")
    print(f"  - Best-so-far MGHD LER: {best_mghd_ler:.6f} (epoch {best_epoch if best_epoch else '-'})")
    
    # Comparison summary
    if ler_tot_mghd < ler_tot_baseline:
        print(f" - MGHD BETTER by {(ler_tot_baseline - ler_tot_mghd):.6f}")
    else:
        print(f" - Baseline BETTER by {(ler_tot_mghd - ler_tot_baseline):.6f}")
    
    print("=" * 50)
    
    # CSV-style output for easy parsing
    print(f"{epoch+1}, {avg_loss_baseline:.6f}, {ler_tot_baseline:.6f}, {avg_loss_mghd:.6f}, {ler_tot_mghd:.6f}, {epoch_time:.2f}")
    sys.stdout.flush()

# Training completed
total_time = time.time() - start_time

# Restore best model
if best_mghd_state is not None:
    mghd_model.load_state_dict(best_mghd_state)
    print(f"Restored MGHD to best epoch {best_epoch} with LER {best_mghd_ler:.6f}")
    print(f"Saved best MGHD weights to {os.path.join(output_dir, f'{run_id}_mghd_best.pt')}")
    print(f"Appended per-epoch metrics to {csv_path}")
    
    # Final averaged evaluation on best checkpoint
    final_lerx, final_lerz, final_lertot = evaluate_model_avg(mghd_model, testloader, code, runs=final_eval_runs)
    print(f"Final (best-checkpoint) averaged over {final_eval_runs} runs -> "
          f"LER_Total: {final_lertot:.6f} (LER_X: {final_lerx:.6f}, LER_Z: {final_lerz:.6f})")
    
    # Append final summary row to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(), 'FINAL_BEST',
            '', '', '', '', '',
            '', f"{final_lertot:.6f}", f"{final_lerx:.6f}", f"{final_lerz:.6f}", '',
            optimizer_mghd.param_groups[0]['lr'] if optimizer_mghd.param_groups else ''
        ])
    
    # Final multi-p evaluation with stronger averaging (raw weights)
    final_grid, final_auc = evaluate_over_p_grid(mghd_model, code, runs=3, set_size=20000)
    with open(pgrid_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        for (pp, lx, lz, lt) in final_grid:
            writer.writerow([datetime.utcnow().isoformat(), 'FINAL_BEST', 'mghd_raw', f"{pp:.5f}", f"{lt:.6f}", f"{lx:.6f}", f"{lz:.6f}"])
        writer.writerow([datetime.utcnow().isoformat(), 'FINAL_BEST', 'mghd_raw', 'AUC', f"{final_auc:.6f}", '', ''])
    print(f"Final MGHD AUC over p-grid: {final_auc:.6f}")
    
    print(f"Best epoch: {best_epoch}, Best LER (avg {eval_runs}): {best_mghd_ler:.6f}")
    print(f"Best weights: {os.path.join(output_dir, f'{run_id}_mghd_best.pt')}")
    print(f"Per-epoch metrics: {csv_path}")
    print(f"Multi-p metrics: {pgrid_csv}")

print(f"\nTraining completed in {total_time:.2f}s")
print(f"Average time per epoch: {np.mean(epoch_times):.2f}s")

# Final comparison - use best performance for both models
final_baseline_ler = min(baseline_lers)  # Best baseline performance
final_mghd_ler = best_mghd_ler  # Best MGHD performance (already tracked)
print(f"\n - FINAL COMPARISON (Best Performance):")
print(f"Baseline GNN - Best LER: {final_baseline_ler:.6f}")
print(f"Hybrid MGHD - Best LER: {final_mghd_ler:.6f}")

if final_mghd_ler < final_baseline_ler:
    improvement = (final_baseline_ler - final_mghd_ler) / final_baseline_ler * 100
    print(f" - MGHD is {improvement:.2f}% better than baseline!")
else:
    degradation = (final_mghd_ler - final_baseline_ler) / final_baseline_ler * 100
    print(f" - MGHD is {degradation:.2f}% worse than baseline")

# Also show final epoch comparison for reference
print(f"\n - Final Epoch Comparison (before early stopping):")
print(f"Baseline GNN - Final Epoch LER: {baseline_lers[-1]:.6f}")
print(f"Hybrid MGHD - Final Epoch LER: {mghd_lers[-1]:.6f}")

# ===============================================
# PLOTTING SECTION
# ===============================================

print("\n - Generating comparison plots...")

# Use actual number of completed epochs for plotting
actual_epochs = len(baseline_lers)
epochs_range = range(1, actual_epochs+1)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'MGHD w/ Channel Attention vs Baseline GNN Comparison\nDistance {d}, {actual_epochs} epochs (early stopped), SE reduction={mamba_params["se_reduction"]}\nRun: {run_timestamp}', fontsize=16)

# Plot 1: Logical Error Rate
axes[0, 0].plot(epochs_range, baseline_lers, 'b-o', label='Baseline GNN', linewidth=2, markersize=6)
axes[0, 0].plot(epochs_range, mghd_lers, 'r-s', label='Hybrid MGHD', linewidth=2, markersize=6)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Logical Error Rate')
axes[0, 0].set_title('Logical Error Rate Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Plot 2: Training Loss
axes[0, 1].plot(epochs_range, baseline_losses, 'b-o', label='Baseline GNN', linewidth=2, markersize=6)
axes[0, 1].plot(epochs_range, mghd_losses, 'r-s', label='Hybrid MGHD', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Training Loss')
axes[0, 1].set_title('Training Loss Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Fraction of Solved Syndromes
axes[1, 0].plot(epochs_range, baseline_frac_solved, 'b-o', label='Baseline GNN', linewidth=2, markersize=6)
axes[1, 0].plot(epochs_range, mghd_frac_solved, 'r-s', label='Hybrid MGHD', linewidth=2, markersize=6)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Fraction of Solved Syndromes')
axes[1, 0].set_title('Fraction of Solved Syndromes')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: LER_X and LER_Z breakdown
axes[1, 1].plot(epochs_range, baseline_lerx, 'b-', label='Baseline LER_X', linewidth=2, alpha=0.7)
axes[1, 1].plot(epochs_range, baseline_lerz, 'b--', label='Baseline LER_Z', linewidth=2, alpha=0.7)
axes[1, 1].plot(epochs_range, mghd_lerx, 'r-', label='MGHD LER_X', linewidth=2, alpha=0.7)
axes[1, 1].plot(epochs_range, mghd_lerz, 'r--', label='MGHD LER_Z', linewidth=2, alpha=0.7)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Logical Error Rate')
axes[1, 1].set_title('LER_X and LER_Z Breakdown')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_yscale('log')

plt.tight_layout()
comparison_plot_path = os.path.join(output_dir, f'{run_id}_comparison_plots.png')
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Create a summary table plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Create summary data
summary_data = [
    ['Model', 'Parameters', 'Best LER', 'Final Loss', 'Final Epoch LER', 'Fraction Solved'],
    ['Baseline GNN', f'{sum(p.numel() for p in gnn_baseline.parameters()):,}', 
     f'{min(baseline_lers):.6f}', f'{baseline_losses[-1]:.6f}', 
     f'{baseline_lers[-1]:.6f}', f'{baseline_frac_solved[-1]:.4f}'],
    ['Hybrid MGHD', f'{sum(p.numel() for p in mghd_model.parameters()):,}', 
     f'{best_mghd_ler:.6f}', f'{mghd_losses[-1]:.6f}', 
     f'{mghd_lers[-1]:.6f}', f'{mghd_frac_solved[-1]:.4f}']
]

table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)

# Color code the table
for i in range(len(summary_data[0])):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight better performance
better_ler = 1 if final_mghd_ler < final_baseline_ler else 2
table[(better_ler, 2)].set_facecolor('#E8F5E8')

plt.title(f'Model Comparison Summary - {run_timestamp}', fontsize=16, fontweight='bold', pad=20)
summary_plot_path = os.path.join(output_dir, f'{run_id}_summary_table.png')
plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Save metrics to file
metrics_data = {
    'run_id': run_id,
    'timestamp': run_timestamp,
    'hyperparameters': {
        'd_model': mamba_params['d_model'],
        'd_state': mamba_params['d_state'],
        'se_reduction': mamba_params['se_reduction'],
        'lr': lr,
        'label_smoothing': label_smoothing,
        'early_stop_patience': early_stop_patience
    },
    'epochs': list(epochs_range),
    'baseline_losses': baseline_losses,
    'baseline_lers': baseline_lers,
    'baseline_lerx': baseline_lerx,
    'baseline_lerz': baseline_lerz,
    'baseline_frac_solved': baseline_frac_solved,
    'mghd_losses': mghd_losses,
    'mghd_lers': mghd_lers,
    'mghd_lerx': mghd_lerx,
    'mghd_lerz': mghd_lerz,
    'mghd_frac_solved': mghd_frac_solved,
    'epoch_times': epoch_times,
    'best_mghd_ler': best_mghd_ler,
    'best_epoch': best_epoch,
    'final_baseline_ler': final_baseline_ler,
    'final_mghd_ler': final_mghd_ler
}

metrics_path = os.path.join(output_dir, f'{run_id}_metrics.npy')
np.save(metrics_path, metrics_data)
print(f"\n - Metrics saved to '{metrics_path}'")
print(f" - Plots saved as '{comparison_plot_path}' and '{summary_plot_path}'")
print(f"\n - Proof of Concept completed successfully!")
print(f"\n=== RUN SUMMARY ===")
print(f"Run ID: {run_id}")
print(f"Output Directory: {output_dir}")
print(f"Files Generated:")
print(f"  - Training metrics: {os.path.basename(csv_path)}")
print(f"  - Multi-p metrics: {os.path.basename(pgrid_csv)}")
print(f"  - Best model weights: {run_id}_mghd_best.pt")
print(f"  - Comparison plots: {run_id}_comparison_plots.png")
print(f"  - Summary table: {run_id}_summary_table.png")
print(f"  - Full metrics: {run_id}_metrics.npy")
print(f"===================")
