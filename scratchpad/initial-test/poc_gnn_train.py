from panqec.codes import surface_2d
from panqec.error_models import PauliErrorModel
from panqec.decoders import MatchingDecoder, BeliefPropagationOSDDecoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

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

# list of hyperparameters
n_node_inputs = 4
n_node_outputs = 4
n_iters = 3
n_node_features = 64
n_edge_features = 64
len_test_set = 5000 # original len_test_set = 10
test_err_rate = 0.05
len_train_set = 20000 # original len_test_set * 10
max_train_err_rate = 0.15
lr = 0.0001
weight_decay = 0.0001
msg_net_size = 128 # original msg_net_size = 512
msg_net_dropout_p = 0.05
gru_dropout_p = 0.05

# Define the Mamba parameters
mamba_params = {
    'd_model': 64, # The main feature dimension for Mamba
    'd_state': 16, # The size of the hidden state (a standard value)
    'd_conv': 4, # The convolution kernel size (a standard value)
    'expand': 2 # The expansion factor (a standard value)
}

print("n_iters: ", n_iters, "n_node_outputs: ", n_node_outputs, "n_node_features: ", n_node_features,
      "n_edge_features: ", n_edge_features)
print("msg_net_size: ", msg_net_size, "msg_net_dropout_p: ", msg_net_dropout_p, "gru_dropout_p: ", gru_dropout_p)
print("learning rate: ", lr, "weight decay: ", weight_decay, "len train set: ", len_train_set, 'max train error rate: ',
      max_train_err_rate, "len test set: ", len_test_set, "test error rate: ", test_err_rate)

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
print(f"Baseline GNN parameters: {sum(p.numel() for p in gnn_baseline.parameters())}")

# 2. Create your new Hybrid MGHD Model
mghd_model = MGHD(gnn_params=gnn_params, mamba_params=mamba_params).to(device)
optimizer_mghd = optim.AdamW(mghd_model.parameters(), lr=lr, weight_decay=weight_decay)
print(f"Hybrid MGHD parameters: {sum(p.numel() for p in mghd_model.parameters())}")

# Generate the test data
testset = adapt_trainset(
    generate_syndrome_error_volume(code, error_model=error_model, p=test_err_rate, batch_size=len_test_set,
                                   for_training=False), code,
    num_classes=n_node_inputs, for_training=False)
testloader = DataLoader(testset, batch_size=512, collate_fn=collate, shuffle=False)

print(f"Dataset size: {len_train_set}")
print(f"Test set size: {len(testset)}")

"""
Train
"""
""" automatic mixed precision """
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

epochs = 50  # Reduced for PoC
batch_size = 128
criterion = nn.CrossEntropyLoss()

start_time = time.time()
size = 2 * GNNDecoder.dist ** 2 - 1
error_index = GNNDecoder.dist ** 2 - 1

""" generate training data """
trainset = adapt_trainset(
    generate_syndrome_error_volume(code, error_model, p=max_train_err_rate, batch_size=len_train_set),
    code, num_classes=n_node_inputs)
trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate, shuffle=False)

print(f"Number of batches per epoch: {len(trainloader)}")
print("=" * 60)

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

# Modified training loop to compare both models
print("Starting training...")
print("epoch, baseline_loss, baseline_LER, mghd_loss, mghd_LER, train_time")
sys.stdout.flush()

for epoch in range(epochs):
    epoch_start_time = time.time()
    
    # Set both models to training mode
    gnn_baseline.train()
    mghd_model.train()

    epoch_loss_baseline = []
    epoch_loss_mghd = []

    # Progress indicator
    print(f"\nEpoch {epoch+1}/{epochs} - Training...")
    
    for i, (inputs, targets, src_ids, dst_ids) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)

        # --- Train the Baseline GNN ---
        optimizer_baseline.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
            outputs_baseline = gnn_baseline(inputs, src_ids, dst_ids)
            loss_baseline = criterion(outputs_baseline[-1], targets)

        scaler.scale(loss_baseline).backward()
        scaler.step(optimizer_baseline)
        scaler.update()
        epoch_loss_baseline.append(loss_baseline.item())

        # --- Train your Hybrid MGHD ---
        optimizer_mghd.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
            outputs_mghd = mghd_model(inputs, src_ids, dst_ids)
            loss_mghd = criterion(outputs_mghd[-1], targets)

        scaler.scale(loss_mghd).backward()
        scaler.step(optimizer_mghd)
        scaler.update()
        epoch_loss_mghd.append(loss_mghd.item())
        
        # Progress indicator every 20 batches
        if (i + 1) % 20 == 0:
            print(f"  Batch {i+1}/{len(trainloader)} completed")
            sys.stdout.flush()

    # Calculate average losses
    avg_loss_baseline = np.mean(epoch_loss_baseline)
    avg_loss_mghd = np.mean(epoch_loss_mghd)

    print(f"Epoch {epoch+1} - Evaluating models...")
    
    # --- After each epoch, evaluate both models ---
    gnn_baseline.eval()
    mghd_model.eval()

    with torch.no_grad():
        # Calculate detailed metrics
        lerx_baseline, lerz_baseline, ler_tot_baseline = logical_error_rate(gnn_baseline, testloader, code)
        lerx_mghd, lerz_mghd, ler_tot_mghd = logical_error_rate(mghd_model, testloader, code)
        
        # Calculate fraction solved
        frac_solved_baseline = fraction_of_solved_puzzles(gnn_baseline, testloader, code)
        frac_solved_mghd = fraction_of_solved_puzzles(mghd_model, testloader, code)

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
print(f"\nTraining completed in {total_time:.2f}s")
print(f"Average time per epoch: {np.mean(epoch_times):.2f}s")

# Final comparison
final_baseline_ler = baseline_lers[-1]
final_mghd_ler = mghd_lers[-1]
print(f"\n - FINAL COMPARISON:")
print(f"Baseline GNN - Final LER: {final_baseline_ler:.6f}")
print(f"Hybrid MGHD - Final LER: {final_mghd_ler:.6f}")

if final_mghd_ler < final_baseline_ler:
    improvement = (final_baseline_ler - final_mghd_ler) / final_baseline_ler * 100
    print(f" - MGHD is {improvement:.2f}% better than baseline!")
else:
    degradation = (final_mghd_ler - final_baseline_ler) / final_baseline_ler * 100
    print(f" - MGHD is {degradation:.2f}% worse than baseline")

# ===============================================
# PLOTTING SECTION
# ===============================================

print("\n - Generating comparison plots...")

epochs_range = range(1, epochs+1)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'Mamba-Graph Hybrid Decoder vs Baseline GNN Comparison\nDistance {d}, {epochs} epochs', fontsize=16)

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
plt.savefig('mghd_vs_baseline_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a summary table plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Create summary data
summary_data = [
    ['Model', 'Parameters', 'Final LER', 'Final Loss', 'Best LER', 'Fraction Solved'],
    ['Baseline GNN', f'{sum(p.numel() for p in gnn_baseline.parameters()):,}', 
     f'{final_baseline_ler:.6f}', f'{baseline_losses[-1]:.6f}', 
     f'{min(baseline_lers):.6f}', f'{baseline_frac_solved[-1]:.4f}'],
    ['Hybrid MGHD', f'{sum(p.numel() for p in mghd_model.parameters()):,}', 
     f'{final_mghd_ler:.6f}', f'{mghd_losses[-1]:.6f}', 
     f'{min(mghd_lers):.6f}', f'{mghd_frac_solved[-1]:.4f}']
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

plt.title('Model Comparison Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig('mghd_vs_baseline_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# Save metrics to file
metrics_data = {
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
    'epoch_times': epoch_times
}

np.save('mghd_vs_baseline_metrics.npy', metrics_data)
print(f"\n - Metrics saved to 'mghd_vs_baseline_metrics.npy'")
print(f" - Plots saved as 'mghd_vs_baseline_comparison.png' and 'mghd_vs_baseline_summary.png'")
print(f"\n - Proof of Concept completed successfully!")
