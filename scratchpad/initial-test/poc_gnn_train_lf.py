
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
from datetime import datetime

# --- NEW IMPORTS ---
from poc_my_models import MGHD
from poc_loss_functions import UnifiedLoss # Import our new unified loss

# --- Original Imports ---
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


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plots_data_folder = "Plots and Data"
os.makedirs(plots_data_folder, exist_ok=True)

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
n_iters = 5
n_node_features = 64
n_edge_features = 64
len_test_set = 5000
test_err_rate = 0.05
len_train_set = 20000
max_train_err_rate = 0.1
lr = 0.0001
weight_decay = 0.0001
msg_net_size = 128
msg_net_dropout_p = 0.05
gru_dropout_p = 0.05

# Define the Mamba parameters
mamba_params = {
    'd_model': 64,
    'd_state': 16,
    'd_conv': 4,
    'expand': 2
}

print("n_iters: ", n_iters, "n_node_outputs: ", n_node_outputs, "n_node_features: ", n_node_features,
      "n_edge_features: ", n_edge_features)
print("msg_net_size: ", msg_net_size, "msg_net_dropout_p: ", msg_net_dropout_p, "gru_dropout_p: ", gru_dropout_p)
print("learning rate: ", lr, "weight decay: ", weight_decay, "len train set: ", len_train_set, 'max train error rate: ',
      max_train_err_rate, "len test set: ", len_test_set, "test error rate: ", test_err_rate)

"""
Create the Surface code
"""
dist = d
print('PoC Target: Surface Code d=', dist)
code = surface_2d.RotatedPlanar2DCode(dist)

src, tgt = surface_code_edges(code)
src_tensor = torch.LongTensor(src)
tgt_tensor = torch.LongTensor(tgt)
GNNDecoder.surface_code_edges = (src_tensor, tgt_tensor)
GNNDecoder.device = device

# ---------------------------
print("Initializing models for comparison...")

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

"""
Train
"""
# --- LOSS FUNCTION SETUP ---
# To test different loss functions, change the lambda values here.
# Step 1: Baseline (lambda=0.0)
# Step 2: Add Syndrome Loss (lambda > 0)
LAMBDA_SYNDROME = 0.5 # Example value to turn on the syndrome loss
BETA_LOGICAL = 0.0    # Keep this at 0.0 for now

# Instantiate our new unified loss function
unified_loss = UnifiedLoss(
    code=code, 
    device=device,
    lambda_syndrome=LAMBDA_SYNDROME,
    beta_logical=BETA_LOGICAL
)
print(f"Unified Loss activated with: lambda_syndrome={LAMBDA_SYNDROME}, beta_logical={BETA_LOGICAL}")

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

epochs = 5
batch_size = 128

start_time = time.time()

""" generate training data """
trainset = adapt_trainset(
    generate_syndrome_error_volume(code, error_model, p=max_train_err_rate, batch_size=len_train_set),
    code, num_classes=n_node_inputs)
trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate, shuffle=False)

# Initialize lists to store metrics for plotting
baseline_losses, baseline_lers, baseline_lerx, baseline_lerz, baseline_frac_solved = [], [], [], [], []
mghd_losses, mghd_lers, mghd_lerx, mghd_lerz, mghd_frac_solved = [], [], [], [], []
epoch_times = []

print("Starting training...")

for epoch in range(epochs):
    epoch_start_time = time.time()
    gnn_baseline.train()
    mghd_model.train()
    
    epoch_loss_baseline, epoch_loss_mghd = [], []

    for i, (inputs, targets, src_ids, dst_ids) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)

        # --- Train the Baseline GNN ---
        optimizer_baseline.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
            outputs_baseline = gnn_baseline(inputs, src_ids, dst_ids)
            loss_baseline = unified_loss(outputs_baseline, targets, inputs)
        
        scaler.scale(loss_baseline).backward()
        scaler.step(optimizer_baseline)
        scaler.update()
        epoch_loss_baseline.append(loss_baseline.item())

        # --- Train your Hybrid MGHD ---
        optimizer_mghd.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
            outputs_mghd = mghd_model(inputs, src_ids, dst_ids)
            loss_mghd = unified_loss(outputs_mghd, targets, inputs)

        scaler.scale(loss_mghd).backward()
        scaler.step(optimizer_mghd)
        scaler.update()
        epoch_loss_mghd.append(loss_mghd.item())

    avg_loss_baseline = np.mean(epoch_loss_baseline)
    avg_loss_mghd = np.mean(epoch_loss_mghd)

    # --- After each epoch, evaluate both models ---
    gnn_baseline.eval()
    mghd_model.eval()
    with torch.no_grad():
        lerx_baseline, lerz_baseline, ler_tot_baseline = logical_error_rate(gnn_baseline, testloader, code)
        lerx_mghd, lerz_mghd, ler_tot_mghd = logical_error_rate(mghd_model, testloader, code)
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
    
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)

    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"  Baseline GNN -> Loss: {avg_loss_baseline:.4f}, LER: {ler_tot_baseline:.4f}")
    print(f"  Hybrid MGHD  -> Loss: {avg_loss_mghd:.4f}, LER: {ler_tot_mghd:.4f}")
    print("-" * 20)

# ===============================================
# FINAL REPORTING AND PLOTTING
# ===============================================

total_time = time.time() - start_time
print(f"\nTraining completed in {total_time:.2f}s")
print(f"Average time per epoch: {np.mean(epoch_times):.2f}s")

# Final comparison
final_baseline_ler = baseline_lers[-1]
final_mghd_ler = mghd_lers[-1]
print(f"\n-------------------------------\nFINAL COMPARISON:")
print(f"Baseline GNN - Final LER: {final_baseline_ler:.6f}")
print(f"Hybrid MGHD - Final LER: {final_mghd_ler:.6f}")

if final_mghd_ler < final_baseline_ler and final_baseline_ler > 0:
    improvement = (final_baseline_ler - final_mghd_ler) / final_baseline_ler * 100
    print(f"\nMGHD is {improvement:.2f}% better than baseline!")
else:
    print("\nMGHD did not show improvement over baseline in this run.")
print("-------------------------------")


print("\nGenerating comparison plots...")

epochs_range = range(1, epochs + 1)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'Mamba-Graph Hybrid Decoder vs Baseline GNN Comparison\nDistance {d}, {epochs} epochs', fontsize=16)

# Plot 1: Logical Error Rate
axes[0, 0].plot(epochs_range, baseline_lers, 'b-o', label='Baseline GNN', linewidth=2, markersize=4)
axes[0, 0].plot(epochs_range, mghd_lers, 'r-s', label='Hybrid MGHD', linewidth=2, markersize=4)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Logical Error Rate')
axes[0, 0].set_title('Logical Error Rate Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
axes[0, 0].set_yscale('log')

# Plot 2: Training Loss
axes[0, 1].plot(epochs_range, baseline_losses, 'b-o', label='Baseline GNN', linewidth=2, markersize=4)
axes[0, 1].plot(epochs_range, mghd_losses, 'r-s', label='Hybrid MGHD', linewidth=2, markersize=4)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Training Loss')
axes[0, 1].set_title('Training Loss Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot 3: Fraction of Solved Syndromes
axes[1, 0].plot(epochs_range, baseline_frac_solved, 'b-o', label='Baseline GNN', linewidth=2, markersize=4)
axes[1, 0].plot(epochs_range, mghd_frac_solved, 'r-s', label='Hybrid MGHD', linewidth=2, markersize=4)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Fraction of Solved Syndromes')
axes[1, 0].set_title('Fraction of Solved Syndromes')
axes[1, 0].legend()
axes[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot 4: LER_X and LER_Z breakdown
axes[1, 1].plot(epochs_range, baseline_lerx, 'b-', label='Baseline LER_X', linewidth=2, alpha=0.7)
axes[1, 1].plot(epochs_range, baseline_lerz, 'b--', label='Baseline LER_Z', linewidth=2, alpha=0.7)
axes[1, 1].plot(epochs_range, mghd_lerx, 'r-', label='MGHD LER_X', linewidth=2, alpha=0.7)
axes[1, 1].plot(epochs_range, mghd_lerz, 'r--', label='MGHD LER_Z', linewidth=2, alpha=0.7)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Logical Error Rate')
axes[1, 1].set_title('LER_X and LER_Z Breakdown')
axes[1, 1].legend()
axes[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
axes[1, 1].set_yscale('log')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
comparison_plot_filename = os.path.join(plots_data_folder, f'mghd_vs_baseline_comparison_d{d}_{error_model_name}_epochs{epochs}_{timestamp}.png')
plt.savefig(comparison_plot_filename, dpi=300, bbox_inches='tight')
plt.show()

# Create a summary table plot
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('tight')
ax.axis('off')

# Create summary data
summary_data = [
    ['Model', 'Parameters', 'Final LER', 'Final Loss', 'Best LER', 'Fraction Solved'],
    ['Baseline GNN', f'{sum(p.numel() for p in gnn_baseline.parameters()):,}', 
     f'{final_baseline_ler:.6f}', f'{baseline_losses[-1]:.6f}', 
     f'{min(baseline_lers) if baseline_lers else "N/A":.6f}', f'{baseline_frac_solved[-1]:.4f}'],
    ['Hybrid MGHD', f'{sum(p.numel() for p in mghd_model.parameters()):,}', 
     f'{final_mghd_ler:.6f}', f'{mghd_losses[-1]:.6f}', 
     f'{min(mghd_lers) if mghd_lers else "N/A":.6f}', f'{mghd_frac_solved[-1]:.4f}']
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
if final_mghd_ler < final_baseline_ler:
    table[(2, 2)].set_facecolor('#E8F5E9') # Highlight MGHD
else:
    table[(1, 2)].set_facecolor('#FFEBEE') # Highlight Baseline

plt.title('Model Comparison Summary', fontsize=16, fontweight='bold', pad=20)
summary_plot_filename = os.path.join(plots_data_folder, f'mghd_vs_baseline_summary_d{d}_{error_model_name}_epochs{epochs}_{timestamp}.png')
plt.savefig(summary_plot_filename, dpi=300, bbox_inches='tight')
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

metrics_filename = os.path.join(plots_data_folder, f'mghd_vs_baseline_metrics_d{d}_{error_model_name}_epochs{epochs}_{timestamp}.npy')
np.save(metrics_filename, metrics_data)
print(f"\nMetrics saved to '{metrics_filename}'")
print(f"Comparison plot saved as '{comparison_plot_filename}'")
print(f"Summary plot saved as '{summary_plot_filename}'")
print(f"All files saved in '{plots_data_folder}' folder")
print(f"\nProof of Concept completed successfully!")
