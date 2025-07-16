# Quick summary: 
This is the Astra's repositiry in which I added my poc_my_models.py and poc_gnn_train.py files with poc meaning (proof-of-concept) to quicky test if our proposed mamba-GNN architechture really would have any advantagr or not. The poc_my_models.py contains a fake mamba class which is currently for mac developement purposes, it has to be commented out and **from mamba_ssm import Mamba** has to be uncommented. 

**MGHD Class:** is the Hybrid architecture combining Mamba + GNN
MGHD processes spatial sequences (not temporal data) through Mamba. This is for quick testing purposes for now.
Spatial error patterns are then processed by the GNN on the code's Tanner graph. This creates a hybrid spatio-temporal processing approach where Mamba handles sequential dependencies between check nodes.

Added features to poc_gnn_train.py (Training and Comparison Script):
1. Side-by-side training of baseline GNN and MGHD
2. Comprehensive metrics: LER_X, LER_Z, LER_Total, fraction solved
3. Automatic plotting after training (4 comparison plots + summary table)
4. Progress tracking with detailed epoch information
5. Metric storage to .npy files for later analysis
6. Generated Outputs:
    mghd_vs_baseline_comparison.png - Main comparison plots
    mghd_vs_baseline_summary.png - Performance summary table
    mghd_vs_baseline_metrics.npy - All training metrics



# GPU Cluster Deployment Instructions

## Step 1: Environment Setup
### Create virtual environment
python -m venv venv_gpu
source venv_gpu/bin/activate

### Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


### Install other dependencies
pip install numpy matplotlib panqec ldpc




## Step 2: Code Modifications for GPU
In **poc_my_models.py** :

### Replace the fake Mamba class with:
from mamba_ssm import Mamba

### Remove the entire fake Mamba class definition




## Step 3: Verification Script
Add to **poc_gnn_train.py**

print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


## Step 4: Launch Training

### Single GPU
python poc_gnn_train.py

### Multiple GPUs (optional)
torchrun --nproc_per_node=4 poc_gnn_train.py
