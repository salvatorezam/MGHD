#!/usr/bin/env python3
"""
Generate MGHD Architecture Diagram using matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_mghd_diagram():
    """Create a clear architectural diagram of MGHD dependencies"""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Colors for different components
    colors = {
        'core': '#e1f5fe',      # Light blue
        'model': '#f3e5f5',     # Light purple
        'data': '#e8f5e8',      # Light green
        'external': '#fff3e0',   # Light orange
        'special': '#fce4ec'     # Light pink
    }

    # Main training file (top)
    main_box = FancyBboxPatch((6, 9.5), 4, 1.5, boxstyle="round,pad=0.1",
                             facecolor=colors['core'], edgecolor='navy', linewidth=2)
    ax.add_patch(main_box)
    ax.text(8, 10.2, 'unified_mghd_optimizer.py', ha='center', va='center',
           fontsize=12, fontweight='bold')
    ax.text(8, 9.8, 'Main Training Entry Point\nOptuna + CUDA-Q Training', ha='center', va='center',
           fontsize=10)

    # Model layer
    model_box1 = FancyBboxPatch((2, 7), 3.5, 1.2, boxstyle="round,pad=0.1",
                               facecolor=colors['model'], edgecolor='purple', linewidth=2)
    ax.add_patch(model_box1)
    ax.text(3.75, 7.6, 'poc_my_models.py', ha='center', va='center',
           fontsize=11, fontweight='bold')
    ax.text(3.75, 7.3, 'MGHD Model\nMamba + GNN', ha='center', va='center', fontsize=9)

    model_box2 = FancyBboxPatch((10.5, 7), 3.5, 1.2, boxstyle="round,pad=0.1",
                               facecolor=colors['model'], edgecolor='purple', linewidth=2)
    ax.add_patch(model_box2)
    ax.text(12.25, 7.6, 'panq_functions.py', ha='center', va='center',
           fontsize=11, fontweight='bold')
    ax.text(12.25, 7.3, 'GNNDecoder + Utils\nTraining Functions', ha='center', va='center', fontsize=9)

    # Data layer
    data_box1 = FancyBboxPatch((1, 4.5), 4, 1.2, boxstyle="round,pad=0.1",
                              facecolor=colors['data'], edgecolor='green', linewidth=2)
    ax.add_patch(data_box1)
    ax.text(3, 5.1, 'tools/cudaq_sampler.py', ha='center', va='center',
           fontsize=11, fontweight='bold')
    ax.text(3, 4.8, 'CUDA-Q Data Generation\nFallback to numpy+LUT', ha='center', va='center', fontsize=9)

    data_box2 = FancyBboxPatch((11, 4.5), 4, 1.2, boxstyle="round,pad=0.1",
                              facecolor=colors['data'], edgecolor='green', linewidth=2)
    ax.add_patch(data_box2)
    ax.text(13, 5.1, 'tools/eval_ler.py', ha='center', va='center',
           fontsize=11, fontweight='bold')
    ax.text(13, 4.8, 'LER Evaluation\nWilson CIs + Latency', ha='center', va='center', fontsize=9)

    # External libraries
    ext_boxes = []
    ext_labels = ['PyTorch', 'mamba_ssm', 'panqec', 'optuna']
    ext_colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99']

    for i, (label, color) in enumerate(zip(ext_labels, ext_colors)):
        x_pos = 2 + i * 3
        ext_box = FancyBboxPatch((x_pos, 2), 2.5, 0.8, boxstyle="round,pad=0.05",
                                facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(ext_box)
        ax.text(x_pos + 1.25, 2.4, label, ha='center', va='center',
               fontsize=10, fontweight='bold')
        ext_boxes.append(ext_box)

    # Specialized components
    special_box1 = FancyBboxPatch((1, 0.5), 3, 0.8, boxstyle="round,pad=0.05",
                                 facecolor=colors['special'], edgecolor='red', linewidth=1)
    ax.add_patch(special_box1)
    ax.text(2.5, 0.9, 'cudaq_backend/', ha='center', va='center',
           fontsize=9, fontweight='bold')

    special_box2 = FancyBboxPatch((5, 0.5), 3, 0.8, boxstyle="round,pad=0.05",
                                 facecolor=colors['special'], edgecolor='red', linewidth=1)
    ax.add_patch(special_box2)
    ax.text(6.5, 0.9, 'fastpath/', ha='center', va='center',
           fontsize=9, fontweight='bold')

    special_box3 = FancyBboxPatch((9, 0.5), 3, 0.8, boxstyle="round,pad=0.05",
                                 facecolor=colors['special'], edgecolor='red', linewidth=1)
    ax.add_patch(special_box3)
    ax.text(10.5, 0.9, 'results/', ha='center', va='center',
           fontsize=9, fontweight='bold')

    # Draw connection arrows
    def draw_arrow(x1, y1, x2, y2, color='black', alpha=0.7):
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.05, head_length=0.1,
                fc=color, ec=color, alpha=alpha, length_includes_head=True)

    # Main connections
    draw_arrow(8, 9.5, 3.75, 8.2, 'navy')  # main -> poc_my_models
    draw_arrow(8, 9.5, 12.25, 8.2, 'navy')  # main -> panq_functions
    draw_arrow(8, 9.5, 3, 5.7, 'green')  # main -> cudaq_sampler
    draw_arrow(8, 9.5, 13, 5.7, 'green')  # main -> eval_ler

    draw_arrow(3.75, 6.8, 12.25, 6.8, 'purple')  # poc_my_models -> panq_functions
    draw_arrow(3, 4.3, 2.5, 1.3, 'red')  # cudaq_sampler -> cudaq_backend
    draw_arrow(13, 4.3, 6.5, 1.3, 'red')  # eval_ler -> fastpath

    # External connections
    draw_arrow(3.75, 6.8, 3.5, 2.8, 'orange')  # poc_my_models -> PyTorch
    draw_arrow(3.75, 6.8, 6.5, 2.8, 'orange')  # poc_my_models -> mamba_ssm
    draw_arrow(12.25, 6.8, 9.5, 2.8, 'orange')  # panq_functions -> panqec
    draw_arrow(8, 9.5, 12.5, 2.8, 'orange')  # main -> optuna

    # Title
    ax.text(8, 11.5, 'MGHD Repository Architecture - Training Dependencies',
           ha='center', va='center', fontsize=16, fontweight='bold')

    # Legend
    ax.text(1, 11, 'Color Legend:', fontsize=12, fontweight='bold')
    legend_items = [
        ('Core Training', colors['core']),
        ('Model Layer', colors['model']),
        ('Data Layer', colors['data']),
        ('External Libs', colors['external']),
        ('Specialized', colors['special'])
    ]

    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(FancyBboxPatch((1, 10.5 - i*0.3), 0.3, 0.2,
                                   facecolor=color, edgecolor='black', linewidth=1))
        ax.text(1.4, 10.6 - i*0.3, label, fontsize=10)

    plt.tight_layout()
    plt.savefig('MGHD_architecture_diagram_matplotlib.png', dpi=150, bbox_inches='tight')
    print("Matplotlib diagram saved as: MGHD_architecture_diagram_matplotlib.png")

if __name__ == "__main__":
    create_mghd_diagram()