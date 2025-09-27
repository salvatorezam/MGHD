#!/usr/bin/env python3
"""Generate conceptual MGHD architecture diagram showing data flow with tensor shapes."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, ConnectionPatch
import numpy as np
from pathlib import Path

# High-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# MGHD-S Profile specifications
PROFILE_S = {
    'n_iters': 7,
    'n_node_features': 128,
    'n_edge_features': 128,
    'msg_net': 96,
    'd_model': 192,
    'd_state': 32,
}

def draw_tensor_stack(ax, x, y, width, height, depth, channels, color, label, shape_text):
    """Draw a 3D-looking tensor stack."""
    
    # Draw the main rectangle
    main_rect = Rectangle((x, y), width, height, facecolor=color, 
                         edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.add_patch(main_rect)
    
    # Draw depth lines to show 3D effect
    offset = min(0.1, depth * 0.02)
    for i in range(min(5, int(depth))):  # Show max 5 layers for visual clarity
        dx = i * offset
        dy = i * offset
        rect = Rectangle((x + dx, y + dy), width, height, 
                        facecolor=color, edgecolor='black', 
                        linewidth=0.8, alpha=0.6)
        ax.add_patch(rect)
    
    # Add label
    ax.text(x + width/2, y - 0.3, label, ha='center', va='top', 
            fontsize=9, fontweight='bold')
    
    # Add shape information
    ax.text(x + width/2, y - 0.6, shape_text, ha='center', va='top', 
            fontsize=8, style='italic')
    
    return main_rect

def draw_processing_block(ax, x, y, width, height, title, description, color):
    """Draw a processing block with title and description."""
    
    # Main block
    block = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(block)
    
    # Title
    ax.text(x + width/2, y + height - 0.2, title, ha='center', va='top',
            fontsize=10, fontweight='bold')
    
    # Description
    ax.text(x + width/2, y + height/2, description, ha='center', va='center',
            fontsize=8, linespacing=1.2)
    
    return block

def draw_arrow(ax, start_x, start_y, end_x, end_y, label=''):
    """Draw an arrow between points."""
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    if label:
        mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', va='center', 
                fontsize=7, bbox=dict(boxstyle="round,pad=0.2", 
                                     facecolor='white', alpha=0.8))

def create_mghd_conceptual_diagram():
    """Create conceptual MGHD-S diagram showing data flow."""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(9, 9.5, 'MGHD-S Conceptual Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(9, 9.1, 'Hybrid Mamba-GNN decoder for surface codes (distance d)', 
            ha='center', va='center', fontsize=12, style='italic')
    
    # Stage 1: Input Syndrome
    draw_tensor_stack(ax, 0.5, 7, 1.5, 1.2, 1, 8, '#E3F2FD', 
                     'INPUT', 'Syndrome bits\n2d(d-1) total\n(8 for d=3)')
    
    # Add small syndrome visualization showing 8 bits
    syndrome_vis = Rectangle((0.7, 6.2), 1.1, 0.6, facecolor='lightgray', 
                           edgecolor='black', linewidth=1)
    ax.add_patch(syndrome_vis)
    # Draw 8 syndrome bits in 2 rows of 4
    for i in range(4):
        for j in range(2):
            bit_color = 'red' if (i + j) % 3 == 0 else 'white'
            bit_rect = Rectangle((0.75 + i*0.25, 6.25 + j*0.25), 0.2, 0.2,
                               facecolor=bit_color, edgecolor='black', linewidth=0.5)
            ax.add_patch(bit_rect)
    
    # Add labels for Z and X checks with general formula
    ax.text(0.6, 6.05, 'd(d-1) Z checks', ha='left', va='center', fontsize=7, color='blue')
    ax.text(0.6, 5.85, 'd(d-1) X checks', ha='left', va='center', fontsize=7, color='blue')
    
    # Stage 2: Graph Construction + Embedding
    draw_processing_block(ax, 2.5, 7.5, 1.8, 1, 'Graph Construction',
                         'Build Tanner graph\n(2d²-1) total nodes', '#D4EFDF')
    
    draw_tensor_stack(ax, 2.5, 6, 1.8, 1.2, 17, 9, '#E8F5E8',
                     'GRAPH NODES', '(2d²-1) × 9 features\n2d(d-1) checks + d² qubits\n(17 nodes for d=3)')
    
    # Stage 3: Linear Embedding
    draw_processing_block(ax, 5, 7.5, 1.8, 1, 'Linear Embedding',
                         'Project to Mamba\n9 → 192', '#FFE6E6')
    
    draw_tensor_stack(ax, 5, 6, 1.8, 1.2, 17, 192, '#FFE6E6',
                     'EMBEDDED', '17 × 192')
    
    # Stage 4: Mamba SSM
    draw_processing_block(ax, 7.5, 7.8, 2.5, 0.8, 'Mamba SSM',
                         'Sequential State-Space Model', '#FFB3B3')
    
    # Mamba internal components
    draw_tensor_stack(ax, 7.5, 6.8, 0.7, 0.8, 6, 192, '#FFCCCC',
                     'Conv1D', 'd_conv=4')
    draw_tensor_stack(ax, 8.3, 6.8, 0.7, 0.8, 3, 32, '#FF9999',
                     'SSM State', f'd_state={PROFILE_S["d_state"]}')
    draw_tensor_stack(ax, 9.1, 6.8, 0.8, 0.8, 6, 192, '#FFCCCC',
                     'Gate', 'SiLU + Gate')
    
    draw_tensor_stack(ax, 7.5, 5.5, 2.5, 1.2, 17, 192, '#FFB3B3',
                     'MAMBA OUT', '17 × 192')
    
    # Stage 5: Channel Attention
    draw_processing_block(ax, 10.5, 7.5, 1.5, 1, 'Channel SE',
                         'Squeeze-Excite\n4:1 reduction', '#FFF3CD')
    
    draw_tensor_stack(ax, 10.5, 6, 1.5, 1.2, 17, 192, '#FFF8DC',
                     'ATTENTION', '17 × 192')
    
    # Stage 6: Projection to Graph
    draw_processing_block(ax, 12.5, 7.5, 1.8, 1, 'Graph Projection',
                         'Linear\n192 → 9', '#F3E5F5')
    
    draw_tensor_stack(ax, 12.5, 6, 1.8, 1.2, 17, 9, '#E8DAEF',
                     'PROJECTED', '17 × 9')
    
    # Stage 7: GNN Message Passing (show iterations)
    draw_processing_block(ax, 15, 8, 2.5, 0.6, 'GNN Message Passing',
                         f'{PROFILE_S["n_iters"]} iterations', '#E8F5E8')
    
    # Message network detail
    draw_tensor_stack(ax, 15, 7, 0.8, 0.6, 4, 96, '#D5E8D4',
                     'Message Net', f'MLP-{PROFILE_S["msg_net"]}')
    
    # GRU update
    draw_tensor_stack(ax, 16, 7, 0.8, 0.6, 5, 128, '#C8E6C9',
                     'GRU', f'Hidden-{PROFILE_S["n_node_features"]}')
    
    # Final output
    draw_tensor_stack(ax, 15, 5.5, 2.5, 1.2, 9, 2, '#FFF2CC',
                     'OUTPUT', 'd² × 2 logits\n(per data qubit)\n(9 qubits for d=3)')
    
    # Add arrows showing data flow
    arrows = [
        (2, 7.6, 2.5, 7.6),          # Input → Graph construction
        (4.3, 6.6, 5, 6.6),          # Graph → Linear embedding
        (6.8, 6.6, 7.5, 6.6),        # Embedding → Mamba
        (10, 6.6, 10.5, 6.6),        # Mamba → Attention
        (12, 6.6, 12.5, 6.6),        # Attention → Projection
        (14.3, 6.6, 15, 6.6),        # Projection → GNN
    ]
    
    for start_x, start_y, end_x, end_y in arrows:
        draw_arrow(ax, start_x, start_y, end_x, end_y)
    
    # Add GNN iteration loop
    ax.annotate('', xy=(15.5, 6.8), xytext=(16.5, 6.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='red',
                              connectionstyle="arc3,rad=0.3"))
    ax.text(16.8, 6.5, f'{PROFILE_S["n_iters"]}×', ha='center', va='center',
            fontsize=10, color='red', fontweight='bold')
    
    # Add technical specifications box
    specs_text = (
        "MGHD-S Specifications:\n"
        f"• d_model: {PROFILE_S['d_model']} (Mamba hidden)\n"
        f"• d_state: {PROFILE_S['d_state']} (SSM state)\n"
        f"• GNN hidden: {PROFILE_S['n_node_features']}\n"
        f"• Message net: {PROFILE_S['msg_net']}\n"
        f"• GNN iterations: {PROFILE_S['n_iters']}\n"
        f"• Total params: ~{estimate_parameters_s():,}"
    )
    
    ax.text(1, 4, specs_text, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
    
    # Add data flow description
    flow_text = (
        "Data Flow (Surface Code d=3):\n"
        "1. 2d(d-1)=8 syndrome bits\n"
        "2. Build graph: 2d²-1=17 nodes\n"  
        "3. Linear embed to Mamba space\n"
        "4. Mamba sequential processing\n"
        "5. Channel attention gating\n"
        "6. Project back to graph space\n"
        "7. GNN message passing (7 iters)\n"
        "8. d²=9 binary logits (corrections)"
    )
    
    ax.text(8, 4, flow_text, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.8))
    
    # Add quantum error correction context
    qec_text = (
        "Surface Code Scaling (general):\n"
        "• Distance d surface code\n"
        "• 2d(d-1) syndrome bits\n"
        "• d² data qubits\n"
        "• 2d²-1 total graph nodes\n"
        "• Example: d=3 → 8+9=17 nodes\n"
        "• Example: d=5 → 20+25=45 nodes"
    )
    
    ax.text(12, 4, qec_text, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
    
    # Add legend for tensor visualization
    ax.text(0.5, 2.5, 'Legend:', ha='left', va='top', fontsize=10, fontweight='bold')
    ax.text(0.5, 2.2, '□ = Tensor/Feature map', ha='left', va='top', fontsize=9)
    ax.text(0.5, 1.9, '3D stack = Multiple channels/features', ha='left', va='top', fontsize=9)
    ax.text(0.5, 1.6, 'Shape: (batch × nodes × features)', ha='left', va='top', fontsize=9)
    
    plt.tight_layout()
    return fig

def estimate_parameters_s():
    """Estimate parameters for MGHD-S."""
    p = PROFILE_S
    
    embedding = 9 * p['d_model']
    mamba = p['d_model'] * (p['d_state'] * 2 + p['d_model'] + 4 * p['d_model'])
    projection = p['d_model'] * 9
    msg_net = (2 * p['n_node_features'] * p['msg_net'] + 
               3 * p['msg_net'] * p['msg_net'] + 
               p['msg_net'] * p['n_edge_features'])
    gru_input_size = p['n_edge_features'] + 9
    gru = 3 * (gru_input_size * p['n_node_features'] + p['n_node_features'] * p['n_node_features'])
    output_head = p['n_node_features'] * 9
    channel_attn = p['d_model'] * (p['d_model'] // 4) * 2
    
    total = embedding + mamba + projection + msg_net + gru + output_head + channel_attn
    return int(total)

def main():
    """Generate the conceptual MGHD-S diagram."""
    
    output_dir = Path("results/figs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Generating MGHD-S conceptual architecture diagram...")
    
    fig = create_mghd_conceptual_diagram()
    
    # Save files
    png_path = output_dir / "mghd_conceptual_S.png"
    pdf_path = output_dir / "mghd_conceptual_S.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    
    print(f"✅ Conceptual diagram saved: {png_path}")
    print("🎉 MGHD-S conceptual visualization complete!")

if __name__ == "__main__":
    main()