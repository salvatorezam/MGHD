#!/usr/bin/env python3
"""Generate classic neural network node-and-edge diagrams for MGHD architecture."""

import matplotlib.pyplot as plt
import networkx as nx
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

# Model profiles from your code
PROFILES = {
    'S': {'n_iters': 7, 'n_node_features': 128, 'n_edge_features': 128, 'msg_net': 96, 'd_model': 192, 'd_state': 32},
    'M': {'n_iters': 8, 'n_node_features': 192, 'n_edge_features': 192, 'msg_net': 128, 'd_model': 256, 'd_state': 48},
    'L': {'n_iters': 9, 'n_node_features': 256, 'n_edge_features': 256, 'msg_net': 160, 'd_model': 320, 'd_state': 64},
    'XL': {'n_iters': 10, 'n_node_features': 256, 'n_edge_features': 256, 'msg_net': 192, 'd_model': 384, 'd_state': 64},
}

def create_neural_network_diagram(profile='S'):
    """Create a classic neural network diagram with nodes and connections proportional to actual sizes."""
    
    p = PROFILES[profile]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(9, 11.5, f'MGHD Neural Network - Profile {profile}', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Scale factor for visualization (adjust max nodes shown)
    max_visual_nodes = 20  # Maximum nodes to show visually
    
    # Calculate proportional node counts (scaled but representative)
    def scale_nodes(actual_size, max_nodes=max_visual_nodes):
        """Scale actual layer size to visual representation"""
        if actual_size <= max_nodes:
            return actual_size
        else:
            # Logarithmic scaling for very large layers
            import math
            return int(max_nodes * (1 + 0.3 * math.log10(actual_size / max_nodes)))
    
    # Layer positions and sizes - NOW PROPORTIONAL TO ACTUAL DIMENSIONS
    layers = [
        {'name': 'Input', 'x': 1.5, 'nodes': 9, 'actual_size': 9, 'color': '#E3F2FD', 'label': f'Input\n[9]'},
        {'name': 'Embedding', 'x': 3.5, 'nodes': scale_nodes(p['d_model']), 'actual_size': p['d_model'], 'color': '#E8F5E8', 'label': f'Embedding\n[{p["d_model"]}]'},
        {'name': 'Mamba_Model', 'x': 5.5, 'nodes': scale_nodes(p['d_model']), 'actual_size': p['d_model'], 'color': '#FFE6E6', 'label': f'Mamba d_model\n[{p["d_model"]}]'},
        {'name': 'Mamba_State', 'x': 7, 'nodes': scale_nodes(p['d_state']), 'actual_size': p['d_state'], 'color': '#FFCCCC', 'label': f'SSM State\n[{p["d_state"]}]'},
        {'name': 'Attention', 'x': 8.5, 'nodes': scale_nodes(p['d_model']//4), 'actual_size': p['d_model']//4, 'color': '#FFF3CD', 'label': f'Channel SE\n[{p["d_model"]//4}]'},
        {'name': 'Projection', 'x': 10, 'nodes': 9, 'actual_size': 9, 'color': '#F3E5F5', 'label': f'Projection\n[9]'},
        {'name': 'GNN_Hidden', 'x': 12, 'nodes': scale_nodes(p['n_node_features']), 'actual_size': p['n_node_features'], 'color': '#E8F8F5', 'label': f'GNN Hidden\n[{p["n_node_features"]}]'},
        {'name': 'Message', 'x': 14, 'nodes': scale_nodes(p['msg_net']), 'actual_size': p['msg_net'], 'color': '#D5E8D4', 'label': f'Message Net\n[{p["msg_net"]}]'},
        {'name': 'Edge_Features', 'x': 15.5, 'nodes': scale_nodes(p['n_edge_features']), 'actual_size': p['n_edge_features'], 'color': '#C8E6C9', 'label': f'Edge Features\n[{p["n_edge_features"]}]'},
        {'name': 'Output', 'x': 17, 'nodes': 9, 'actual_size': 9, 'color': '#FFF2CC', 'label': f'Output\n[9]'},
    ]
    
    # Draw nodes for each layer - PROPORTIONAL SIZING
    node_positions = {}
    node_colors = {}
    node_labels = {}
    
    for layer in layers:
        nodes = layer['nodes']
        actual_size = layer['actual_size']
        x = layer['x']
        
        # Calculate y positions for nodes (centered and spread based on actual count)
        if nodes == 1:
            y_positions = [6.0]
        elif nodes <= 5:
            y_positions = np.linspace(4, 8, nodes)
        elif nodes <= 10:
            y_positions = np.linspace(3, 9, nodes)
        elif nodes <= 15:
            y_positions = np.linspace(2.5, 9.5, nodes)
        else:
            y_positions = np.linspace(2, 10, nodes)
        
        # Create nodes with size indicating importance/actual size
        for i, y in enumerate(y_positions):
            node_id = f"{layer['name']}_{i}"
            node_positions[node_id] = (x, y)
            node_colors[node_id] = layer['color']
            
        # Add layer label with actual size
        if actual_size > 50:
            size_indicator = "●●●"  # Large layer
        elif actual_size > 20:
            size_indicator = "●●"   # Medium layer  
        else:
            size_indicator = "●"    # Small layer
            
        ax.text(x, 1.2, layer['label'], ha='center', va='center', fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=layer['color'], alpha=0.7))
        ax.text(x, 0.6, f"{size_indicator} {actual_size} neurons", ha='center', va='center', 
                fontsize=8, style='italic', color='darkblue')
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, pos in node_positions.items():
        G.add_node(node_id, pos=pos)
    
    # Add edges between consecutive layers
    for i in range(len(layers) - 1):
        current_layer = layers[i]['name']
        next_layer = layers[i + 1]['name']
        
        current_nodes = [n for n in node_positions.keys() if n.startswith(current_layer)]
        next_nodes = [n for n in node_positions.keys() if n.startswith(next_layer)]
        
        # Connect each node in current layer to each node in next layer
        for curr_node in current_nodes:
            for next_node in next_nodes:
                G.add_edge(curr_node, next_node)
    
    # Draw the network
    pos = node_positions
    
    # Draw edges (connections)
    nx.draw_networkx_edges(G, pos, ax=ax, 
                          edge_color='lightgray', 
                          alpha=0.6, 
                          width=0.8,
                          arrows=True,
                          arrowsize=8,
                          arrowstyle='->')
    
    # Draw nodes with sizes proportional to layer importance
    for node_id, position in pos.items():
        color = node_colors[node_id]
        
        # Get layer info to determine node size
        layer_name = node_id.split('_')[0]
        layer_info = next((l for l in layers if l['name'] == layer_name), None)
        
        # Node size based on actual layer size
        if layer_info:
            actual_size = layer_info['actual_size']
            if actual_size >= 200:
                node_radius = 0.20  # Large layers
            elif actual_size >= 100:
                node_radius = 0.17  # Medium-large layers
            elif actual_size >= 50:
                node_radius = 0.15  # Medium layers
            else:
                node_radius = 0.12  # Small layers
        else:
            node_radius = 0.15
            
        circle = plt.Circle(position, node_radius, color=color, ec='black', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        
        # Add node number/identifier for input and output layers
        node_num = node_id.split('_')[1]
        if layer_name in ['Input', 'Output']:
            ax.text(position[0], position[1], f'{node_num}', ha='center', va='center', 
                   fontsize=8, fontweight='bold', zorder=4)
        elif layer_info and layer_info['actual_size'] >= 100:
            # For large hidden layers, show a sample of indices
            if int(node_num) % max(1, len([n for n in pos.keys() if n.startswith(layer_name)]) // 3) == 0:
                ax.text(position[0], position[1], '●', ha='center', va='center', 
                       fontsize=6, fontweight='bold', zorder=4, color='white')
    
    # Add architectural annotations
    annotations = [
        (2.5, 10.5, "Input Processing", "9 syndrome bits"),
        (5, 10.5, "Sequential Modeling", f"Mamba d_model={p['d_model']}\nd_state={p['d_state']}"),
        (8.5, 10.5, "Attention", f"SE reduction 4:1\n{p['d_model']}→{p['d_model']//4}"),
        (13.5, 10.5, "Graph Message Passing", f"{p['n_iters']} GNN iterations\nHidden: {p['n_node_features']}"),
    ]
    
    for x, y, title, desc in annotations:
        ax.text(x, y, title, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, y-0.4, desc, ha='center', va='center', fontsize=8, style='italic')
    
    # Add parameter count and profile comparison
    total_params = estimate_parameters(profile)
    ax.text(9, 0.3, f'Profile {profile} - Total Parameters: ~{total_params:,}', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
    
    # Add scale legend
    ax.text(1, 10.5, f'Node sizes proportional to layer dimensions\n'
                     f'Larger circles = larger layers\n'
                     f'Profile {profile}: {layers[1]["actual_size"]} to {layers[-3]["actual_size"]} neurons per layer', 
            ha='left', va='center', fontsize=8, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    # Add data flow arrows at the bottom
    arrow_y = 0.2
    for i in range(len(layers) - 1):
        x1 = layers[i]['x']
        x2 = layers[i + 1]['x']
        ax.annotate('', xy=(x2 - 0.3, arrow_y), xytext=(x1 + 0.3, arrow_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_detailed_layer_diagram(profile='S'):
    """Create a detailed view of individual layers."""
    
    p = PROFILES[profile]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    fig.suptitle(f'MGHD Layer Details - Profile {profile}', fontsize=16, fontweight='bold')
    
    # 1. Mamba SSM Internal Structure
    ax1.set_title('Mamba SSM Internal Structure', fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    
    mamba_layers = [
        {'name': 'Input', 'x': 1, 'nodes': 4, 'color': '#E3F2FD'},
        {'name': 'Conv1D', 'x': 3, 'nodes': 4, 'color': '#FFE6E6'},
        {'name': 'SSM_Core', 'x': 5, 'nodes': min(6, p['d_state']//8), 'color': '#FFE6E6'},
        {'name': 'Gate', 'x': 7, 'nodes': 4, 'color': '#FFF3CD'},
        {'name': 'Output', 'x': 9, 'nodes': 4, 'color': '#E8F5E8'},
    ]
    
    draw_layer_diagram(ax1, mamba_layers, f"d_model={p['d_model']}, d_state={p['d_state']}")
    
    # 2. Message Network Structure
    ax2.set_title('GNN Message Network', fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    
    msg_layers = [
        {'name': 'Concat', 'x': 1, 'nodes': 6, 'color': '#E3F2FD'},
        {'name': 'Linear1', 'x': 3, 'nodes': min(6, p['msg_net']//16), 'color': '#E8F5E8'},
        {'name': 'Linear2', 'x': 5, 'nodes': min(6, p['msg_net']//16), 'color': '#E8F5E8'},
        {'name': 'Linear3', 'x': 7, 'nodes': min(6, p['msg_net']//16), 'color': '#E8F5E8'},
        {'name': 'Output', 'x': 9, 'nodes': min(6, p['n_edge_features']//20), 'color': '#FFF2CC'},
    ]
    
    draw_layer_diagram(ax2, msg_layers, f"Hidden size: {p['msg_net']}")
    
    # 3. GRU Update Structure
    ax3.set_title('GRU Update Mechanism', fontweight='bold')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 8)
    ax3.axis('off')
    
    gru_layers = [
        {'name': 'Messages', 'x': 1, 'nodes': 4, 'color': '#E8F5E8'},
        {'name': 'Concat', 'x': 3, 'nodes': 6, 'color': '#F3E5F5'},
        {'name': 'GRU_Gates', 'x': 5, 'nodes': 8, 'color': '#FFE6E6'},
        {'name': 'Hidden', 'x': 7, 'nodes': min(6, p['n_node_features']//20), 'color': '#E8F8F5'},
        {'name': 'Output', 'x': 9, 'nodes': 4, 'color': '#FFF2CC'},
    ]
    
    draw_layer_diagram(ax3, gru_layers, f"Hidden: {p['n_node_features']}")
    
    # 4. Full Forward Pass
    ax4.set_title(f'Complete Forward Pass ({p["n_iters"]} iterations)', fontweight='bold')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 8)
    ax4.axis('off')
    
    # Show iteration loop
    full_layers = [
        {'name': 'Syndrome', 'x': 1, 'nodes': 4, 'color': '#E3F2FD'},
        {'name': 'Mamba', 'x': 3, 'nodes': 5, 'color': '#FFE6E6'},
        {'name': 'GNN_Iter', 'x': 5, 'nodes': 6, 'color': '#E8F8F5'},
        {'name': 'Update', 'x': 7, 'nodes': 6, 'color': '#F3E5F5'},
        {'name': 'Logits', 'x': 9, 'nodes': 4, 'color': '#FFF2CC'},
    ]
    
    draw_layer_diagram(ax4, full_layers, f"Iterations: {p['n_iters']}")
    
    # Add iteration loop arrow
    ax4.annotate('', xy=(5, 1), xytext=(7, 1),
                arrowprops=dict(arrowstyle='->', lw=3, color='red', 
                              connectionstyle="arc3,rad=-0.5"))
    ax4.text(6, 0.5, f'{p["n_iters"]}× iterations', ha='center', va='center', 
            fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    return fig

def draw_layer_diagram(ax, layers, subtitle):
    """Helper function to draw layer diagrams."""
    
    node_positions = {}
    
    for layer in layers:
        nodes = layer['nodes']
        x = layer['x']
        
        # Calculate y positions
        if nodes == 1:
            y_positions = [4.0]
        else:
            y_positions = np.linspace(1.5, 6.5, nodes)
        
        # Draw nodes
        for i, y in enumerate(y_positions):
            node_id = f"{layer['name']}_{i}"
            node_positions[node_id] = (x, y)
            
            circle = plt.Circle((x, y), 0.12, color=layer['color'], 
                              ec='black', linewidth=1.0)
            ax.add_patch(circle)
        
        # Layer label
        ax.text(x, 0.8, layer['name'], ha='center', va='center', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor=layer['color'], alpha=0.7))
    
    # Draw connections
    for i in range(len(layers) - 1):
        current_layer = layers[i]['name']
        next_layer = layers[i + 1]['name']
        
        current_nodes = [(k, v) for k, v in node_positions.items() if k.startswith(current_layer)]
        next_nodes = [(k, v) for k, v in node_positions.items() if k.startswith(next_layer)]
        
        for _, curr_pos in current_nodes:
            for _, next_pos in next_nodes:
                ax.plot([curr_pos[0], next_pos[0]], [curr_pos[1], next_pos[1]], 
                       'lightgray', alpha=0.6, linewidth=0.5)
    
    ax.text(5, 7.5, subtitle, ha='center', va='center', fontsize=9, style='italic')

def estimate_parameters(profile):
    """Estimate total parameters for a profile."""
    p = PROFILES[profile]
    
    # Rough parameter estimation
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
    """Generate classic neural network diagrams."""
    
    output_dir = Path("results/figs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧠 Generating classic neural network diagrams...")
    
    # Generate main network diagrams for each profile
    for profile in ['S', 'M', 'L', 'XL']:
        print(f"  🎯 Creating {profile} profile network diagram...")
        
        # Main network diagram
        fig1 = create_neural_network_diagram(profile)
        main_path = output_dir / f"mghd_network_classic_{profile}.png"
        fig1.savefig(main_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig1.savefig(output_dir / f"mghd_network_classic_{profile}.pdf", bbox_inches='tight', facecolor='white')
        plt.close(fig1)
        print(f"    ✅ Main network: {main_path}")
        
        # Detailed layer diagram
        fig2 = create_detailed_layer_diagram(profile)
        detail_path = output_dir / f"mghd_layers_detail_{profile}.png"
        fig2.savefig(detail_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig2.savefig(output_dir / f"mghd_layers_detail_{profile}.pdf", bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f"    ✅ Layer details: {detail_path}")
    
    print("🎉 Classic neural network diagrams complete!")

if __name__ == "__main__":
    main()