from graphviz import Digraph

# Create a directed graph
dot = Digraph(comment='ConformationNN Architecture', format='png')
dot.attr(rankdir='LR', size='8,5')  # Left-to-right layout

# Define layers and nodes
input_size = 5  # Example: RMSD (1) + sec_struct (1) + coords_apo (3)
hidden_size = 128
output_size = 3

# Input Layer
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input Layer (5 features)', color='blue')
    for i in range(input_size):
        c.node(f'input_{i}', shape='circle', width='0.6')

# Hidden Layers
for layer_idx in range(1, 5):
    with dot.subgraph(name=f'cluster_hidden_{layer_idx}') as c:
        c.attr(label=f'Hidden Layer {layer_idx} (128 units)', color='black')
        for i in range(hidden_size):
            c.node(f'hidden_{layer_idx}_{i}', shape='circle', width='0.6')

# Output Layer
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Output Layer (3 coordinates)', color='red')
    for i in range(output_size):
        c.node(f'output_{i}', shape='circle', width='0.6')

# Connect layers
for i in range(input_size):
    for j in range(hidden_size):
        dot.edge(f'input_{i}', f'hidden_1_{j}')

for layer_idx in range(1, 4):
    for i in range(hidden_size):
        for j in range(hidden_size):
            dot.edge(f'hidden_{layer_idx}_{i}', f'hidden_{layer_idx+1}_{j}')

for i in range(hidden_size):
    for j in range(output_size):
        dot.edge(f'hidden_4_{i}', f'output_{j}')

# Save and render
dot.render('ConformationNN_Architecture', view=True)
