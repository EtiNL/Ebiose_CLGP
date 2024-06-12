import json
import pickle as pkl
from collections import defaultdict
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
import webbrowser
import math

# Step 1: Load the data
def load_data(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def load_validation_set(filepath):
    with open(filepath, 'rb') as f:
        validation_set = pkl.load(f)
    return validation_set

# Step 2: Process the data
def process_data(data, validation_set):
    question_success_counts = defaultdict(int)
    graph_success_counts = defaultdict(int)
    question_graph_links = defaultdict(lambda: defaultdict(int))

    for entry in data:
        agent = entry[0]
        evaluations = entry[1]["evaluations"]
        dataset_indexes = entry[1]["dataset_indexes"]
        
        graph_id = agent["id"]
        for eval_result, dataset_index in zip(evaluations, dataset_indexes):
            question = validation_set["question"][dataset_index]
            if eval_result:
                question_success_counts[question] += 1
                graph_success_counts[graph_id] += 1
                question_graph_links[question][graph_id] += 1

    return question_success_counts, graph_success_counts, question_graph_links

# Function to apply repulsion
def apply_repulsion(positions, connectivity, k=0.1, iterations=50):
    n = len(positions)
    for _ in range(iterations):
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                distance = np.linalg.norm(diff)
                if distance > 0:
                    force = k * connectivity[i] * connectivity[j] / distance**2
                    positions[i] += force * (diff / distance)
                    positions[j] -= force * (diff / distance)
    return positions

# Step 3: Plot the data in 3D
def plot_3d_graph(question_graph_links):
    nodes = set()
    for question, graphs in question_graph_links.items():
        nodes.add(question)
        nodes.update(graphs.keys())

    nodes = list(nodes)
    node_indices = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    positions = np.random.rand(n, 3)

    edges = []
    for question, graphs in question_graph_links.items():
        for graph, count in graphs.items():
            edges.append((node_indices[question], node_indices[graph], count))

    edge_x = []
    edge_y = []
    edge_z = []
    edge_weights = []
    connectivity = np.zeros(n)
    for edge in edges:
        x0, y0, z0 = positions[edge[0]]
        x1, y1, z1 = positions[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]
        edge_weights.append(edge[2])
        connectivity[edge[0]] += edge[2]
        connectivity[edge[1]] += edge[2]

    positions = apply_repulsion(positions, connectivity)

    node_x = positions[:, 0]
    node_y = positions[:, 1]
    node_z = positions[:, 2]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='rgba(20, 20, 20, 0.9)'),
        hoverinfo='none',
        mode='lines'
    )

    question_nodes = [node for node in nodes if node in question_graph_links]
    graph_nodes = [node for node in nodes if node not in question_graph_links]

    question_node_indices = [node_indices[node] for node in question_nodes]
    graph_node_indices = [node_indices[node] for node in graph_nodes]

    question_node_x = positions[question_node_indices, 0]
    question_node_y = positions[question_node_indices, 1]
    question_node_z = positions[question_node_indices, 2]

    graph_node_x = positions[graph_node_indices, 0]
    graph_node_y = positions[graph_node_indices, 1]
    graph_node_z = positions[graph_node_indices, 2]

    question_node_trace = go.Scatter3d(
        x=question_node_x, y=question_node_y, z=question_node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color='blue',
            size=[math.sqrt(connectivity[node_indices[node]]) for node in question_nodes]
        ),
        text=question_nodes
    )

    graph_node_trace = go.Scatter3d(
        x=graph_node_x, y=graph_node_y, z=graph_node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color='green',
            size=[math.sqrt(connectivity[node_indices[node]]) for node in graph_nodes]
        ),
        text=graph_nodes
    )

    fig = go.Figure(data=[edge_trace, question_node_trace, graph_node_trace],
                    layout=go.Layout(
                        title='Graph-Question Links in 3D',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="3D visualization of graph-question links",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )],
                        scene=dict(
                            xaxis=dict(showbackground=False),
                            yaxis=dict(showbackground=False),
                            zaxis=dict(showbackground=False)
                        )
                    )
    )

    # Save the plot as an HTML file and open it in the default web browser
    filename = 'graph_question_links_3d.html'
    plot(fig, filename=filename, auto_open=False)

    # Open the HTML file in the default web browser
    webbrowser.open_new_tab(filename)

# Main function
def main():
    data_filepath = 'data/data.jsonl'
    validation_filepath = 'data/gsm8k-validation.pkl'

    data = load_data(data_filepath)
    validation_set = load_validation_set(validation_filepath)

    question_success_counts, graph_success_counts, question_graph_links = process_data(data, validation_set)
    plot_3d_graph(question_graph_links)

if __name__ == "__main__":
    main()
