import numpy as np
from scipy.stats import dirichlet
from collections import defaultdict

def generate_edge_server_locations(grid_size, num_edge_servers):
    edge_points = []
    rows = int(np.sqrt(num_edge_servers))
    cols = num_edge_servers // rows
    for i in range(rows):
        for j in range(cols):
            x = (i + 0.5) * (grid_size / rows)
            y = (j + 0.5) * (grid_size / cols)
            edge_points.append((x, y))
    return edge_points

def generate_client_locations(grid_size, num_clients):
    return [(np.random.uniform(0, grid_size), np.random.uniform(0, grid_size)) for _ in range(num_clients)]

def generate_label_distributions(grid_size, num_classes, alpha):
    distributions = {}
    for i in range(grid_size):
        for j in range(grid_size):
            distributions[(i, j)] = dirichlet.rvs([alpha] * num_classes)[0]
    return distributions

def calculate_kl_divergence(p, q):
    return sum(p[i] * np.log(p[i] / q[i]) for i in range(len(p)) if p[i] > 0 and q[i] > 0)
