import numpy as np
from scipy.stats import dirichlet
from collections import defaultdict
from typing import List, Tuple, Dict
import random

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


def assign_clients_to_edges(
    client_locations: List[Tuple[float, float]], 
    edge_points: List[Tuple[float, float]], 
    coverage_radius: float, 
    client_repetition: bool = True
) -> Dict[int, List[int]]:
    """
    Assign clients to multiple edge servers within coverage radius.

    Args:
        client_locations: List of (x, y) coordinates for clients.
        edge_points: List of (x, y) coordinates for edge servers.
        coverage_radius: Maximum distance for a client to connect to an edge server.
        client_repetition: Whether a client can be assigned to multiple edge servers.

    Returns:
        Dictionary mapping edge server indices to lists of client indices.
    """
    # Initialize assignments
    assignments = defaultdict(list)
    unassigned_clients = []
    assigned_clients = set()  # Track uniquely assigned clients

    for client_idx, client_loc in enumerate(client_locations):
        # Find all edge servers within coverage radius
        nearby_edges = []
        for edge_idx, edge_loc in enumerate(edge_points):
            distance = np.sqrt((client_loc[0] - edge_loc[0])**2 + (client_loc[1] - edge_loc[1])**2)
            if distance <= coverage_radius:
                nearby_edges.append((edge_idx, distance))
        
        # Sort nearby edges by distance
        nearby_edges.sort(key=lambda x: x[1])
        
        # If no nearby edges, add to unassigned
        if not nearby_edges:
            unassigned_clients.append(client_idx)
            continue

        # Handle client assignments based on repetition
        if client_repetition:
            # Assign to top 2 nearby edges (or all within coverage)
            max_nearby = min(2, len(nearby_edges))
            for i in range(max_nearby):
                assignments[nearby_edges[i][0]].append(client_idx)
        else:
            # Assign to the closest edge server only if not already assigned
            for edge_info in nearby_edges:
                if client_idx not in assigned_clients:
                    assignments[edge_info[0]].append(client_idx)
                    assigned_clients.add(client_idx)
                    break

    # Handle unassigned clients
    if unassigned_clients:
        print(f"Warning: {len(unassigned_clients)} clients are not covered by any edge server.")

    # Validate assignments if no repetition is allowed
    if not client_repetition:
        all_assigned_clients = [client for clients in assignments.values() for client in clients]
        assert len(set(all_assigned_clients)) == len(all_assigned_clients), \
            "Clients should not be repeated when client_repetition is False."

    return assignments


def distribute_data_to_clients(
    client_locations: List[Tuple[float, float]], 
    label_distributions: Dict[Tuple[int, int], np.ndarray], 
    y_train: np.ndarray, 
    x_train: np.ndarray, 
    num_classes: int, 
    sample_per_client: int
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Distribute data to clients based on their location and label distribution.
    
    Args:
        client_locations: List of (x, y) coordinates for clients.
        label_distributions: Label distributions for each grid point.
        y_train: Training labels.
        x_train: Training data.
        num_classes: Number of classes.
        sample_per_client: Number of samples per client.

    Returns:
        Dictionary containing data ('x' and 'y') for each client.
    """
    client_data = {}
    client_label_counts = defaultdict(lambda: defaultdict(int))

    # Get indices for each class
    class_indices = defaultdict(list)
    for idx, label in enumerate(y_train):
        class_indices[label].append(idx)

    for client_idx, location in enumerate(client_locations):
        grid_x, grid_y = int(location[0]), int(location[1])
        dist = label_distributions[(grid_x, grid_y)]

        client_indices = []
        remaining_samples = sample_per_client

        while remaining_samples > 0:
            class_label = np.random.choice(num_classes, p=dist)

            if class_indices[class_label]:
                sampled_idx = random.choice(class_indices[class_label])
                client_indices.append(sampled_idx)
                class_indices[class_label].remove(sampled_idx)
                remaining_samples -= 1
                client_label_counts[client_idx][class_label] += 1

        # Store actual data for each client
        client_data[client_idx] = {
            "x": x_train[client_indices],
            "y": y_train[client_indices],
        }

    return client_data