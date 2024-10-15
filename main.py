# main.py
import numpy as np
from HierFed.dataset import create_model, load_and_preprocess_data
from fed import hierarchical_federated_learning, create_client_data_dirichlet
from visualization_analysis import analyze_client_data, visualize_class_probabilities

dataset = 'mnist'
grid_size = 10
clients_per_region = [10, 100, 2]
alpha = 0.5
samples_per_client = 100
rounds = 2

client_data, class_probs = create_client_data_dirichlet(dataset, grid_size, clients_per_region, alpha, samples_per_client)
analyze_client_data(client_data, dataset, clients_per_region)

random_x, random_y = np.random.randint(0, grid_size, 2)
visualize_class_probabilities(class_probs, grid_size, random_x, random_y)

final_model = hierarchical_federated_learning(dataset, grid_size, clients_per_region, alpha, samples_per_client, rounds)
