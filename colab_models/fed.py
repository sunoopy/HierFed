# fed.py

import numpy as np
from scipy.stats import dirichlet
from HierFed.dataset import create_model, load_and_preprocess_data

def create_client_data_dirichlet(dataset, grid_size, clients_per_region, alpha, samples_per_client):
    (x_train, y_train), _ = load_and_preprocess_data(dataset)
    num_classes = 10 if dataset in ['mnist', 'cifar10'] else 100
    num_clients = sum(clients_per_region)
    
    dirichlet_x = dirichlet.rvs(alpha * np.ones(grid_size), size=1)[0]
    dirichlet_y = dirichlet.rvs(alpha * np.ones(grid_size), size=1)[0]
    
    class_probs = np.zeros((num_classes, grid_size, grid_size))
    for k in range(num_classes):
        class_probs[k] = np.outer(dirichlet_x, dirichlet_y)
    
    class_probs /= class_probs.sum(axis=(1, 2), keepdims=True)
    class_probs_flat = class_probs.reshape(num_classes, -1)
    class_probs_flat /= class_probs_flat.sum(axis=1, keepdims=True)
    
    client_data = [[] for _ in range(num_clients)]
    for i in range(len(x_train)):
        class_idx = y_train[i][0] if dataset != 'mnist' else y_train[i]
        probs = class_probs_flat[class_idx]
        grid_idx = np.random.choice(grid_size * grid_size, p=probs)
        client_idx = np.random.choice(num_clients)
        client_data[client_idx].append((x_train[i], y_train[i]))
    
    for i in range(num_clients):
        if len(client_data[i]) < samples_per_client:
            additional = np.random.choice(len(client_data[i]), samples_per_client - len(client_data[i]))
            client_data[i].extend([client_data[i][j] for j in additional])
        elif len(client_data[i]) > samples_per_client:
            client_data[i] = [client_data[i][j] for j in np.random.choice(len(client_data[i]), samples_per_client, replace=False)]
    
    for i in range(num_clients):
        x, y = zip(*client_data[i])
        client_data[i] = (np.array(x), np.array(y))
    
    return client_data, class_probs

def client_update(client_model, client_data):
    x, y = client_data
    client_model.fit(x, y, epochs=5, verbose=0)
    return client_model.get_weights()

def regional_server_update(global_weights, client_weights):
    averaged_weights = []
    for weights_list_tuple in zip(*client_weights):
        averaged_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )
    return averaged_weights

def global_server_update(global_weights, regional_weights):
    averaged_weights = []
    for weights_list_tuple in zip(*regional_weights):
        averaged_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )
    return averaged_weights

def hierarchical_federated_learning(dataset, grid_size, clients_per_region, alpha, samples_per_client, rounds):
    global_model = create_model(dataset)
    global_weights = global_model.get_weights()
    
    client_data, _ = create_client_data_dirichlet(dataset, grid_size, clients_per_region, alpha, samples_per_client)
    
    for round in range(rounds):
        print(f"Round {round + 1}/{rounds}")
        
        regional_weights = []
        client_idx_start = 0
        for region_clients in clients_per_region:
            client_weights = []
            for client in range(region_clients):
                client_idx = client_idx_start + client
                client_model = create_model(dataset)
                client_model.set_weights(global_weights)
                client_weights.append(client_update(client_model, client_data[client_idx]))
            
            regional_weights.append(regional_server_update(global_weights, client_weights))
            client_idx_start += region_clients
        
        global_weights = global_server_update(global_weights, regional_weights)
        global_model.set_weights(global_weights)
        
        _, (x_test, y_test) = load_and_preprocess_data(dataset)
        test_loss, test_accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"Global model accuracy: {test_accuracy:.4f}")
    
    return global_model
