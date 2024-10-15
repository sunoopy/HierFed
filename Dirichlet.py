import tensorflow as tf
import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt

# Define models for different datasets
def create_model(dataset):
    if dataset == 'mnist':
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    elif dataset in ['cifar10', 'cifar100']:
        num_classes = 10 if dataset == 'cifar10' else 100
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist', 'cifar10', or 'cifar100'.")
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load and preprocess data for different datasets
def load_and_preprocess_data(dataset):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
    elif dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist', 'cifar10', or 'cifar100'.")
    
    return (x_train, y_train), (x_test, y_test)



def create_client_data_dirichlet(dataset, grid_size, num_clients, alpha, samples_per_client):
    (x_train, y_train), _ = load_and_preprocess_data(dataset)
    num_classes = 10 if dataset in ['mnist', 'cifar10'] else 100
    
    # Generate Dirichlet distributions for x and y axes
    dirichlet_x = dirichlet.rvs(alpha * np.ones(grid_size), size=1)[0]
    dirichlet_y = dirichlet.rvs(alpha * np.ones(grid_size), size=1)[0]
    
    # Create probability matrix for each class
    class_probs = np.zeros((num_classes, grid_size, grid_size))
    for k in range(num_classes):
        class_probs[k] = np.outer(dirichlet_x, dirichlet_y)
    
    # Normalize probabilities and ensure they sum to 1
    class_probs /= class_probs.sum(axis=(1, 2), keepdims=True)
    class_probs_flat = class_probs.reshape(num_classes, -1)
    class_probs_flat /= class_probs_flat.sum(axis=1, keepdims=True)
    
    # Assign data points and locations to clients
    client_data = [[] for _ in range(num_clients)]
    client_locations = np.random.randint(0, grid_size, size=(num_clients, 2))
    
    for i in range(len(x_train)):
        class_idx = y_train[i][0] if dataset != 'mnist' else y_train[i]
        probs = class_probs_flat[class_idx]
        grid_idx = np.random.choice(grid_size * grid_size, p=probs)
        client_idx = np.random.choice(num_clients)
        client_data[client_idx].append((x_train[i], y_train[i]))
    
    # Ensure each client has exactly samples_per_client
    for i in range(num_clients):
        if len(client_data[i]) < samples_per_client:
            additional = np.random.choice(len(client_data[i]), samples_per_client - len(client_data[i]))
            client_data[i].extend([client_data[i][j] for j in additional])
        elif len(client_data[i]) > samples_per_client:
            client_data[i] = [client_data[i][j] for j in np.random.choice(len(client_data[i]), samples_per_client, replace=False)]
    
    # Convert to numpy arrays
    for i in range(num_clients):
        x, y = zip(*client_data[i])
        client_data[i] = (np.array(x), np.array(y))
    
    return client_data, class_probs, client_locations

def distribute_clients_to_edge_servers(client_locations, num_edge_servers, grid_size):
    edge_server_size = grid_size // int(np.sqrt(num_edge_servers))
    edge_servers = []
    
    for i in range(int(np.sqrt(num_edge_servers))):
        for j in range(int(np.sqrt(num_edge_servers))):
            edge_servers.append((i * edge_server_size, j * edge_server_size, 
                                 (i+1) * edge_server_size, (j+1) * edge_server_size))
    
    client_assignment = [[] for _ in range(num_edge_servers)]
    
    for client_idx, (x, y) in enumerate(client_locations):
        for server_idx, (x1, y1, x2, y2) in enumerate(edge_servers):
            if x1 <= x < x2 and y1 <= y < y2:
                client_assignment[server_idx].append(client_idx)
                break
    
    return client_assignment

# Client update function
def client_update(client_model, client_data):
    x, y = client_data
    client_model.fit(x, y, epochs=5, verbose=0)
    return client_model.get_weights()

# Regional server update function
def regional_server_update(global_weights, client_weights):
    averaged_weights = []
    for weights_list_tuple in zip(*client_weights):
        averaged_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )
    return averaged_weights

# Global server update function
def global_server_update(global_weights, regional_weights):
    averaged_weights = []
    for weights_list_tuple in zip(*regional_weights):
        averaged_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )
    return averaged_weights

def hierarchical_federated_learning(dataset, grid_size, num_clients, num_edge_servers, alpha, samples_per_client, rounds):
    # Initialize global model
    global_model = create_model(dataset)
    global_weights = global_model.get_weights()
    
    # Create client data
    client_data, class_probs, client_locations = create_client_data_dirichlet(dataset, grid_size, num_clients, alpha, samples_per_client)
    
    # Distribute clients to edge servers
    client_assignment = distribute_clients_to_edge_servers(client_locations, num_edge_servers, grid_size)
    
    for round in range(rounds):
        print(f"Round {round + 1}/{rounds}")
        
        regional_weights = []
        for server_clients in client_assignment:
            client_weights = []
            for client_idx in server_clients:
                client_model = create_model(dataset)
                client_model.set_weights(global_weights)
                client_weights.append(client_update(client_model, client_data[client_idx]))
            
            if client_weights:  # Only update if there are clients in this edge server
                regional_weights.append(regional_server_update(global_weights, client_weights))
        
        global_weights = global_server_update(global_weights, regional_weights)
        global_model.set_weights(global_weights)
        
        # Evaluate global model
        _, (x_test, y_test) = load_and_preprocess_data(dataset)
        test_loss, test_accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"Global model accuracy: {test_accuracy:.4f}")

    return global_model, client_assignment

# Function to analyze data distribution among clients and edge servers
def analyze_client_data(client_data, dataset, client_assignment):
    num_classes = 10 if dataset in ['mnist', 'cifar10'] else 100
    client_labels = []
    for x, y in client_data:
        unique_labels, counts = np.unique(y, return_counts=True)
        client_labels.append(dict(zip(unique_labels, counts)))
    
    print("Data distribution among edge servers and clients:")
    for server_idx, server_clients in enumerate(client_assignment):
        print(f"Edge Server {server_idx + 1} (Clients: {len(server_clients)}):")
        for client_idx in server_clients:
            print(f"  Client {client_idx}: {client_labels[client_idx]}")
    
    print("\nLabel distribution across all clients:")
    label_counts = {i: 0 for i in range(num_classes)}
    for labels in client_labels:
        for label, count in labels.items():
            label_counts[label] += count
    
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")

# Function to visualize client distribution and edge server areas
def visualize_client_distribution(client_locations, client_assignment, grid_size, num_edge_servers):
    plt.figure(figsize=(10, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_edge_servers))
    
    # Plot client locations
    for server_idx, server_clients in enumerate(client_assignment):
        server_client_locations = client_locations[server_clients]
        plt.scatter(server_client_locations[:, 0], server_client_locations[:, 1], 
                    color=colors[server_idx], alpha=0.6, label=f'Server {server_idx + 1}')
    
    # Plot edge server areas
    edge_server_size = grid_size // int(np.sqrt(num_edge_servers))
    for i in range(int(np.sqrt(num_edge_servers))):
        for j in range(int(np.sqrt(num_edge_servers))):
            x1, y1 = i * edge_server_size, j * edge_server_size
            x2, y2 = (i+1) * edge_server_size, (j+1) * edge_server_size
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'k-')
    
    plt.title("Client Distribution and Edge Server Areas")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the hierarchical federated learning process
dataset = 'mnist'
grid_size = 100  # 100x100 grid for data distribution
num_clients = 100
num_edge_servers = 9  # 3x3 grid of edge servers
alpha = 0.1  # Dirichlet concentration parameter (lower values = more non-IID)
samples_per_client = 100
rounds = 2

final_model, client_assignment = hierarchical_federated_learning(dataset, grid_size, num_clients, num_edge_servers, alpha, samples_per_client, rounds)

# Analyze and visualize the results
client_data, class_probs, client_locations = create_client_data_dirichlet(dataset, grid_size, num_clients, alpha, samples_per_client)
analyze_client_data(client_data, dataset, client_assignment)
visualize_client_distribution(client_locations, client_assignment, grid_size, num_edge_servers)