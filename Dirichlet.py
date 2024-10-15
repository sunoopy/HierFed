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



def create_client_data_dirichlet(dataset, grid_size, clients_per_region, alpha, samples_per_client):
    (x_train, y_train), _ = load_and_preprocess_data(dataset)
    num_classes = 10 if dataset in ['mnist', 'cifar10'] else 100
    num_clients = sum(clients_per_region)
    
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
    
    # Assign data points to clients
    client_data = [[] for _ in range(num_clients)]
    for i in range(len(x_train)):
        class_idx = y_train[i][0] if dataset != 'mnist' else y_train[i]
        probs = class_probs_flat[class_idx]
        grid_idx = np.random.choice(grid_size * grid_size, p=probs)
        client_idx = np.random.choice(num_clients)  # Randomly assign to any client
        client_data[client_idx].append((x_train[i], y_train[i]))
    
    # Ensure each client has exactly samples_per_client
    for i in range(num_clients):
        if len(client_data[i]) < samples_per_client:
            # If not enough samples, randomly duplicate existing samples
            additional = np.random.choice(len(client_data[i]), samples_per_client - len(client_data[i]))
            client_data[i].extend([client_data[i][j] for j in additional])
        elif len(client_data[i]) > samples_per_client:
            # If too many samples, randomly select samples_per_client
            client_data[i] = [client_data[i][j] for j in np.random.choice(len(client_data[i]), samples_per_client, replace=False)]
    
    # Convert to numpy arrays
    for i in range(num_clients):
        x, y = zip(*client_data[i])
        client_data[i] = (np.array(x), np.array(y))
    
    return client_data, class_probs

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

def hierarchical_federated_learning(dataset, grid_size, clients_per_region, alpha, samples_per_client, rounds):
    # Initialize global model
    global_model = create_model(dataset)
    global_weights = global_model.get_weights()
    
    # Create client data
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
        
        # Evaluate global model
        _, (x_test, y_test) = load_and_preprocess_data(dataset)
        test_loss, test_accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"Global model accuracy: {test_accuracy:.4f}")

    return global_model

# Function to analyze data distribution among clients
def analyze_client_data(client_data, dataset, clients_per_region):
    num_classes = 10 if dataset in ['mnist', 'cifar10'] else 100
    client_labels = []
    for x, y in client_data:
        unique_labels, counts = np.unique(y, return_counts=True)
        client_labels.append(dict(zip(unique_labels, counts)))
    
    print("Data distribution among clients:")
    client_idx = 0
    for region, region_clients in enumerate(clients_per_region):
        print(f"Edge Server {region + 1}:")
        for _ in range(region_clients):
            print(f"  Client {client_idx}: {client_labels[client_idx]}")
            client_idx += 1
    
    print("\nLabel distribution across all clients:")
    label_counts = {i: 0 for i in range(num_classes)}
    for labels in client_labels:
        for label, count in labels.items():
            label_counts[label] += count
    
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")

# Function to visualize class probabilities at a specific grid location
def visualize_class_probabilities(class_probs, grid_size, x, y):
    num_classes = class_probs.shape[0]
    probs = class_probs[:, x, y]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), probs)
    plt.title(f"Class Probabilities at Grid Location ({x}, {y})")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.xticks(range(num_classes))
    plt.ylim(0, 1)
    plt.show()

# Run the hierarchical federated learning process
dataset = 'mnist'  # Change this to 'cifar10' or 'cifar100' as needed
grid_size = 10  # 10x10 grid for data distribution
clients_per_region = [10, 100, 2]  # Different number of clients for each edge server
alpha = 0.5  # Dirichlet concentration parameter (lower values = more non-IID)
samples_per_client = 100
rounds = 2

client_data, class_probs = create_client_data_dirichlet(dataset, grid_size, clients_per_region, alpha, samples_per_client)
analyze_client_data(client_data, dataset, clients_per_region)

# Visualize class probabilities for a random grid location
random_x, random_y = np.random.randint(0, grid_size, 2)
visualize_class_probabilities(class_probs, grid_size, random_x, random_y)

final_model = hierarchical_federated_learning(dataset, grid_size, clients_per_region, alpha, samples_per_client, rounds)