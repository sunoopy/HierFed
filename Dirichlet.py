import tensorflow as tf
import numpy as np
from scipy.stats import dirichlet

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
    
    # Normalize probabilities
    class_probs /= class_probs.sum(axis=0, keepdims=True)
    
    # Assign data points to clients
    client_data = [[] for _ in range(num_clients)]
    for i in range(len(x_train)):
        class_idx = y_train[i][0] if dataset != 'mnist' else y_train[i]
        probs = class_probs[class_idx].flatten()
        grid_idx = np.random.choice(grid_size * grid_size, p=probs)
        client_idx = grid_idx % num_clients  # Map grid cells to clients
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
    
    return client_data

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

def hierarchical_federated_learning(dataset, grid_size, num_regions, clients_per_region, alpha, samples_per_client, rounds):
    # Initialize global model
    global_model = create_model(dataset)
    global_weights = global_model.get_weights()
    
    num_clients = num_regions * clients_per_region
    # Create client data
    client_data = create_client_data_dirichlet(dataset, grid_size, num_clients, alpha, samples_per_client)
    
    for round in range(rounds):
        print(f"Round {round + 1}/{rounds}")
        
        regional_weights = []
        for region in range(num_regions):
            client_weights = []
            for client in range(clients_per_region):
                client_idx = region * clients_per_region + client
                client_model = create_model(dataset)
                client_model.set_weights(global_weights)
                client_weights.append(client_update(client_model, client_data[client_idx]))
            
            regional_weights.append(regional_server_update(global_weights, client_weights))
        
        global_weights = global_server_update(global_weights, regional_weights)
        global_model.set_weights(global_weights)
        
        # Evaluate global model
        _, (x_test, y_test) = load_and_preprocess_data(dataset)
        test_loss, test_accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"Global model accuracy: {test_accuracy:.4f}")

    return global_model

# Function to analyze data distribution among clients
def analyze_client_data(client_data, dataset):
    num_classes = 10 if dataset in ['mnist', 'cifar10'] else 100
    client_labels = []
    for x, y in client_data:
        unique_labels, counts = np.unique(y, return_counts=True)
        client_labels.append(dict(zip(unique_labels, counts)))
    
    print("Data distribution among clients:")
    for i, labels in enumerate(client_labels):
        print(f"Client {i}: {labels}")
    
    print("\nLabel distribution across clients:")
    label_counts = {i: 0 for i in range(num_classes)}
    for labels in client_labels:
        for label, count in labels.items():
            label_counts[label] += count
    
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")

# Run the hierarchical federated learning process
dataset = 'mnist'  # Change this to 'cifar10' or 'cifar100' as needed
grid_size = 10  # 10x10 grid for data distribution
num_regions = 5  # Number of regions
clients_per_region = 4  # Number of clients per region
alpha = 0.5  # Dirichlet concentration parameter (lower values = more non-IID)
samples_per_client = 1000
rounds = 5

num_clients = num_regions * clients_per_region
client_data = create_client_data_dirichlet(dataset, grid_size, num_clients, alpha, samples_per_client)
analyze_client_data(client_data, dataset)

final_model = hierarchical_federated_learning(dataset, grid_size, num_regions, clients_per_region, alpha, samples_per_client, rounds)