import tensorflow as tf
import numpy as np
from collections import defaultdict

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

def distribute_data_dirichlet(x_train, y_train, num_clients, num_classes, alpha_label=1.0):
    """Distribute data to clients using Dirichlet distribution for labels only"""
    # Generate Dirichlet distribution for each client
    proportions = np.random.dirichlet(np.ones(num_classes) * alpha_label, size=num_clients)
    
    # Initialize client data structures
    client_data = []
    label_indices = [np.where(y_train.flatten() == i)[0] for i in range(num_classes)]
    
    # Distribute data to each client
    for client_idx in range(num_clients):
        client_indices = []
        
        # Sample data according to the client's label distribution
        for label_idx in range(num_classes):
            label_proportion = proportions[client_idx][label_idx]
            num_samples = int(label_proportion * len(label_indices[label_idx]))
            
            if num_samples > 0:
                selected_indices = np.random.choice(
                    label_indices[label_idx],
                    size=num_samples,
                    replace=False
                )
                client_indices.extend(selected_indices)
        
        # Shuffle the selected indices
        np.random.shuffle(client_indices)
        
        x_client = x_train[client_indices]
        y_client = y_train[client_indices]
        
        client_data.append((x_client, y_client))
    
    return client_data

def create_grid_client_data(dataset, grid_size, samples_per_client, alpha_label=1.0):
    """Create client data for grid-based topology with Dirichlet distribution for labels"""
    (x_train, y_train), _ = load_and_preprocess_data(dataset)
    num_classes = 10 if dataset in ['mnist', 'cifar10'] else 100
    num_clients = grid_size * grid_size
    
    # Distribute data using Dirichlet distribution
    client_data = distribute_data_dirichlet(
        x_train, 
        y_train, 
        num_clients, 
        num_classes, 
        alpha_label
    )
    
    # Normalize number of samples per client
    normalized_client_data = []
    for x, y in client_data:
        if len(x) > samples_per_client:
            indices = np.random.choice(len(x), samples_per_client, replace=False)
        else:
            indices = np.random.choice(len(x), samples_per_client, replace=True)
        
        normalized_client_data.append((x[indices], y[indices]))
    
    return normalized_client_data

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

def get_neighbor_regions(region_idx, grid_size):
    """Get indices of neighboring regions in the grid"""
    row = region_idx // grid_size
    col = region_idx % grid_size
    neighbors = []
    
    # Check all adjacent cells (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        new_row, new_col = row + dx, col + dy
        if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
            neighbor_idx = new_row * grid_size + new_col
            neighbors.append(neighbor_idx)
    
    return neighbors

def grid_federated_learning(dataset, grid_size, samples_per_client, rounds, alpha_label=1.0):
    """Main federated learning process with grid-based topology"""
    # Initialize global model
    global_model = create_model(dataset)
    global_weights = global_model.get_weights()
    
    # Create client data with Dirichlet distribution
    total_clients = grid_size * grid_size
    client_data = create_grid_client_data(dataset, grid_size, samples_per_client, alpha_label)
    
    # Track metrics for each region
    region_metrics = defaultdict(list)
    
    for round in range(rounds):
        print(f"Round {round + 1}/{rounds}")
        
        regional_weights = []
        for region in range(total_clients):
            # Get neighboring regions
            neighbors = get_neighbor_regions(region, grid_size)
            
            # Update client model with average of neighbors' weights
            client_model = create_model(dataset)
            if round > 0:
                neighbor_weights = [regional_weights[n] for n in neighbors if n < len(regional_weights)]
                if neighbor_weights:
                    averaged_neighbor_weights = regional_server_update(global_weights, neighbor_weights)
                    client_model.set_weights(averaged_neighbor_weights)
                else:
                    client_model.set_weights(global_weights)
            else:
                client_model.set_weights(global_weights)
            
            # Train client model
            new_weights = client_update(client_model, client_data[region])
            regional_weights.append(new_weights)
            
            # Evaluate regional model
            _, (x_test, y_test) = load_and_preprocess_data(dataset)
            test_loss, test_accuracy = client_model.evaluate(x_test, y_test, verbose=0)
            region_metrics[region].append(test_accuracy)
            print(f"Region ({region // grid_size}, {region % grid_size}) accuracy: {test_accuracy:.4f}")
        
        # Update global weights
        global_weights = global_server_update(global_weights, regional_weights)
        global_model.set_weights(global_weights)
        
        # Evaluate global model
        _, (x_test, y_test) = load_and_preprocess_data(dataset)
        test_loss, test_accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"Global model accuracy: {test_accuracy:.4f}")
        print()

    return global_model, region_metrics

def analyze_grid_data(client_data, grid_size):
    """Analyze data distribution among clients in the grid"""
    for i, (x, y) in enumerate(client_data):
        row, col = i // grid_size, i % grid_size
        unique_labels, counts = np.unique(y, return_counts=True)
        distribution = counts / len(y)
        
        print(f"\nRegion ({row}, {col}):")
        print(f"Labels present: {set(unique_labels)}")
        print("Label distribution:")
        for label, prob in enumerate(distribution):
            if prob > 0:
                print(f"  Label {label}: {prob:.3f}")

# Example usage
dataset = 'mnist'
grid_size = 100  # Creates a 3x3 grid of regions
samples_per_client = 100
rounds = 1
alpha_label = 0.5  # Lower alpha means more non-IID (label-wise)

final_model, region_metrics = grid_federated_learning(
    dataset, 
    grid_size, 
    samples_per_client, 
    rounds, 
    alpha_label
)

# Analyze the data distribution
client_data = create_grid_client_data(dataset, grid_size, samples_per_client, alpha_label)
analyze_grid_data(client_data, grid_size)

# Plot region performance over time
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
for region, accuracies in region_metrics.items():
    row, col = region // grid_size, region % grid_size
    plt.plot(accuracies, label=f'Region ({row}, {col})')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Region Performance Over Time')
plt.legend()
plt.grid(True)
plt.show()