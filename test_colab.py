import tensorflow as tf
import numpy as np
from collections import defaultdict

# Previous model creation and other utility functions remain the same
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

def apply_dirichlet_pixel_distribution(x_data, alpha_pixel=1.0):
    """Apply Dirichlet distribution to pixel values"""
    shape = x_data.shape
    # Reshape to 2D array (samples, features)
    x_flat = x_data.reshape(shape[0], -1)
    
    # Generate Dirichlet weights for each feature
    dirichlet_weights = np.random.dirichlet(np.ones(x_flat.shape[1]) * alpha_pixel, size=1)[0]
    
    # Apply weights to features
    x_transformed = x_flat * dirichlet_weights
    
    # Reshape back to original shape
    return x_transformed.reshape(shape)

def distribute_data_dirichlet(x_train, y_train, num_clients, num_classes, alpha_label=1.0):
    """Distribute data to clients using Dirichlet distribution for labels"""
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
        
        # Apply pixel-wise Dirichlet distribution
        x_client = apply_dirichlet_pixel_distribution(x_train[client_indices])
        y_client = y_train[client_indices]
        
        client_data.append((x_client, y_client))
    
    return client_data

def create_client_data(dataset, num_clients, samples_per_client, alpha_label=1.0, alpha_pixel=1.0):
    """Create client data with Dirichlet distribution for both labels and pixels"""
    (x_train, y_train), _ = load_and_preprocess_data(dataset)
    num_classes = 10 if dataset in ['mnist', 'cifar10'] else 100
    
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

# Client update function remains the same
def client_update(client_model, client_data):
    x, y = client_data
    client_model.fit(x, y, epochs=5, verbose=0)
    return client_model.get_weights()

# Regional and global server update functions remain the same
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

def hierarchical_federated_learning(dataset, num_regions, clients_per_region, 
                                  samples_per_client, rounds, alpha_label=1.0, 
                                  alpha_pixel=1.0):
    """Main federated learning process with Dirichlet distribution"""
    # Initialize global model
    global_model = create_model(dataset)
    global_weights = global_model.get_weights()
    
    # Create client data with Dirichlet distribution
    total_clients = num_regions * clients_per_region
    client_data = create_client_data(dataset, total_clients, samples_per_client, 
                                   alpha_label, alpha_pixel)
    
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

def analyze_client_data(client_data):
    """Analyze data distribution among clients"""
    client_labels = []
    client_distributions = []
    
    for x, y in client_data:
        unique_labels, counts = np.unique(y, return_counts=True)
        distribution = counts / len(y)
        client_labels.append(set(unique_labels))
        client_distributions.append(distribution)
    
    print("Data distribution among clients:")
    for i, (labels, dist) in enumerate(zip(client_labels, client_distributions)):
        print(f"\nClient {i}:")
        print(f"Labels present: {labels}")
        print("Label distribution:")
        for label, prob in enumerate(dist):
            if prob > 0:
                print(f"  Label {label}: {prob:.3f}")

# Example usage
dataset = 'mnist'
num_regions = 3
clients_per_region = 5
samples_per_client = 1000
rounds = 1
alpha_label = 0.5  # Lower alpha means more non-IID (label-wise)
alpha_pixel = 1.0  # Lower alpha means more non-IID (pixel-wise)

final_model = hierarchical_federated_learning(
    dataset, 
    num_regions, 
    clients_per_region, 
    samples_per_client, 
    rounds, 
    alpha_label,
    alpha_pixel
)

# Analyze the data distribution
total_clients = num_regions * clients_per_region
client_data = create_client_data(dataset, total_clients, samples_per_client, 
                               alpha_label, alpha_pixel)
analyze_client_data(client_data)