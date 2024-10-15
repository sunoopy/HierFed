import tensorflow as tf
import numpy as np

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

# Simulate data for clients
def create_client_data(dataset, num_clients, samples_per_client):
    (x_train, y_train), _ = load_and_preprocess_data(dataset)
    
    client_data = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_data.append((x_train[start_idx:end_idx], y_train[start_idx:end_idx]))
    
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

# Main federated learning process
def hierarchical_federated_learning(dataset, num_regions, clients_per_region, samples_per_client, rounds):
    # Initialize global model
    global_model = create_model(dataset)
    global_weights = global_model.get_weights()
    
    # Create client data
    total_clients = num_regions * clients_per_region
    client_data = create_client_data(dataset, total_clients, samples_per_client)
    
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

# Run the hierarchical federated learning process
dataset = 'mnist'  # Change this to 'cifar10' or 'cifar100' as needed
num_regions = 3
clients_per_region = 5
samples_per_client = 1000
rounds = 5

final_model = hierarchical_federated_learning(dataset, num_regions, clients_per_region, samples_per_client, rounds)