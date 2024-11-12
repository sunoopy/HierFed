import tensorflow as tf
from keras import layers, models
import numpy as np
from typing import List, Tuple
import time
from datetime import timedelta

class SimpleCNN(tf.keras.Model):
    def __init__(self, num_classes=10, input_shape=(32, 32, 3)):
        super(SimpleCNN, self).__init__()
        self.input_shape = input_shape
        
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape)
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(num_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
    def build_model(self):
        dummy_input = tf.keras.Input(shape=self.input_shape)
        self(dummy_input)
        self.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

class ModelTrainer:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.global_model = SimpleCNN(num_classes=num_classes, input_shape=input_shape)
        self.global_model.build_model()

    def train_client(self, client_data: dict, epochs: int = 1):
        """Train the model on a single client's data"""
        client_x = client_data['x']
        client_y = client_data['y']
        client_y = tf.keras.utils.to_categorical(client_y, self.num_classes)
        
        # Create and initialize client model
        client_model = SimpleCNN(num_classes=self.num_classes, input_shape=self.input_shape)
        client_model.build_model()
        client_model.set_weights(self.global_model.get_weights())
        
        history = client_model.fit(
            client_x, client_y,
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        
        loss, accuracy = client_model.evaluate(client_x, client_y, verbose=0)
        return client_model.get_weights(), loss, accuracy

    def aggregate_models(self, model_weights_list: List[List[np.ndarray]]):
        """Aggregate model parameters using FedAvg"""
        avg_weights = []
        for weights_list_tuple in zip(*model_weights_list):
            avg_weights.append(np.mean(weights_list_tuple, axis=0))
        return avg_weights

    def evaluate_global_model(self, x_test, y_test):
        """Evaluate the global model on test data"""
        return self.global_model.evaluate(x_test, y_test, verbose=0)

    def train_federated(self, client_data, client_assignments, total_rounds):
        """Perform hierarchical federated learning"""
        training_history = {
            'losses': [], 'accuracies': [],
            'client_times': [], 'edge_times': [], 'total_times': []
        }
        
        for round in range(total_rounds):
            round_start_time = time.time()
            print(f"\nRound {round + 1}/{total_rounds}")
            
            # Training metrics for this round
            round_losses = []
            round_accuracies = []
            client_training_times = []
            edge_aggregation_times = []
            
            # First level: Client → Edge Server aggregation
            edge_models = {}
            
            for edge_idx, client_indices in client_assignments.items():
                edge_start_time = time.time()
                client_weights = []
                
                # Train each client
                for client_idx in client_indices:
                    client_start_time = time.time()
                    weights, loss, accuracy = self.train_client(client_data[client_idx])
                    
                    client_weights.append(weights)
                    round_losses.append(loss)
                    round_accuracies.append(accuracy)
                    
                    client_training_times.append(time.time() - client_start_time)
                
                # Aggregate at edge server
                edge_models[edge_idx] = self.aggregate_models(client_weights)
                edge_aggregation_times.append(time.time() - edge_start_time)
            
            # Second level: Edge Server → Global aggregation
            global_weights = self.aggregate_models(list(edge_models.values()))
            self.global_model.set_weights(global_weights)
            
            # Calculate timing metrics
            total_round_time = time.time() - round_start_time
            
            # Update history
            training_history['losses'].append(np.mean(round_losses))
            training_history['accuracies'].append(np.mean(round_accuracies))
            training_history['client_times'].append(np.mean(client_training_times))
            training_history['edge_times'].append(np.mean(edge_aggregation_times))
            training_history['total_times'].append(total_round_time)
            
            # Print round summary
            print(f"Round {round + 1} Summary:")
            print(f"Average Training Loss: {np.mean(round_losses):.4f}")
            print(f"Average Accuracy: {np.mean(round_accuracies):.4f}")
            print(f"Average Client Training Time: {timedelta(seconds=np.mean(client_training_times))}")
            print(f"Average Edge Aggregation Time: {timedelta(seconds=np.mean(edge_aggregation_times))}")
            print(f"Total Round Time: {timedelta(seconds=total_round_time)}")
        
        return training_history
