import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist, cifar10, cifar100
from collections import defaultdict
from typing import List, Dict, Tuple
import random
from scipy.stats import dirichlet
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import timedelta

class SimpleCNN(tf.keras.Model):
    def __init__(self, num_classes=10, input_shape=(32, 32, 3)):
        super(SimpleCNN, self).__init__()
        self.input_shape = input_shape
        
        # Define layers
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
        """Build the model by passing a dummy input"""
        dummy_input = tf.keras.Input(shape=self.input_shape)
        self(dummy_input)  # This triggers the model building
        self.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

class HierFedLearning:
    def __init__(
        self,
        dataset_name: str,
        total_rounds: int,
        num_clients: int,
        samples_per_client: int,
        num_edge_servers: int,
        grid_size: int,
        alpha: float
    ):
        # ... (keep existing initialization code)
        self.dataset_name = dataset_name.lower()
        self.total_rounds = total_rounds
        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        self.num_edge_servers = num_edge_servers
        self.grid_size = grid_size
        self.alpha = alpha
        
        # Initialize dataset
        self.load_dataset()
        
        # Set input shape and number of classes based on dataset
        if self.dataset_name == "mnist":
            self.input_shape = (28, 28, 1)
            self.num_classes = 10
        elif self.dataset_name == "cifar-10":
            self.input_shape = (32, 32, 3)
            self.num_classes = 10
        elif self.dataset_name == "cifar-100":
            self.input_shape = (32, 32, 3)
            self.num_classes = 100
        else:
            raise ValueError("Dataset must be 'mnist', 'cifar-10', or 'cifar-100'")
            
        # Initialize and build global model
        self.global_model = SimpleCNN(num_classes=self.num_classes, 
                                    input_shape=self.input_shape)
        self.global_model.build_model()
        
        # Initialize client locations and data distribution
        self.setup_topology()
        self.load_test_data()

    def setup_topology(self):
        """Initialize the network topology with optimal edge server placement"""
        # First, generate random client locations
        self.client_locations = self.generate_client_locations()
        
        # Then, determine optimal edge server locations based on client distribution
        self.edge_points = self.optimize_edge_server_locations()
        
        # Generate Dirichlet distribution for each grid point
        self.label_distributions = self.generate_label_distributions()
        
        # Assign clients to nearest edge servers
        self.client_assignments = self.assign_clients_to_edges(
            self.client_locations, self.edge_points)
        
        # Distribute data to clients
        self.client_data = self.distribute_data_to_clients(self.client_locations)

    def generate_client_locations(self) -> List[Tuple[float, float]]:
        """Generate random client locations on the grid"""
        return [(random.uniform(0, self.grid_size), 
                random.uniform(0, self.grid_size)) 
                for _ in range(self.num_clients)]

    def optimize_edge_server_locations(self) -> List[Tuple[float, float]]:
        """
        Determine optimal edge server locations using K-means clustering
        to minimize average distance between clients and their assigned edge servers
        """
        # Convert client locations to numpy array for K-means
        client_points = np.array(self.client_locations)
        
        # Use K-means to find optimal edge server locations
        kmeans = KMeans(n_clusters=self.num_edge_servers, 
                       random_state=42, 
                       n_init=10)
        kmeans.fit(client_points)
        
        # Get the cluster centers as edge server locations
        edge_points = [(float(x), float(y)) for x, y in kmeans.cluster_centers_]
        
        # Ensure edge servers are within grid boundaries
        edge_points = [(min(max(x, 0), self.grid_size), 
                       min(max(y, 0), self.grid_size)) 
                      for x, y in edge_points]
        
        return edge_points

    def calculate_average_delay(self) -> float:
        """
        Calculate average communication delay (based on distance) 
        between clients and their assigned edge servers
        """
        total_delay = 0
        for edge_idx, client_indices in self.client_assignments.items():
            edge_x, edge_y = self.edge_points[edge_idx]
            for client_idx in client_indices:
                client_x, client_y = self.client_locations[client_idx]
                # Calculate Euclidean distance as a proxy for delay
                delay = np.sqrt((client_x - edge_x)**2 + (client_y - edge_y)**2)
                total_delay += delay
        
        return total_delay / self.num_clients

    def visualize_topology(self, show_grid: bool = True, show_distances: bool = False):
        """
        Visualize the distribution of clients and edge servers on the grid
        with additional delay information
        """
        plt.figure(figsize=(12, 12))
        
        # Set up the plot
        plt.xlim(-0.5, self.grid_size + 0.5)
        plt.ylim(-0.5, self.grid_size + 0.5)
        
        if show_grid:
            for i in range(self.grid_size + 1):
                plt.axhline(y=i, color='gray', linestyle=':', alpha=0.3)
                plt.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
        
        # Generate colors for each edge server
        num_edges = len(self.edge_points)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_edges))
        
        # Calculate average delay
        avg_delay = self.calculate_average_delay()
        
        # Plot edge servers and their coverage
        for edge_idx, (edge_x, edge_y) in enumerate(self.edge_points):
            # Plot edge server
            plt.scatter(edge_x, edge_y, c=[colors[edge_idx]], s=200, marker='s',
                       label=f'Edge Server {edge_idx}')
            
            # Get assigned clients
            assigned_clients = self.client_assignments[edge_idx]
            client_points = [self.client_locations[i] for i in assigned_clients]
            
            if client_points:
                client_x, client_y = zip(*client_points)
                plt.scatter(client_x, client_y, c=[colors[edge_idx]], s=50, alpha=0.5)
                
                if show_distances:
                    for cx, cy in zip(client_x, client_y):
                        plt.plot([edge_x, cx], [edge_y, cy],
                               c=colors[edge_idx], alpha=0.1)
        
        # Add title and labels
        plt.title('Optimized Client and Edge Server Distribution')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add statistics including delay information
        stats_text = [
            f'Total Clients: {self.num_clients}',
            f'Edge Servers: {num_edges}',
            f'Grid Size: {self.grid_size}x{self.grid_size}',
            f'Alpha: {self.alpha}',
            f'Average Delay: {avg_delay:.2f}',
        ]
        
        # Add client distribution stats
        clients_per_edge = [len(clients) for clients in self.client_assignments.values()]
        stats_text.extend([
            f'Min Clients/Edge: {min(clients_per_edge)}',
            f'Max Clients/Edge: {max(clients_per_edge)}',
            f'Avg Clients/Edge: {np.mean(clients_per_edge):.1f}'
        ])
        
        plt.text(1.05*self.grid_size, 0.5*self.grid_size,
                '\n'.join(stats_text),
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    def load_dataset(self):
        """Load and preprocess the selected dataset"""
        if self.dataset_name == "mnist":
            (x_train, y_train), _ = mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            
        elif self.dataset_name == "cifar-10":
            (x_train, y_train), _ = cifar10.load_data()
            x_train = x_train.astype('float32') / 255.0
            y_train = y_train.squeeze()
            
        elif self.dataset_name == "cifar-100":
            (x_train, y_train), _ = cifar100.load_data()
            x_train = x_train.astype('float32') / 255.0
            y_train = y_train.squeeze()
            
        self.x_train = x_train
        self.y_train = y_train

    def load_test_data(self):
        """Load and preprocess the test dataset"""
        if self.dataset_name == "mnist":
            _, (x_test, y_test) = mnist.load_data()
            x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            
        elif self.dataset_name == "cifar-10":
            _, (x_test, y_test) = cifar10.load_data()
            x_test = x_test.astype('float32') / 255.0
            y_test = y_test.squeeze()
            
        elif self.dataset_name == "cifar-100":
            _, (x_test, y_test) = cifar100.load_data()
            x_test = x_test.astype('float32') / 255.0
            y_test = y_test.squeeze()
            
        self.x_test = x_test
        self.y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
    
    def evaluate_global_model(self):
        """Evaluate the global model on test data"""
        test_loss, test_accuracy = self.global_model.evaluate(
            self.x_test, self.y_test, verbose=0
        )
        return test_loss, test_accuracy

    