import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist, cifar10, cifar100
from collections import defaultdict
from typing import List, Dict, Tuple
import random
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Circle, Rectangle

class SimpleCNN(tf.keras.Model):
    def __init__(self, num_classes=10, input_shape=(32, 32, 3)):
        super(SimpleCNN, self).__init__()
        
        # Define layers
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(num_classes)
        
        # Build the model
        self.build((None,) + input_shape)

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
        
    def compile_and_build(self):
        """Compile and build the model"""
        self.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Build the model with the correct input shape
        if hasattr(self, 'input_shape'):
            self.build((None,) + self.input_shape)

class HierFedLearning:
    def __init__(
        self,
        dataset_name: str,
        total_rounds: int,
        num_clients: int,
        samples_per_client: int,
        num_edge_servers: int,
        grid_size: int,
        alpha: float  # Dirichlet distribution parameter
    ):
        self.dataset_name = dataset_name.lower()
        self.total_rounds = total_rounds
        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        self.num_edge_servers = num_edge_servers
        self.grid_size = grid_size
        self.alpha = alpha
        
        # Initialize dataset
        self.load_dataset()
        
        # Initialize model
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
        self.global_model.compile_and_build()
        
        # Initialize client locations and data distribution
        self.setup_topology()
        
        # Store class names for visualization
        self.class_names = self.get_class_names()

    def get_class_names(self):
        """Get class names for the selected dataset"""
        if self.dataset_name == "cifar-10":
            return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.dataset_name == "cifar-100":
            return [f'class_{i}' for i in range(100)]
        elif self.dataset_name == "mnist":
            return [str(i) for i in range(10)]
        
    def load_dataset(self):
        """Load and preprocess the selected dataset"""
        if self.dataset_name == "mnist":
            (x_train, y_train), _ = mnist.load_data()
            # Reshape and normalize MNIST
            x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            
        elif self.dataset_name == "cifar-10":
            (x_train, y_train), _ = cifar10.load_data()
            # Normalize CIFAR-10
            x_train = x_train.astype('float32') / 255.0
            y_train = y_train.squeeze()
            
        elif self.dataset_name == "cifar-100":
            (x_train, y_train), _ = cifar100.load_data()
            # Normalize CIFAR-100
            x_train = x_train.astype('float32') / 255.0
            y_train = y_train.squeeze()
            
        self.x_train = x_train
        self.y_train = y_train
        
    def setup_topology(self):
        """Initialize the network topology"""
        # Generate grid points for edge servers
        self.edge_points = self.generate_edge_server_locations()
        
        # Generate client locations on the grid
        self.client_locations = self.generate_client_locations()
        
        # Generate Dirichlet distribution for each grid point
        self.label_distributions = self.generate_label_distributions()
        
        # Assign clients to edge servers
        self.client_assignments = self.assign_clients_to_edges(
            self.client_locations, self.edge_points)
        
        # Distribute data to clients
        self.client_data = self.distribute_data_to_clients(self.client_locations)

    def generate_edge_server_locations(self) -> List[Tuple[float, float]]:
        """Generate evenly distributed edge server locations"""
        edge_points = []
        rows = int(np.sqrt(self.num_edge_servers))
        cols = self.num_edge_servers // rows
        
        for i in range(rows):
            for j in range(cols):
                x = (i + 0.5) * (self.grid_size / rows)
                y = (j + 0.5) * (self.grid_size / cols)
                edge_points.append((x, y))
                
        return edge_points

    def generate_client_locations(self) -> List[Tuple[float, float]]:
        """Generate random client locations on the grid"""
        return [(random.uniform(0, self.grid_size), 
                random.uniform(0, self.grid_size)) 
                for _ in range(self.num_clients)]

    def generate_label_distributions(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Generate Dirichlet distribution for each grid point"""
        distributions = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                distributions[(i, j)] = dirichlet.rvs(
                    [self.alpha] * self.num_classes)[0]
        return distributions

    def assign_clients_to_edges(
        self,
        client_locations: List[Tuple[float, float]],
        edge_points: List[Tuple[float, float]]
    ) -> Dict[int, List[int]]:
        """Assign clients to nearest edge server based on Euclidean distance"""
        assignments = defaultdict(list)
        
        for client_idx, client_loc in enumerate(client_locations):
            distances = [np.sqrt((client_loc[0] - edge[0])**2 + 
                               (client_loc[1] - edge[1])**2) 
                       for edge in edge_points]
            nearest_edge = np.argmin(distances)
            assignments[nearest_edge].append(client_idx)
            
        return assignments

    def distribute_data_to_clients(
        self,
        client_locations: List[Tuple[float, float]]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Distribute data to clients based on their location and label distribution"""
        client_data = {}
        self.client_label_counts = defaultdict(lambda: defaultdict(int))
        
        # Get indices for each class
        class_indices = defaultdict(list)
        for idx, label in enumerate(self.y_train):
            class_indices[label].append(idx)
            
        for client_idx, location in enumerate(client_locations):
            grid_x, grid_y = int(location[0]), int(location[1])
            dist = self.label_distributions[(grid_x, grid_y)]
            
            client_indices = []
            remaining_samples = self.samples_per_client
            
            while remaining_samples > 0:
                class_label = np.random.choice(self.num_classes, p=dist)
                
                if class_indices[class_label]:
                    sampled_idx = random.choice(class_indices[class_label])
                    client_indices.append(sampled_idx)
                    class_indices[class_label].remove(sampled_idx)
                    remaining_samples -= 1
                    self.client_label_counts[client_idx][class_label] += 1
            
            # Store actual data for each client
            client_data[client_idx] = {
                'x': self.x_train[client_indices],
                'y': self.y_train[client_indices]
            }
            
        return client_data

    def train_client(self, client_idx: int, model: tf.keras.Model, epochs: int = 1):
        """Train the model on a single client's data"""
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        
        # Get client's data
        client_x = self.client_data[client_idx]['x']
        client_y = self.client_data[client_idx]['y']
        
        # Convert labels to one-hot encoding
        client_y = tf.keras.utils.to_categorical(client_y, self.num_classes)
        
        # Compile model
        model.compile(optimizer=optimizer,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Train model
        model.fit(client_x, client_y, 
                 epochs=epochs, 
                 batch_size=32, 
                 verbose=0)
        
        return model.get_weights()

    def aggregate_models(self, model_weights_list: List[List[np.ndarray]]):
        """Aggregate model parameters using FedAvg"""
        avg_weights = []
        for weights_list_tuple in zip(*model_weights_list):
            avg_weights.append(np.mean(weights_list_tuple, axis=0))
        return avg_weights

    def visualize_topology(self, show_grid: bool = True, show_distances: bool = False):
        """
        Visualize the distribution of clients and edge servers on the grid
        
        Args:
            show_grid: If True, show the grid lines
            show_distances: If True, show lines connecting clients to their edge servers
        """
        # Create figure
        plt.figure(figsize=(12, 12))
        
        # Set up the plot
        plt.xlim(-0.5, self.grid_size + 0.5)
        plt.ylim(-0.5, self.grid_size + 0.5)
        
        # Draw grid if requested
        if show_grid:
            for i in range(self.grid_size + 1):
                plt.axhline(y=i, color='gray', linestyle=':', alpha=0.3)
                plt.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
        
        # Generate colors for each edge server
        num_edges = len(self.edge_points)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_edges))
        
        # Plot edge servers and their coverage
        for edge_idx, (edge_x, edge_y) in enumerate(self.edge_points):
            # Plot edge server as a larger point
            plt.scatter(edge_x, edge_y, c=[colors[edge_idx]], s=200, marker='s', 
                       label=f'Edge Server {edge_idx}')
            
            # Get all clients assigned to this edge server
            assigned_clients = self.client_assignments[edge_idx]
            client_points = [self.client_locations[i] for i in assigned_clients]
            
            # Plot clients with same color as their edge server
            if client_points:
                client_x, client_y = zip(*client_points)
                plt.scatter(client_x, client_y, c=[colors[edge_idx]], s=50, alpha=0.5)
                
                # Draw lines to show assignment if requested
                if show_distances:
                    for cx, cy in zip(client_x, client_y):
                        plt.plot([edge_x, cx], [edge_y, cy], 
                               c=colors[edge_idx], alpha=0.1)
        
        # Add title and labels
        plt.title('Client and Edge Server Distribution')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add text with statistics
        stats_text = [
            f'Total Clients: {self.num_clients}',
            f'Edge Servers: {num_edges}',
            f'Grid Size: {self.grid_size}x{self.grid_size}',
            f'Alpha: {self.alpha}'
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
        
    
    def visualize_edge_coverage(self):
        """
        Visualize the coverage area of each edge server using a heatmap
        """
        resolution = 50
        x = np.linspace(0, self.grid_size, resolution)
        y = np.linspace(0, self.grid_size, resolution)
        X, Y = np.meshgrid(x, y)
        
        # For each point, determine the nearest edge server
        Z = np.zeros((resolution, resolution))
        
        for i in range(resolution):
            for j in range(resolution):
                point = (X[i, j], Y[i, j])
                distances = [np.sqrt((point[0] - ex)**2 + (point[1] - ey)**2) 
                           for ex, ey in self.edge_points]
                Z[i, j] = np.argmin(distances)
        
        plt.figure(figsize=(12, 10))
        
        # Plot the coverage areas
        plt.imshow(Z, extent=[0, self.grid_size, 0, self.grid_size], 
                  origin='lower', cmap='rainbow', alpha=0.3)
        
        # Plot edge servers
        edge_x, edge_y = zip(*self.edge_points)
        plt.scatter(edge_x, edge_y, c='black', s=200, marker='s', 
                   label='Edge Servers')
        
        # Plot clients
        client_x, client_y = zip(*self.client_locations)
        plt.scatter(client_x, client_y, c='red', s=50, alpha=0.5, 
                   label='Clients')
        
        plt.title('Edge Server Coverage Areas and Client Distribution')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.legend()
        plt.colorbar(label='Edge Server ID')
        plt.grid(True, alpha=0.3)
        plt.show()

    def train(self):
        """Perform hierarchical federated learning"""
        for round in range(self.total_rounds):
            print(f"Round {round + 1}/{self.total_rounds}")
            
            # First level: Client → Edge Server aggregation
            edge_models = {}
            
            for edge_idx, client_indices in self.client_assignments.items():
                client_weights = []
                
                # Train each client assigned to this edge server
                for client_idx in client_indices:
                    client_model = SimpleCNN(num_classes=self.num_classes,
                                           input_shape=self.input_shape)
                    client_model.compile_and_build()  # Make sure model is built
                    client_model.set_weights(self.global_model.get_weights())
                    client_weights.append(self.train_client(client_idx, client_model))
                
                # Aggregate client models at edge server
                edge_models[edge_idx] = self.aggregate_models(client_weights)
            
            # Second level: Edge Server → Global aggregation
            global_weights = self.aggregate_models(list(edge_models.values()))
            self.global_model.set_weights(global_weights)
            
        return self.global_model

# Example usage
if __name__ == "__main__":
    hierfed = HierFedLearning(
        dataset_name="mnist",
        total_rounds=10,
        num_clients=100,
        samples_per_client=500,
        num_edge_servers=4,
        grid_size=10,
        alpha=0.5
    )
    
    # Visualize the topology
    hierfed.visualize_topology(show_grid=True, show_distances=True)
    
    # Visualize edge server coverage
    hierfed.visualize_edge_coverage()
    
    # Train the model
    final_model = hierfed.train()