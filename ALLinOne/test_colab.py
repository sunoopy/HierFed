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
import time
from datetime import timedelta

plt.ion()  # Enable interactive mode
import matplotlib
matplotlib.use('Agg')

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
    def calculate_optimal_edge_positions(grid_size: int, num_edges: int, coverage_radius: float) -> List[Tuple[float, float]]:
        """
        Calculate optimal edge server positions based on grid size and number of edges.
        Returns list of (x,y) coordinates for edge server placement.
        """
        if num_edges == 2:
            # For 2 edges, place them at 1/4 and 3/4 of the grid horizontally, centered vertically
            return [
                (grid_size * 0.25, grid_size * 0.5),
                (grid_size * 0.75, grid_size * 0.5)
            ]
        elif num_edges == 3:
            # For 3 edges, place in triangle formation
            return [
                (grid_size * 0.5, grid_size * 0.75),
                (grid_size * 0.25, grid_size * 0.25),
                (grid_size * 0.75, grid_size * 0.25)
            ]
        elif num_edges == 4:
            # For 4 edges, place in square formation
            d = grid_size * 0.25  # Distance from edge
            return [
                (d, d),
                (grid_size - d, d),
                (d, grid_size - d),
                (grid_size - d, grid_size - d)
            ]
        elif num_edges == 5:
            # For 5 edges, place in pentagon formation
            center = grid_size * 0.5
            radius = grid_size * 0.35
            angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 equally spaced angles
            return [
                (center + radius * np.cos(angle), center + radius * np.sin(angle))
                for angle in angles
            ]
        elif num_edges == 6:
            # For 6 edges, place in hexagon formation
            center = grid_size * 0.5
            radius = grid_size * 0.35
            angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 equally spaced angles
            return [
                (center + radius * np.cos(angle), center + radius * np.sin(angle))
                for angle in angles
            ]
        elif num_edges == 7:
            # For 7 edges, place in hexagon formation with center point
            center = grid_size * 0.5
            radius = grid_size * 0.35
            positions = [(center, center)]  # Center point
            angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 equally spaced angles
            positions.extend([
                (center + radius * np.cos(angle), center + radius * np.sin(angle))
                for angle in angles
            ])
            return positions
        elif num_edges == 8:
            # For 8 edges, place in octagon formation
            center = grid_size * 0.5
            radius = grid_size * 0.35
            angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 equally spaced angles
            return [
                (center + radius * np.cos(angle), center + radius * np.sin(angle))
                for angle in angles
            ]
        else:
            raise ValueError("Number of edge servers must be between 2 and 8")

    def __init__(
        self,
        dataset_name: str,
        total_rounds: int,
        num_clients: int,
        samples_per_client: int,
        num_edge_servers: int,
        grid_size: int,
        alpha: float,
        coverage_radius: float = None
    ):
        if not 2 <= num_edge_servers <= 8:
            raise ValueError("Number of edge servers must be between 2 and 8")
            
        self.dataset_name = dataset_name.lower()
        self.total_rounds = total_rounds
        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        self.num_edge_servers = num_edge_servers
        self.grid_size = grid_size
        self.alpha = alpha
        self.coverage_radius = coverage_radius or (grid_size / np.sqrt(num_edge_servers))
        
        # Initialize dataset and other attributes
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

    def generate_edge_server_locations(self) -> List[Tuple[float, float]]:
        """Generate optimal edge server locations based on number of edges"""
        return self.calculate_optimal_edge_positions(
            self.grid_size,
            self.num_edge_servers,
            self.coverage_radius
        )

    def analyze_coverage(self) -> Dict[str, float]:
        """
        Analyze the coverage and overlap of the current edge server configuration
        """
        resolution = 100
        x = np.linspace(0, self.grid_size, resolution)
        y = np.linspace(0, self.grid_size, resolution)
        X, Y = np.meshgrid(x, y)
        coverage_count = np.zeros((resolution, resolution))
        
        for i in range(resolution):
            for j in range(resolution):
                point = (X[i, j], Y[i, j])
                for edge_x, edge_y in self.edge_points:
                    dist = np.sqrt((point[0] - edge_x)**2 + (point[1] - edge_y)**2)
                    if dist <= self.coverage_radius:
                        coverage_count[i, j] += 1
        
        total_area = self.grid_size * self.grid_size
        covered_area = np.sum(coverage_count > 0) / (resolution * resolution) * total_area
        coverage_percentage = (covered_area / total_area) * 100
        
        overlap_area = np.sum(coverage_count > 1) / (resolution * resolution) * total_area
        overlap_percentage = (overlap_area / total_area) * 100
        
        efficiency = coverage_percentage / (overlap_percentage + 1)  # Add 1 to avoid division by zero
        
        return {
            'coverage_percentage': coverage_percentage,
            'overlap_percentage': overlap_percentage,
            'efficiency': efficiency
        }
   
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
        """Generate evenly distributed edge server locations with minimum distance constraints"""
        edge_points = []
        min_distance = self.coverage_radius * 1.5  # Minimum distance between edge servers
        
        while len(edge_points) < self.num_edge_servers:
            # Generate a candidate point
            x = random.uniform(self.coverage_radius, self.grid_size - self.coverage_radius)
            y = random.uniform(self.coverage_radius, self.grid_size - self.coverage_radius)
            
            # Check distance from existing points
            if not edge_points:
                edge_points.append((x, y))
                continue
                
            distances = [np.sqrt((x - ex)**2 + (y - ey)**2) for ex, ey in edge_points]
            if min(distances) >= min_distance:
                edge_points.append((x, y))
        
        return edge_points
    
    def is_point_in_coverage(self, point: Tuple[float, float], 
                           edge_point: Tuple[float, float]) -> bool:
        """Check if a point is within the circular coverage area of an edge server"""
        distance = np.sqrt((point[0] - edge_point[0])**2 + 
                         (point[1] - edge_point[1])**2)
        return distance <= self.coverage_radius

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
        """Assign clients to edge servers based on circular coverage areas"""
        assignments = defaultdict(list)
        unassigned_clients = []
        
        # First pass: assign clients to their nearest edge server within coverage
        for client_idx, client_loc in enumerate(client_locations):
            assigned = False
            distances = [(idx, np.sqrt((client_loc[0] - edge[0])**2 + 
                                     (client_loc[1] - edge[1])**2))
                        for idx, edge in enumerate(edge_points)]
            distances.sort(key=lambda x: x[1])
            
            for edge_idx, distance in distances:
                if distance <= self.coverage_radius:
                    assignments[edge_idx].append(client_idx)
                    assigned = True
                    break
            
            if not assigned:
                unassigned_clients.append(client_idx)
        
        # Second pass: assign any unassigned clients to the nearest edge server
        for client_idx in unassigned_clients:
            client_loc = client_locations[client_idx]
            distances = [(idx, np.sqrt((client_loc[0] - edge[0])**2 + 
                                     (client_loc[1] - edge[1])**2))
                        for idx, edge in enumerate(edge_points)]
            nearest_edge = min(distances, key=lambda x: x[1])[0]
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

    def train_client(self, client_idx: int, model: SimpleCNN, epochs: int = 1):
        """Train the model on a single client's data"""
        # Get client's data
        client_x = self.client_data[client_idx]['x']
        client_y = self.client_data[client_idx]['y']
        
        # Convert labels to one-hot encoding
        client_y = tf.keras.utils.to_categorical(client_y, self.num_classes)
        
        # Train model
        history = model.fit(
            client_x, client_y,
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate model on client's data
        loss, accuracy = model.evaluate(client_x, client_y, verbose=0)
        return model.get_weights(), loss, accuracy

    def aggregate_models(self, model_weights_list: List[List[np.ndarray]]):
        """Aggregate model parameters using FedAvg"""
        avg_weights = []
        for weights_list_tuple in zip(*model_weights_list):
            avg_weights.append(np.mean(weights_list_tuple, axis=0))
        return avg_weights

    def visualize_topology(self, show_grid: bool = True, show_distances: bool = False):
        """Visualize the distribution of clients and edge servers with circular coverage"""
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
        
        # Plot coverage areas first
        for edge_idx, (edge_x, edge_y) in enumerate(self.edge_points):
            circle = Circle((edge_x, edge_y), self.coverage_radius, 
                          alpha=0.2, color=colors[edge_idx])
            plt.gca().add_patch(circle)
        
        # Plot edge servers and clients
        for edge_idx, (edge_x, edge_y) in enumerate(self.edge_points):
            plt.scatter(edge_x, edge_y, c=[colors[edge_idx]], s=200, marker='s',
                       label=f'Edge Server {edge_idx}')
            
            assigned_clients = self.client_assignments[edge_idx]
            client_points = [self.client_locations[i] for i in assigned_clients]
            
            if client_points:
                client_x, client_y = zip(*client_points)
                plt.scatter(client_x, client_y, c=[colors[edge_idx]], s=50, alpha=0.5)
                
                if show_distances:
                    for cx, cy in zip(client_x, client_y):
                        plt.plot([edge_x, cx], [edge_y, cy],
                               c=colors[edge_idx], alpha=0.1)
        
        plt.title('Client and Edge Server Distribution (Circular Coverage)')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        stats_text = [
            f'Total Clients: {self.num_clients}',
            f'Edge Servers: {num_edges}',
            f'Grid Size: {self.grid_size}x{self.grid_size}',
            f'Coverage Radius: {self.coverage_radius:.2f}',
            f'Alpha: {self.alpha}'
        ]
        
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
        plt.draw()
        plt.pause(0.1)  # Add small pause to ensure display
        
    
    def visualize_edge_coverage(self):
        """Visualize the circular coverage area of each edge server"""
        resolution = 100
        x = np.linspace(0, self.grid_size, resolution)
        y = np.linspace(0, self.grid_size, resolution)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros((resolution, resolution))
        coverage_count = np.zeros((resolution, resolution))
        
        for i in range(resolution):
            for j in range(resolution):
                point = (X[i, j], Y[i, j])
                for edge_idx, edge_point in enumerate(self.edge_points):
                    if self.is_point_in_coverage(point, edge_point):
                        Z[i, j] = edge_idx + 1
                        coverage_count[i, j] += 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        im1 = ax1.imshow(Z, extent=[0, self.grid_size, 0, self.grid_size],
                        origin='lower', cmap='rainbow', alpha=0.5)
        ax1.set_title('Edge Server Coverage Areas')
        plt.colorbar(im1, ax=ax1, label='Edge Server ID')
        
        edge_x, edge_y = zip(*self.edge_points)
        ax1.scatter(edge_x, edge_y, c='black', s=200, marker='s',
                   label='Edge Servers')
        
        client_x, client_y = zip(*self.client_locations)
        ax1.scatter(client_x, client_y, c='red', s=50, alpha=0.5,
                   label='Clients')
        
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        im2 = ax2.imshow(coverage_count, extent=[0, self.grid_size, 0, self.grid_size],
                        origin='lower', cmap='YlOrRd')
        ax2.set_title('Coverage Overlap')
        plt.colorbar(im2, ax=ax2, label='Number of Overlapping Coverage Areas')
        
        ax2.scatter(edge_x, edge_y, c='black', s=200, marker='s',
                   label='Edge Servers')
        ax2.scatter(client_x, client_y, c='blue', s=50, alpha=0.5,
                   label='Clients')
        
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)  # Add small pause to ensure display

    def train(self):
        """Perform hierarchical federated learning with timing and accuracy metrics"""
        training_history = {
            'losses': [],
            'accuracies': [],
            'client_times': [],
            'edge_times': [],
            'total_times': []
        }
        
        for round in range(self.total_rounds):
            round_start_time = time.time()
            print(f"\nRound {round + 1}/{self.total_rounds}")
            
            round_losses = []
            round_accuracies = []
            client_training_times = []
            edge_aggregation_times = []
            
            # First level: Client → Edge Server aggregation
            edge_models = {}
            
            for edge_idx, client_indices in self.client_assignments.items():
                edge_start_time = time.time()
                client_weights = []
                edge_losses = []
                edge_accuracies = []
                
                # Train each client assigned to this edge server
                for client_idx in client_indices:
                    client_start_time = time.time()
                    
                    # Create and build a new client model
                    client_model = SimpleCNN(num_classes=self.num_classes,
                                           input_shape=self.input_shape)
                    client_model.build_model()
                    
                    # Set weights from global model
                    client_model.set_weights(self.global_model.get_weights())
                    
                    # Train the client model
                    weights, loss, accuracy = self.train_client(client_idx, client_model)
                    client_weights.append(weights)
                    edge_losses.append(loss)
                    edge_accuracies.append(accuracy)
                    
                    client_end_time = time.time()
                    client_training_times.append(client_end_time - client_start_time)
                
                # Aggregate client models at edge server
                edge_models[edge_idx] = self.aggregate_models(client_weights)
                round_losses.extend(edge_losses)
                round_accuracies.extend(edge_accuracies)
                
                edge_end_time = time.time()
                edge_aggregation_times.append(edge_end_time - edge_start_time)
            
            # Second level: Edge Server → Global aggregation
            global_weights = self.aggregate_models(list(edge_models.values()))
            self.global_model.set_weights(global_weights)
            
            # Evaluate global model
            test_loss, test_accuracy = self.evaluate_global_model()
            
            # Calculate timing metrics
            round_end_time = time.time()
            total_round_time = round_end_time - round_start_time
            
            # Record metrics for this round
            training_history['losses'].append(np.mean(round_losses))
            training_history['accuracies'].append(test_accuracy)
            training_history['client_times'].append(np.mean(client_training_times))
            training_history['edge_times'].append(np.mean(edge_aggregation_times))
            training_history['total_times'].append(total_round_time)
            
            # Print round summary
            print(f"Round {round + 1} Summary:")
            print(f"Average Training Loss: {np.mean(round_losses):.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Average Client Training Time: {timedelta(seconds=np.mean(client_training_times))}")
            print(f"Average Edge Aggregation Time: {timedelta(seconds=np.mean(edge_aggregation_times))}")
            print(f"Total Round Time: {timedelta(seconds=total_round_time)}")
        
        return self.global_model, training_history
    
    def plot_training_metrics(self, history):
        """Plot training metrics over rounds"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        ax1.plot(range(1, self.total_rounds + 1), history['losses'])
        ax1.set_title('Average Training Loss per Round')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.plot(range(1, self.total_rounds + 1), history['accuracies'])
        ax2.set_title('Test Accuracy per Round')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        ax3.plot(range(1, self.total_rounds + 1), history['client_times'], 
                label='Client Training')
        ax3.plot(range(1, self.total_rounds + 1), history['edge_times'], 
                label='Edge Aggregation')
        ax3.set_title('Average Times per Round')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Time (seconds)')
        ax3.legend()
        ax3.grid(True)
        
        ax4.plot(range(1, self.total_rounds + 1), history['total_times'])
        ax4.set_title('Total Round Time')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)  # Add small pause to ensure display

    def analyze_edge_server_distribution(self):
        """Analyze the distribution of labels across edge servers"""
        edge_label_distributions = defaultdict(lambda: defaultdict(int))
    
        # Aggregate label counts for each edge server
        for edge_idx, client_indices in self.client_assignments.items():
            for client_idx in client_indices:
                for label, count in self.client_label_counts[client_idx].items():
                    edge_label_distributions[edge_idx][label] += count
    
        # Convert to normalized distributions
        edge_distributions = {}
        for edge_idx, label_counts in edge_label_distributions.items():
            total_samples = sum(label_counts.values())
            edge_distributions[edge_idx] = {
                label: count/total_samples 
                for label, count in label_counts.items()
            }
    
        return edge_distributions

    def calculate_kl_divergence(self, p, q):
        """Calculate KL divergence between two distributions"""
        kl_div = 0
        for i in range(self.num_classes):
            if p.get(i, 0) > 0 and q.get(i, 0) > 0:
                kl_div += p[i] * np.log(p[i] / q[i])
        return kl_div

    def calculate_distribution_divergence(self):
        """Calculate pairwise KL divergence between edge servers"""
        edge_distributions = self.analyze_edge_server_distribution()
    
        # Calculate global distribution (average across all edge servers)
        global_dist = defaultdict(float)
        for dist in edge_distributions.values():
            for label, prob in dist.items():
                global_dist[label] += prob
    
        for label in global_dist:
            global_dist[label] /= len(edge_distributions)
    
        # Calculate KL divergence for each edge server from global distribution
        divergences = {}
        for edge_idx, dist in edge_distributions.items():
            div = self.calculate_kl_divergence(dist, global_dist)
            divergences[edge_idx] = div
    
        return divergences, edge_distributions

    def visualize_label_distributions(self):
        """Visualize the label distribution across edge servers"""
        divergences, edge_distributions = self.calculate_distribution_divergence()
    
        num_edges = len(self.edge_points)
        fig, axes = plt.subplots(2, (num_edges + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()
    
        for edge_idx, dist in edge_distributions.items():
            labels = list(range(self.num_classes))
            values = [dist.get(label, 0) for label in labels]
        
            axes[edge_idx].bar(labels, values)
            axes[edge_idx].set_title(f'Edge Server {edge_idx}\nKL Div: {divergences[edge_idx]:.4f}')
            axes[edge_idx].set_xlabel('Class Label')
            axes[edge_idx].set_ylabel('Proportion')
    
        for idx in range(len(edge_distributions), len(axes)):
            fig.delaxes(axes[idx])
    
        plt.suptitle('Label Distribution Across Edge Servers')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)  # Add small pause to ensure display

    def calculate_noniid_metrics(self):
        """Calculate and print comprehensive non-IID metrics"""
        divergences, edge_distributions = self.calculate_distribution_divergence()
    
        # Calculate various non-IID metrics
        metrics = {
        'avg_kl_divergence': np.mean(list(divergences.values())),
        'max_kl_divergence': max(divergences.values()),
        'min_kl_divergence': min(divergences.values()),
        'std_kl_divergence': np.std(list(divergences.values()))
        }
    
        # Calculate label diversity per edge server
        label_diversity = {}
        for edge_idx, dist in edge_distributions.items():
            non_zero_labels = sum(1 for v in dist.values() if v > 0.01)  # Count labels with >1% presence
            label_diversity[edge_idx] = non_zero_labels
    
        metrics.update({
        'avg_label_diversity': np.mean(list(label_diversity.values())),
        'min_label_diversity': min(label_diversity.values()),
        'max_label_diversity': max(label_diversity.values())
        })
    
        # Print detailed metrics
        print("\nNon-IID Analysis Metrics:")
        print("-" * 50)
        print(f"KL Divergence Statistics:")
        print(f"  Average: {metrics['avg_kl_divergence']:.4f}")
        print(f"  Maximum: {metrics['max_kl_divergence']:.4f}")
        print(f"  Minimum: {metrics['min_kl_divergence']:.4f}")
        print(f"  Std Dev: {metrics['std_kl_divergence']:.4f}")
        print("\nLabel Diversity Statistics:")
        print(f"  Average Labels per Edge: {metrics['avg_label_diversity']:.1f}")
        print(f"  Minimum Labels per Edge: {metrics['min_label_diversity']}")
        print(f"  Maximum Labels per Edge: {metrics['max_label_diversity']}")

        return metrics
def analyze_edge_configurations(
    dataset_name: str,
    total_rounds: int,
    num_clients: int,
    samples_per_client: int,
    grid_size: int,
    alpha: float,
    coverage_radius: float
) -> Dict[int, Dict[str, float]]:
    """
    Analyze different edge server configurations and their performance metrics.
    """
    results = {}
    
    for num_edges in range(2, 9):
        print(f"\nAnalyzing configuration with {num_edges} edge servers...")
        
        # Initialize HierFedLearning with current configuration
        hierfed = HierFedLearning(
            dataset_name=dataset_name,
            total_rounds=total_rounds,
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            num_edge_servers=num_edges,
            grid_size=grid_size,
            alpha=alpha,
            coverage_radius=coverage_radius
        )
        
        # Analyze coverage metrics
        coverage_metrics = hierfed.analyze_coverage()
        
        # Calculate non-IID metrics
        noniid_metrics = hierfed.calculate_noniid_metrics()
        
        # Store results
        results[num_edges] = {
            **coverage_metrics,
            **noniid_metrics
        }
        
        # Visualize current configuration
        hierfed.visualize_topology(show_grid=True, show_distances=True)
        hierfed.visualize_edge_coverage()
        plt.show()
    
    # Print comparison table
    print("\nConfiguration Comparison:")
    print("-" * 80)
    print(f"{'Edges':^6} | {'Coverage':^10} | {'Overlap':^10} | {'Efficiency':^10} | {'Avg KL Div':^12}")
    print("-" * 80)
    
    for num_edges, metrics in results.items():
        print(f"{num_edges:^6} | "
              f"{metrics['coverage_percentage']:^10.1f}% | "
              f"{metrics['overlap_percentage']:^10.1f}% | "
              f"{metrics['efficiency']:^10.1f} | "
              f"{metrics['avg_kl_divergence']:^12.4f}")
    
    return results



# Example usage
if __name__ == "__main__":
    # Analyze different edge server configurations
    results = analyze_edge_configurations(
        dataset_name="mnist",
        total_rounds=1,
        num_clients=100,
        samples_per_client=100,
        grid_size=10,
        alpha=0.5,
        coverage_radius=3.0
    )
    
    # Create and train a specific configuration
    hierfed = HierFedLearning(
        dataset_name="mnist",
        total_rounds=1,
        num_clients=100,
        samples_per_client=100,
        num_edge_servers=4,  # Choose number of edge servers (2-8)
        grid_size=10,
        alpha=0.5,
        coverage_radius=3.0
    )
    
    # Train the model
    final_model, history = hierfed.train()
    
    # Plot training metrics
    hierfed.plot_training_metrics(history)