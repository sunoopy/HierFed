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
import pandas as pd

class SimpleCNN(tf.keras.Model):
    def __init__(self, num_classes=10, model_input_shape=(32, 32, 3)):
        super(SimpleCNN, self).__init__()
        self.model_input_shape = model_input_shape
        
        # Define layers
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=model_input_shape)
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
        """
        Build the model by passing a dummy input
        """
        dummy_input = tf.keras.Input(shape=self.model_input_shape)
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
        sample_per_client: int,
        num_edge_servers: int,
        grid_size: int,
        coverage_radius: float,
        alpha: float
    ):
        self.dataset_name = dataset_name.lower()
        self.total_rounds = total_rounds
        self.num_clients = num_clients
        self.sample_per_client = sample_per_client
        self.num_edge_servers = num_edge_servers
        self.grid_size = grid_size
        self.alpha = alpha
        self.coverage_radius = coverage_radius
        
        # Initialize dataset
        self.load_dataset()
        
        # Set input shape and number of classes based on dataset
        if self.dataset_name == "mnist":
            self.model_input_shape = (28, 28, 1)
            self.num_classes = 10
        elif self.dataset_name == "cifar-10":
            self.model_input_shape = (32, 32, 3)
            self.num_classes = 10
        elif self.dataset_name == "cifar-100":
            self.model_input_shape = (32, 32, 3)
            self.num_classes = 100
        else:
            raise ValueError("Dataset must be 'mnist', 'cifar-10', or 'cifar-100'")
            
        # Initialize and build global model
        self.global_model = SimpleCNN(num_classes=self.num_classes, 
                                    model_input_shape=self.model_input_shape)
        self.global_model.build_model()  # Build the model 
        
        # Initialize client locations and data distribution
        self.setup_topology()
        self.load_test_data()
   
    def load_dataset(self):
        """
        Load and preprocess the selected dataset
        """
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
        """
        Load and preprocess the test dataset
        """
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
        """
        Evaluate the global model on test data
        """
        test_loss, test_accuracy = self.global_model.evaluate(
            self.x_test, self.y_test, verbose=0
        )
        return test_loss, test_accuracy
        
    def setup_topology(self):
        """
        Initialize the network topology
        """
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
        """
        Generate evenly distributed edge server locations 
        """
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
        """
        Generate random client locations on the grid 
        """
        return [(random.uniform(0, self.grid_size), 
                random.uniform(0, self.grid_size)) 
                for _ in range(self.num_clients)]

    def generate_label_distributions(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Generate Dirichlet distribution for each grid point 
        """
        distributions = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                distributions[(i, j)] = dirichlet.rvs(
                    [self.alpha] * self.num_classes)[0]
        return distributions

    def assign_clients_to_edges(self, client_locations: List[Tuple[float, float]], edge_points: List[Tuple[float, float]]) -> Dict[int, List[int]]:
        """
        Assign clients to multiple edge servers within coverage radius
    
        Args:
        client_locations: List of (x, y) coordinates for clients
        edge_points: List of (x, y) coordinates for edge servers
    
        Returns:
            Dictionary mapping edge server indices to lists of client indices
        """
        # Initialize assignments with defaultdict to allow multiple assignments
        assignments = defaultdict(list)
        unassigned_clients = []
    
        for client_idx, client_loc in enumerate(client_locations):
            # Find all edge servers within coverage radius
            nearby_edges = []
        
            for edge_idx, edge_loc in enumerate(edge_points):
                distance = np.sqrt((client_loc[0] - edge_loc[0])**2 + 
                                 (client_loc[1] - edge_loc[1])**2)
            
                # If client is within coverage radius, add to nearby edges
                if distance <= self.coverage_radius:
                    nearby_edges.append((edge_idx, distance))
        
            # Sort nearby edges by distance 
            nearby_edges.sort(key=lambda x: x[1])
        
            # If no nearby edges, add to unassigned
            if not nearby_edges:
                unassigned_clients.append(client_idx)
                continue
        
            # If multiple edges are nearby, assign to top 2 (can be adjusted)
            max_nearby = min(2, len(nearby_edges))
            for i in range(max_nearby):
                assignments[nearby_edges[i][0]].append(client_idx)
    
        if unassigned_clients:
            print(f"Warning: {len(unassigned_clients)} clients are not covered by any edge server")
    
        # Print assignment distribution for transparency
        print("\nClient Assignment Distribution:")
        for edge_idx, clients in sorted(assignments.items()):
            print(f"Edge Server {edge_idx}: {len(clients)} clients")
    
        return assignments

    def distribute_data_to_clients(self,client_locations: List[Tuple[float, float]]) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Distribute data to clients based on their location and label distribution
        """

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
            remaining_samples = self.sample_per_client
            
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
        """
        Train the model on a single client's data
        """

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
        """
        Aggregate model parameters using FedAvg
        """
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
        Visualize the circular coverage area of each edge server using a heatmap
        """
        resolution = 50
        x = np.linspace(0, self.grid_size, resolution)
        y = np.linspace(0, self.grid_size, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Initialize coverage map
        Z = np.full((resolution, resolution), -1)  # -1 indicates no coverage
        
        # Calculate coverage for each point
        for i in range(resolution):
            for j in range(resolution):
                point = (X[i, j], Y[i, j])
                min_distance = float('inf')
                nearest_edge = None
                
                # Find nearest edge server within coverage radius
                for edge_idx, (ex, ey) in enumerate(self.edge_points):
                    distance = np.sqrt((point[0] - ex)**2 + (point[1] - ey)**2)
                    if distance <= self.coverage_radius and distance < min_distance:
                        min_distance = distance
                        nearest_edge = edge_idx
                
                if nearest_edge is not None:
                    Z[i, j] = nearest_edge
        
        plt.figure(figsize=(12, 10))
        
        # Create custom colormap with transparency for uncovered areas
        cmap = plt.cm.rainbow.copy()
        cmap.set_bad('white', alpha=0)
        
        # Plot the coverage areas
        masked_Z = np.ma.masked_where(Z < 0, Z)
        plt.imshow(masked_Z, extent=[0, self.grid_size, 0, self.grid_size], 
                  origin='lower', cmap=cmap, alpha=0.3)
        
        # Plot edge servers
        edge_x, edge_y = zip(*self.edge_points)
        plt.scatter(edge_x, edge_y, c='black', s=200, marker='s', 
                   label='Edge Servers')
        
        # Draw coverage circles
        for ex, ey in self.edge_points:
            circle = plt.Circle((ex, ey), self.coverage_radius, 
                              fill=False, color='black', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
        
        # Plot clients and color them based on assignment status
        for client_idx, (cx, cy) in enumerate(self.client_locations):
            assigned = False
            for edge_clients in self.client_assignments.values():
                if client_idx in edge_clients:
                    assigned = True
                    break
            color = 'red' if assigned else 'gray'
            plt.scatter(cx, cy, c=color, s=50, alpha=0.5)
        
        plt.title('Edge Server Coverage Areas (Circular) and Client Distribution')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.legend(['Edge Servers', 'Assigned Clients', 'Uncovered Clients'])
        #plt.colorbar(label='Edge Server ID')
        plt.grid(True, alpha=0.3)
        
        # Add coverage statistics
        total_clients = len(self.client_locations)
        covered_clients = sum(len(clients) for clients in self.client_assignments.values())
        coverage_text = f'Coverage Statistics:\n' \
                       f'Total Clients: {total_clients}\n' \
                       f'Covered Clients: {covered_clients}\n' \
                       f'Coverage Rate: {(covered_clients/total_clients)*100:.1f}%'
        plt.text(1.05*self.grid_size, 0.2*self.grid_size, coverage_text,
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.show()

    def train(self):
        """Perform hierarchical federated learning with timing and accuracy metrics"""
        training_history = {
            'round': [],
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
                                           model_input_shape=self.model_input_shape)
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
            training_history['round'].append(round + 1)
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
        # Convert history to DataFrame and save to Excel
        history_df = pd.DataFrame(training_history)
        history_df.to_excel('fdalpha100.xlsx', index=False)
        print("\nTraining history saved to 'federated_learning_history.xlsx'")

        return self.global_model, history_df
    
    def plot_training_metrics(self, history):
        """Plot training metrics over rounds"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot loss
        ax1.plot(range(1, self.total_rounds + 1), history['losses'])
        ax1.set_title('Average Training Loss per Round')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(range(1, self.total_rounds + 1), history['accuracies'])
        ax2.set_title('Test Accuracy per Round')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        # Plot client and edge times
        ax3.plot(range(1, self.total_rounds + 1), history['client_times'], 
                label='Client Training')
        ax3.plot(range(1, self.total_rounds + 1), history['edge_times'], 
                label='Edge Aggregation')
        ax3.set_title('Average Times per Round')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Time (seconds)')
        ax3.legend()
        ax3.grid(True)
        
        # Plot total round time
        ax4.plot(range(1, self.total_rounds + 1), history['total_times'])
        ax4.set_title('Total Round Time')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

    def analyze_edge_server_distribution(self):
        #Analyze the distribution of labels across edge servers
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
        #Calculate KL divergence between two distributions
        kl_div = 0
        for i in range(self.num_classes):
            if p.get(i, 0) > 0 and q.get(i, 0) > 0:
                kl_div += p[i] * np.log(p[i] / q[i])
        return kl_div

    def calculate_distribution_divergence(self):
        #Calculate pairwise KL divergence between edge servers
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
        #Visualize the label distribution across edge servers
        # Get distributions
        divergences, edge_distributions = self.calculate_distribution_divergence()
    
        # Create subplot for each edge server
        num_edges = len(self.edge_points)
        fig, axes = plt.subplots(2, (num_edges + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()
    
        # Plot distribution for each edge server
        for edge_idx, dist in edge_distributions.items():
            labels = list(range(self.num_classes))
            values = [dist.get(label, 0) for label in labels]
        
            axes[edge_idx].bar(labels, values)
            axes[edge_idx].set_title(f'Edge Server {edge_idx}\nKL Div: {divergences[edge_idx]:.4f}')
            axes[edge_idx].set_xlabel('Class Label')
            axes[edge_idx].set_ylabel('Proportion')
    
        # Remove any extra subplots
        for idx in range(len(edge_distributions), len(axes)):
            fig.delaxes(axes[idx])
    
        plt.suptitle('Label Distribution Across Edge Servers')
        plt.tight_layout()
        plt.show()

    def calculate_noniid_metrics(self):
        #Calculate and print comprehensive non-IID metrics
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
    def visualize_dirichlet_distribution(self):
        """
        Visualize how the Dirichlet distribution affects label distribution across the grid
        """
        # Create subplots for each class
        fig = plt.figure(figsize=(20, 4 * ((self.num_classes + 3) // 4)))
        gs = plt.GridSpec(((self.num_classes + 3) // 4), 4, figure=fig)
    
        # Plot distribution for each class
        for class_idx in range(self.num_classes):
            ax = fig.add_subplot(gs[class_idx // 4, class_idx % 4])
        
            # Create grid for visualization
            grid_probs = np.zeros((self.grid_size, self.grid_size))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    grid_probs[i, j] = self.label_distributions[(i, j)][class_idx]
        
            # Plot heatmap for this class
            im = ax.imshow(grid_probs, origin='lower', cmap='YlOrRd')
            ax.set_title(f'Class {class_idx} Distribution')
            plt.colorbar(im, ax=ax)
    
        plt.suptitle(f'Spatial Distribution of Class Probabilities (alpha={self.alpha})')
        plt.tight_layout()
        plt.show()

    def analyze_spatial_iidness(self):
        """
        Analyze the IIDness of data distribution across the spatial grid
        """
        # Calculate global distribution (average across all grid points)
        global_dist = np.zeros(self.num_classes)
        for dist in self.label_distributions.values():
            global_dist += dist
        global_dist /= len(self.label_distributions)
    
        # Calculate KL divergence for each grid point
        kl_divergences = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                local_dist = self.label_distributions[(i, j)]
                kl_div = sum(local_dist[k] * np.log(local_dist[k] / global_dist[k])
                        for k in range(self.num_classes)
                        if local_dist[k] > 0 and global_dist[k] > 0)
                kl_divergences[i, j] = kl_div
    
        # Visualize KL divergence
        plt.figure(figsize=(10, 8))
        im = plt.imshow(kl_divergences, origin='lower', cmap='viridis')
        plt.colorbar(im, label='KL Divergence')
        plt.title(f'Spatial Distribution of Non-IIDness (alpha={self.alpha})')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.show()
    
        # Calculate and return summary statistics
        stats = {
        'mean_kl': np.mean(kl_divergences),
        'max_kl': np.max(kl_divergences),
        'min_kl': np.min(kl_divergences),
        'std_kl': np.std(kl_divergences)
        }
    
        print("\nSpatial IIDness Analysis:")
        print(f"Mean KL Divergence: {stats['mean_kl']:.4f}")
        print(f"Max KL Divergence: {stats['max_kl']:.4f}")
        print(f"Min KL Divergence: {stats['min_kl']:.4f}")
        print(f"Std KL Divergence: {stats['std_kl']:.4f}")
    
        return stats

    def analyze_client_label_distribution(self):
        """
        Analyze and visualize the actual distribution of labels among clients
        """
        # Gather all client distributions
        client_distributions = []
        for client_idx in range(self.num_clients):
            dist = np.zeros(self.num_classes)
            total_samples = sum(self.client_label_counts[client_idx].values())
            if total_samples > 0:
                for label, count in self.client_label_counts[client_idx].items():
                    dist[label] = count / total_samples
            client_distributions.append(dist)
    
        client_distributions = np.array(client_distributions)
    
        # Calculate global distribution
        global_dist = np.mean(client_distributions, axis=0)
    
        # Calculate KL divergence for each client
        client_kl_divs = []
        for dist in client_distributions:
            kl_div = sum(dist[k] * np.log(dist[k] / global_dist[k])
                        for k in range(self.num_classes)
                        if dist[k] > 0 and global_dist[k] > 0)
            client_kl_divs.append(kl_div)
    
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
        # Plot 1: Distribution of labels across all clients
        im1 = ax1.imshow(client_distributions.T, aspect='auto', cmap='YlOrRd')
        ax1.set_xlabel('Client ID')
        ax1.set_ylabel('Class Label')
        ax1.set_title('Label Distribution Across Clients')
        plt.colorbar(im1, ax=ax1, label='Proportion')
    
        # Plot 2: Box plot of label distributions
        ax2.boxplot([client_distributions[:, i] for i in range(self.num_classes)])
        ax2.set_xlabel('Class Label')
        ax2.set_ylabel('Proportion')
        ax2.set_title('Distribution of Label Proportions')
    
        # Plot 3: Histogram of KL divergences
        ax3.hist(client_kl_divs, bins=30)
        ax3.set_xlabel('KL Divergence')
        ax3.set_ylabel('Number of Clients')
        ax3.set_title('Distribution of Client KL Divergences')
    
        plt.suptitle(f'Analysis of Client Label Distributions (alpha={self.alpha})')
        plt.tight_layout()
        plt.show()
    
        # Print summary statistics
        print("\nClient Label Distribution Analysis:")
        print(f"Mean KL Divergence: {np.mean(client_kl_divs):.4f}")
        print(f"Max KL Divergence: {np.max(client_kl_divs):.4f}")
        print(f"Min KL Divergence: {np.min(client_kl_divs):.4f}")
        print(f"Std KL Divergence: {np.std(client_kl_divs):.4f}")
    
        return {
        'client_distributions': client_distributions,
        'kl_divergences': client_kl_divs,
        'global_distribution': global_dist
        }

    def analyze_dirichlet_effect(self, num_samples=1000):
        """
        Analyze the theoretical effect of different alpha values on the Dirichlet distribution
        """
        # Generate sample distributions for different alpha values
        alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
        samples = {}
    
        for alpha in alpha_values:
            samples[alpha] = dirichlet.rvs([alpha] * self.num_classes, size=num_samples)
    
        # Visualize the distributions
        fig, axes = plt.subplots(2, len(alpha_values), figsize=(20, 8))
    
        for idx, alpha in enumerate(alpha_values):
            # Plot 1: Example distribution across classes
            axes[0, idx].bar(range(self.num_classes), samples[alpha][0])
            axes[0, idx].set_title(f'α={alpha}')
            axes[0, idx].set_ylim(0, 1)
            if idx == 0:
                axes[0, idx].set_ylabel('Probability')
        
            # Plot 2: Distribution of probabilities
            axes[1, idx].hist(samples[alpha][:, 0], bins=30, density=True)
            axes[1, idx].set_ylim(0, 5)
            if idx == 0:
                axes[1, idx].set_ylabel('Density')
    
        plt.suptitle('Effect of Alpha on Dirichlet Distribution')
        axes[0, 0].set_ylabel('Class Probabilities')
        axes[1, 0].set_ylabel('Probability Density')
        plt.tight_layout()
        plt.show()
    
        # Calculate concentration metrics
        concentration_metrics = {}
        for alpha in alpha_values:
            # Calculate entropy for each sample
            entropies = [-np.sum(s * np.log(s + 1e-10)) for s in samples[alpha]]
            concentration_metrics[alpha] = {
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies)
        }
    
        print("\nConcentration Analysis:")
        for alpha, metrics in concentration_metrics.items():
            print(f"\nAlpha = {alpha}:")
            print(f"Mean Entropy: {metrics['mean_entropy']:.4f}")
            print(f"Std Entropy: {metrics['std_entropy']:.4f}")
    
        return concentration_metrics



# Example usage
if __name__ == "__main__":
    hierfed = HierFedLearning(
        dataset_name="mnist",
        total_rounds=100,
        num_clients=100,
        sample_per_client=100,
        num_edge_servers=4,
        grid_size=10,
        alpha=100,
        coverage_radius=3.0
    )
    
    hierfed.calculate_noniid_metrics()
    #hierfed.visualize_label_distributions()
    
    # Visualize the topology
    #hierfed.visualize_topology(show_grid=True, show_distances=True)
    
    # Visualize edge server coverage
    #hierfed.visualize_edge_coverage()
    #hierfed.visualize_dirichlet_distribution()  # Shows spatial distribution of each class
    #hierfed.analyze_spatial_iidness()  # Analyzes IIDness across the grid
    #hierfed.analyze_client_label_distribution()  # Analyzes actual client data distribution
    #hierfed.analyze_dirichlet_effect()
    # Train the model and get history
    final_model, history = hierfed.train()
    
    # Plot training metrics
    hierfed.plot_training_metrics(history)

    