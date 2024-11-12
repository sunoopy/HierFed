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
        self.global_model.build_model()  # Build the model properly
        
        # Initialize client locations and data distribution
        self.setup_topology()
        self.load_test_data()
   
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
    
    hierfed.calculate_noniid_metrics()
    hierfed.visualize_label_distributions()
    
    # Visualize the topology
    hierfed.visualize_topology(show_grid=True, show_distances=True)
    
    # Visualize edge server coverage
    hierfed.visualize_edge_coverage()
    
    # Train the model and get history
    final_model, history = hierfed.train()
    
    # Plot training metrics
    hierfed.plot_training_metrics(history)