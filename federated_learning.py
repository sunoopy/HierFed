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
import csv 
import os
import random
import time
from concurrent import futures
from datetime import datetime
from models import SimpleCNN
from dataset import (load_dataset, load_test_data)
from utils import (generate_edge_server_locations, generate_client_locations, generate_label_distributions, assign_clients_to_edges,distribute_data_to_clients)
from metrics import (calculate_kl_divergence, analyze_edge_server_distribution,calculate_noniid_metrics,visualize_dirichlet_distribution, analyze_spatial_iidness,analyze_client_label_distribution,analyze_dirichlet_effect)
from visualization import (plot_training_metrics,analyze_edge_server_distribution,calculate_kl_divergence,visualize_label_distributions,visualize_dirichlet_distribution,)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tensorboard_log_dir = "tb_logs0"

class HierFedLearning:
    def __init__(
        self,
        dataset_name,
        total_rounds,
        num_clients,
        sample_per_client,
        num_edge_servers,
        grid_size,
        coverage_radius,
        alpha,
        client_repetition=True,
    ):
        # Initialize the dataset name and other parameters
        self.dataset_name = dataset_name.lower()
        self.total_rounds = total_rounds
        self.num_clients = num_clients
        self.sample_per_client = sample_per_client
        self.num_edge_servers = num_edge_servers
        self.grid_size = grid_size
        self.coverage_radius = coverage_radius
        self.client_repetition = client_repetition
        self.alpha = alpha
        
        # Initialize num_classes based on dataset_name
        if self.dataset_name == "mnist":
            self.num_classes = 10
            self.model_input_shape = (28, 28, 1)
        elif self.dataset_name == "cifar-10":
            self.num_classes = 10
            self.model_input_shape = (32, 32, 3)
        elif self.dataset_name == "cifar-100":
            self.num_classes = 100
            self.model_input_shape = (32, 32, 3)
        else:
            raise ValueError("Dataset must be 'mnist', 'cifar-10', or 'cifar-100'.")

        # Now num_classes is initialized, so you can use it safely
        self.client_label_counts = defaultdict(lambda: defaultdict(int))
        
        # Load dataset and initialize model
        self.x_train, self.y_train = load_dataset(self.dataset_name)
        self.x_test, self.y_test = load_test_data(self.dataset_name, self.num_classes)

        # Initialize global model
        self.global_model = SimpleCNN(num_classes=self.num_classes, model_input_shape=self.model_input_shape)
        self.global_model.build_model()

        # Setup network topology
        self.setup_topology()
    
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
        Initialize the network topology Assign clients to edge servers and distribute data
        """
        # Generate edge server locations
        self.edge_points = generate_edge_server_locations(self.grid_size, self.num_edge_servers)

        # Generate client locations
        self.client_locations = generate_client_locations(self.grid_size, self.num_clients)

        # Generate label distributions using Dirichlet distribution
        self.label_distributions = generate_label_distributions(self.grid_size, self.num_classes, self.alpha)
        
        # Assign clients to edge servers
        self.client_assignments = assign_clients_to_edges(
            client_locations=self.client_locations,
            edge_points=self.edge_points,
            coverage_radius=self.coverage_radius,
            client_repetition=self.client_repetition)
        
        # Distribute data to clients
        self.client_data = distribute_data_to_clients(
            client_locations=self.client_locations,
            label_distributions=self.label_distributions,
            y_train=self.y_train,
            x_train=self.x_train,
            num_classes=self.num_classes,
            sample_per_client=self.sample_per_client,
        )

        # Count labels for each client
        for client_idx, data in self.client_data.items():
            labels = data["y"]
            for label in labels:
                self.client_label_counts[client_idx][label] += 1


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

    def train(self):
        # CSV setup
        csv_file = 'training_metrics.csv'
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Round', 'Average Training Loss', 'Test Accuracy', 'Total Round Time (s)'])

        training_history = {
            'round': [],
            'losses': [],
            'accuracies': [],
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

            # Timing and metrics
            round_end_time = time.time()
            total_round_time = round_end_time - round_start_time
            round_step = round + 1  # Increment step safely
            assert round_step > 0, f"Invalid TensorBoard step: {round_step}"  # Ensure it's valid

            with tf.summary.create_file_writer(tensorboard_log_dir).as_default() as writer:
                tf.summary.scalar('Test Accuracy', test_accuracy, step=round_step)
                tf.summary.scalar('Average Training Loss', np.mean(round_losses), step=round_step)
                tf.summary.scalar('Total Round Time (s)', total_round_time, step=round_step)

            # Record metrics for this round
            training_history['round'].append(round + 1)
            training_history['losses'].append(np.mean(round_losses))
            training_history['accuracies'].append(test_accuracy)
            training_history['total_times'].append(total_round_time)

            # Write to CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([round + 1, np.mean(round_losses), test_accuracy, total_round_time])

            # Log metrics to TensorBoard
            """
            with summary_writer.as_default():
                tf.summary.scalar('Test Accuracy', test_accuracy, step=round + 1)
                tf.summary.scalar('Average Training Loss', np.mean(round_losses), step=round + 1)
                tf.summary.scalar('Total Round Time (s)', total_round_time, step=round + 1)
            """

            print(f"Round {round + 1} Summary:")
            print(f"Average Training Loss: {np.mean(round_losses):.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Total Round Time: {timedelta(seconds=total_round_time)}")

        return self.global_model
    
    def calculate_distribution_divergence(self):
        """Calculate pairwise KL divergence between edge servers."""
        edge_distributions = analyze_edge_server_distribution(self.client_assignments, self.client_label_counts)

        global_dist = defaultdict(float)
        for dist in edge_distributions.values():
            for label, prob in dist.items():
                global_dist[label] += prob

        for label in global_dist:
            global_dist[label] /= len(edge_distributions)

        divergences = {
            edge_idx: calculate_kl_divergence(dist, global_dist)
            for edge_idx, dist in edge_distributions.items()
        }

        return divergences, edge_distributions

    def visualize_label_distributions(self):
        """Visualize label distributions using the external function."""
        divergences, edge_distributions = self.calculate_distribution_divergence()
        visualize_label_distributions(edge_distributions, divergences, self.num_classes, len(self.edge_points))

    def visualize_dirichlet_distribution(self):
        """Visualize Dirichlet distribution using the external function."""
        visualize_dirichlet_distribution(self.label_distributions, self.grid_size, self.num_classes)

    def plot_training_metrics(self, history):
        """Plot training metrics using the external function."""
        plot_training_metrics(history, self.total_rounds)
    
    def calculate_noniid_metrics(self):
        """
        Calculate and print comprehensive non-IID metrics for edge server label distributions.
        """
        # Analyze the distribution of labels across edge servers
        edge_distributions = analyze_edge_server_distribution(self.client_assignments, self.client_label_counts)

        # Use the imported function from metrics.py to calculate non-IID metrics
        metrics = calculate_noniid_metrics(edge_distributions)

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
        Visualize Dirichlet distribution using the function in metrics.py.
        """
        visualize_dirichlet_distribution(self.label_distributions, self.grid_size, self.num_classes, self.alpha)

    def analyze_spatial_iidness(self):
        """
        Analyze and visualize spatial IIDness using the function in metrics.py.
        """
        return analyze_spatial_iidness(self.label_distributions, self.grid_size, self.num_classes, self.alpha)

    def analyze_client_label_distribution(self):
        """
        Analyze and visualize client label distributions using the function in metrics.py.
        """
        return analyze_client_label_distribution(self.client_label_counts, self.num_classes)

    def analyze_dirichlet_effect(self, num_samples=1000):
        """
        Analyze and visualize Dirichlet effect using the function in metrics.py.
        """
        return analyze_dirichlet_effect(self.num_classes, num_samples)
    
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