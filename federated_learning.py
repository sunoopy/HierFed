import numpy as np
import tensorflow as tf
from models import SimpleCNN
import csv
import time
from dataset import load_dataset, generate_label_distributions
from visualization import visualize_topology, visualize_edge_coverage
import os 
from datetime import timedelta

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tensorboard_log_dir = "tb_logs0"

class HierFedLearning:
    def __init__(self, dataset_name, total_rounds, num_clients, sample_per_client, 
                 num_edge_servers, grid_size, alpha, coverage_radius, client_repetition=True):
        self.dataset_name = dataset_name.lower()
        self.total_rounds = total_rounds
        self.num_clients = num_clients
        self.sample_per_client = sample_per_client
        self.num_edge_servers = num_edge_servers
        self.grid_size = grid_size
        self.alpha = alpha
        self.coverage_radius = coverage_radius
        self.client_repetition = client_repetition
        
        self.x_train, self.y_train, self.x_test, self.y_test = load_dataset(dataset_name)
        self.global_model = SimpleCNN(num_classes=10)
        self.global_model.build_model((32, 32, 3))
        self.label_distributions = generate_label_distributions(grid_size, 10, alpha)

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

    def visualize_topology(self, show_grid=True, show_distances=False):
        visualize_topology(self.x_train, self.label_distributions, self.grid_size)

    def visualize_edge_coverage(self):
        visualize_edge_coverage(self.label_distributions, self.grid_size)

    def train(self):
        """Perform hierarchical federated learning with timing and accuracy metrics"""
        # Initialize TensorBoard writers
        #log_dir = "./logs/fit/" + time.strftime("%Y-%m-%d_%H-%M-%S")
        #s.makedirs(log_dir, exist_ok=True)
        #summary_writer = tf.summary.create_file_writer(log_dir)
        
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


            with tf.summary.create_file_writer(tensorboard_log_dir).as_default():
                    tf.summary.scalar('Test Accuracy', test_accuracy, step=round + 1)
                    tf.summary.scalar('Average Training Loss', np.mean(round_losses), step=round + 1)
                    tf.summary.scalar('Total Round Time (s)', total_round_time, step=round + 1)

          

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
        