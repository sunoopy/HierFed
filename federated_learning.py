import numpy as np
from collections import defaultdict
from keras.utils import to_categorical
from models import SimpleCNN
from dataset import load_dataset
from utils import (
    generate_edge_server_locations, generate_client_locations, generate_label_distributions
)
from metrics import calculate_kl_divergence, analyze_edge_server_distribution

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
        # Load and preprocess dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_dataset(dataset_name)

        self.total_rounds = total_rounds
        self.num_clients = num_clients
        self.sample_per_client = sample_per_client
        self.num_edge_servers = num_edge_servers
        self.grid_size = grid_size
        self.alpha = alpha
        self.coverage_radius = coverage_radius
        self.client_repetition = client_repetition

        self.num_classes = 10 if dataset_name == "mnist" else (10 if dataset_name == "cifar-10" else 100)
        self.model_input_shape = (28, 28, 1) if dataset_name == "mnist" else (32, 32, 3)

        # Initialize global model
        self.global_model = SimpleCNN(num_classes=self.num_classes, model_input_shape=self.model_input_shape)
        self.global_model.build_model()

        # Setup network topology
        self.edge_points = generate_edge_server_locations(grid_size, num_edge_servers)
        self.client_locations = generate_client_locations(grid_size, num_clients)
        self.label_distributions = generate_label_distributions(grid_size, self.num_classes, alpha)

        # Assign clients to edges and distribute data
        self.setup_topology()

    def setup_topology(self):
        """
        Setup network topology: assign clients to edge servers and distribute data.
        """
        self.client_assignments = defaultdict(list)
        self.client_label_counts = defaultdict(lambda: defaultdict(int))

        for client_idx, location in enumerate(self.client_locations):
            x, y = int(location[0]), int(location[1])
            label_dist = self.label_distributions[(x, y)]
            indices = np.random.choice(len(self.x_train), self.sample_per_client, p=label_dist)

            self.client_assignments[client_idx] = {
                "x": self.x_train[indices],
                "y": self.y_train[indices],
            }

    def train_client(self, client_idx, model, epochs=1):
        """
        Train a model on a client's local data.
        """
        data = self.client_assignments[client_idx]
        x, y = data["x"], to_categorical(data["y"], self.num_classes)

        model.fit(x, y, epochs=epochs, batch_size=32, verbose=0)
        loss, acc = model.evaluate(x, y, verbose=0)
        return model.get_weights(), loss, acc

    def aggregate_models(self, model_weights_list):
        """
        Perform federated averaging to aggregate model weights.
        """
        return [np.mean(weights, axis=0) for weights in zip(*model_weights_list)]

    def train(self):
        """
        Execute the federated training process.
        """
        for round_idx in range(self.total_rounds):
            print(f"Round {round_idx + 1}/{self.total_rounds}")

            edge_models = {}
            for edge_idx in range(self.num_edge_servers):
                client_weights = []
                for client_idx in self.client_assignments:
                    # Train client model
                    model = SimpleCNN(self.num_classes, self.model_input_shape)
                    model.build_model()
                    model.set_weights(self.global_model.get_weights())
                    weights, _, _ = self.train_client(client_idx, model)
                    client_weights.append(weights)

                # Aggregate client weights to create edge model
                edge_models[edge_idx] = self.aggregate_models(client_weights)

            # Aggregate edge models into global model
            global_weights = self.aggregate_models(list(edge_models.values()))
            self.global_model.set_weights(global_weights)

            # Evaluate global model
            loss, accuracy = self.global_model.evaluate(self.x_test, to_categorical(self.y_test, self.num_classes))
            print(f"Global Model Accuracy: {accuracy:.4f}")

    def visualize_label_distributions(self):
        """
        Visualize label distributions across edge servers.
        """
        edge_distributions = analyze_edge_server_distribution(self.client_assignments)
        print("Edge Server Label Distributions:", edge_distributions)
