import numpy as np
from keras.datasets import mnist, cifar10, cifar100
import tensorflow as tf
from scipy.stats import dirichlet
from collections import defaultdict
from typing import Dict, List, Tuple
import random

class DataHandler:
    def __init__(self, dataset_name: str, num_clients: int, samples_per_client: int, alpha: float):
        self.dataset_name = dataset_name.lower()
        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        self.alpha = alpha
        
        # Initialize dataset
        self.load_dataset()
        self.load_test_data()
        
        # Set dataset specific parameters
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

    def generate_label_distributions(self, grid_size: int) -> Dict[Tuple[int, int], np.ndarray]:
        """Generate Dirichlet distribution for each grid point"""
        distributions = {}
        for i in range(grid_size):
            for j in range(grid_size):
                distributions[(i, j)] = dirichlet.rvs(
                    [self.alpha] * self.num_classes)[0]
        return distributions

    def distribute_data_to_clients(self, client_locations: List[Tuple[float, float]], 
                                 label_distributions: Dict[Tuple[int, int], np.ndarray],
                                 grid_size: int) -> Dict[int, Dict[str, np.ndarray]]:
        """Distribute data to clients based on their location and label distribution"""
        client_data = {}
        self.client_label_counts = defaultdict(lambda: defaultdict(int))
        
        # Get indices for each class
        class_indices = defaultdict(list)
        for idx, label in enumerate(self.y_train):
            class_indices[label].append(idx)
            
        for client_idx, location in enumerate(client_locations):
            grid_x, grid_y = int(location[0]), int(location[1])
            dist = label_distributions[(grid_x, grid_y)]
            
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
            
            client_data[client_idx] = {
                'x': self.x_train[client_indices],
                'y': self.y_train[client_indices]
            }
            
        return client_data
