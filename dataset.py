from keras.datasets import mnist, cifar10, cifar100
import numpy as np
from scipy.stats import dirichlet

def load_dataset(dataset_name):
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    elif dataset_name == "cifar-10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
    elif dataset_name == "cifar-100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32') / 255.0
    else:
        raise ValueError("Unsupported dataset.")
    return x_train, y_train, x_test, y_test

def generate_label_distributions(grid_size, num_classes, alpha):
    distributions = {}
    for i in range(grid_size):
        for j in range(grid_size):
            distributions[(i, j)] = dirichlet.rvs([alpha] * num_classes)[0]
    return distributions
