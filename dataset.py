import numpy as np
from keras.datasets import mnist, cifar10, cifar100
from keras.utils import to_categorical

def load_dataset(dataset_name):
    """
    Load and preprocess the training dataset.

    Args:
        dataset_name: Name of the dataset ("mnist", "cifar-10", or "cifar-100").

    Returns:
        x_train: Preprocessed training images.
        y_train: Corresponding training labels.
    """
    if dataset_name == "mnist":
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    elif dataset_name == "cifar-10":
        (x_train, y_train), _ = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        y_train = y_train.squeeze()

    elif dataset_name == "cifar-100":
        (x_train, y_train), _ = cifar100.load_data()
        x_train = x_train.astype('float32') / 255.0
        y_train = y_train.squeeze()

    else:
        raise ValueError("Dataset must be 'mnist', 'cifar-10', or 'cifar-100'.")

    return x_train, y_train


def load_test_data(dataset_name, num_classes):
    """
    Load and preprocess the test dataset.

    Args:
        dataset_name: Name of the dataset ("mnist", "cifar-10", or "cifar-100").
        num_classes: Number of classes for one-hot encoding.

    Returns:
        x_test: Preprocessed test images.
        y_test: One-hot encoded test labels.
    """
    if dataset_name == "mnist":
        _, (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    elif dataset_name == "cifar-10":
        _, (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0
        y_test = y_test.squeeze()

    elif dataset_name == "cifar-100":
        _, (x_test, y_test) = cifar100.load_data()
        x_test = x_test.astype('float32') / 255.0
        y_test = y_test.squeeze()

    else:
        raise ValueError("Dataset must be 'mnist', 'cifar-10', or 'cifar-100'.")

    y_test = to_categorical(y_test, num_classes)
    return x_test, y_test
