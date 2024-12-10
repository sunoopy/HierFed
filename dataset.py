import numpy as np
from keras.datasets import mnist, cifar10, cifar100

def load_dataset(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    elif dataset_name == "cifar-10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
        y_train, y_test = y_train.squeeze(), y_test.squeeze()
    elif dataset_name == "cifar-100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
        y_train, y_test = y_train.squeeze(), y_test.squeeze()
    else:
        raise ValueError("Dataset must be 'mnist', 'cifar-10', or 'cifar-100'")
    return (x_train, y_train), (x_test, y_test)
