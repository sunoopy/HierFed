# Hierarchical Federated Learning with Edge Servers

This repository implements a Hierarchical Federated Learning(HFL) system where clients are distributed across multiple edge servers in a grid-based topology. 
The implementation uses a CNN model and supports various datasets (MNIST, CIFAR-10, CIFAR-100) for training in a federated setting. 

The system implements a two-level hierarchical federated learning approach:
1. First level: Clients train local models and send updates to their assigned edge servers
2. Second level: Edge servers aggregate client models and communicate with the global server

The implementation uses a grid-based topology where clients are distributed across a defined area and assigned to the nearest edge server.

## Table of Contents
- [Getting Started](#getting-started)
- [Implementation Details](#implementation-details)
- [Usage](#usage)

## Getting Started

1. create new env
```
conda create -n HFLD python=3.10.6
conda activate HFLD
```

2. clone the repository
```
git clone https://github.com/sunoopy/HierFed.git
```

3. install required version libraries 
```
pip install -r requirments.txt
```


## Implementation Details

### Features

- **Hierarchical Learning Structure**
  - Two-level federated aggregation (Client → Edge Server  → Global)
  - Proximity-based client-to-edge server assignment
  - FedAvg aggregation at both edge and global levels

- **Flexible Dataset Support**
  - MNIST (28x28x1)
  - CIFAR-10 (32x32x3)
  - CIFAR-100 (32x32x3)

- **Non-IID Data Distribution**
  - Dirichlet distribution-based data allocation
  - Controllable data heterogeneity via alpha parameter

- **Visualization Tools**
  - Client distribution visualization
  - Edge server coverage visualization
  - Training progress monitoring


### Model Architecture
The implementation uses a simple CNN with the following structure:
- 3 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- Dense layers for classification

### Key Parameters

| Parameter              | Type   | Values          | Default      | Description                                                         |
|------------------------|--------|-----------------|--------------|---------------------------------------------------------------------|
| `-total_rounds`        | int    | 1~              | 10           | Number of federated learning rounds                                 |
| `-dataset_name`        | string | xxx.xxx.xxx.xxx | MNIST        | dataset selection ( mnist, cifar-10, cifar-100)                      |
| `-num_clients`         | string | 1~              | 100          | Total number of clients in the system                               |
| `-sample_per_client`   | int    | 1~              | 100          | Number of samples per client                                        |
| `-num_edge_servers`    | int    | 1~              | 4            | Number of edge servers                                              |
| `-grid_size`           | int    | 1~              | 10           | Size of the simulation grid                                         |
| `-alpha`               | float  | 1~              | 1.0          | Dirichlet distribution parameter for non-IID                        |
| `-coverage_radius`     | float  | 1~              | 3.0          | edge server coverage area radius setting                            |
| `-client_repetition`   | boolean| True/False      | True         | Joint area edge server client repetition                            |

## Usage

to be modified