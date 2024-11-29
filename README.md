# Hierarchical Federated Learning with Edge Servers

This repository implements a hierarchical federated learning system where clients are distributed across multiple edge servers in a grid-based topology. The implementation uses a CNN model and supports various datasets (MNIST, CIFAR-10, CIFAR-100) for training in a federated setting. 

## Overview

The system implements a two-level hierarchical federated learning approach:
1. First level: Clients train local models and send updates to their assigned edge servers
2. Second level: Edge servers aggregate client models and communicate with the global server

The implementation uses a grid-based topology where clients are distributed across a defined area and assigned to the nearest edge server.

## Table of Contents
- [Getting Started](#getting-started)
- [Features](#features)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Visualization](#visualization)

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

## Features

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


## Implementation Details

### Model Architecture
The implementation uses a simple CNN with the following structure:
- 3 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- Dense layers for classification

### Key Parameters

| Parameter            | Type   | Values          | Default      | Description                                                         |
|----------------------|--------|-----------------|--------------|---------------------------------------------------------------------|
| `-total_rounds`      | int    | 1~              | 10           | Number of federated learning rounds                                 |
| `-dataset_name`      | string | xxx.xxx.xxx.xxx | MNIST        | dataset selection ( MNIST, Cifar-10, Cifar100)                      |
| `-num_clients`       | string | 1~              | 100          | Total number of clients in the system                               |
| `-sample_per_client` | int    | 1~              | 100          | Number of samples per client                                        |
| `-num_edge_servers`  | int    | 1~              | 4            | Number of edge servers                                              |
| `-grid_size`         | int    | 1~              | 10           | Size of the simulation grid                                         |
| `-alpha`             | float  | 1~              | 1.0          | Dirichlet distribution parameter for non-IID                        |
| `-coverage_radius`   | float  | 1~              | 3.0          | edge server coverage area radius setting                            |

## Usage

to be modified

## Visualization

The implementation provides two main visualization functions:

1. `visualize_topology()`: Shows the distribution of clients and edge servers
   - Displays client locations
   - Shows edge server positions
   - Optionally shows client-to-edge server assignments
   - Includes distribution statistics

2. `visualize_edge_coverage()`: Shows the coverage areas of edge servers
   - Heat map of edge server coverage zones
   - Client and edge server positions
   - Coverage boundaries

