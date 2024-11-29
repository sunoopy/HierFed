# Hierarchical Federated Learning with Edge Servers

This repository implements a hierarchical federated learning system where clients are distributed across multiple edge servers in a grid-based topology. The implementation uses a CNN model and supports various datasets (MNIST, CIFAR-10, CIFAR-100) for training in a federated setting. 

## Getting Started

```
conda create -n HFLD python=3.10.6
conda activate HFLD
git clone https://github.com/sunoopy/HierFed.git
pip install -r requirments.txt

```

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Visualization](#visualization)

## Overview

The system implements a two-level hierarchical federated learning approach:
1. First level: Clients train local models and send updates to their assigned edge servers
2. Second level: Edge servers aggregate client models and communicate with the global server

The implementation uses a grid-based topology where clients are distributed across a defined area and assigned to the nearest edge server.

## Features

- **Hierarchical Learning Structure**
  - Two-level federated aggregation (Client → Edge → Global)
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

## Requirements

```
tensorflow
numpy
matplotlib
seaborn
scipy
```

## Installation

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install tensorflow numpy matplotlib seaborn scipy
```

## Implementation Details

### Model Architecture
The implementation uses a simple CNN with the following structure:
- 3 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- Dense layers for classification

### Key Parameters
- `dataset_name`: Choice of dataset ('mnist', 'cifar-10', 'cifar-100')
- `total_rounds`: Number of federated learning rounds
- `num_clients`: Total number of clients in the system
- `samples_per_client`: Number of samples per client
- `num_edge_servers`: Number of edge servers
- `grid_size`: Size of the simulation grid
- `alpha`: Dirichlet distribution parameter for non-IID data distribution

## Usage

```python
from hierarchical_federated import HierFedLearning

# Initialize the system
hierfed = HierFedLearning(
    dataset_name="mnist",
    total_rounds=10,
    num_clients=100,
    samples_per_client=500,
    num_edge_servers=4,
    grid_size=10,
    alpha=0.5
)

# Visualize the topology
hierfed.visualize_topology(show_grid=True, show_distances=True)

# Visualize edge server coverage
hierfed.visualize_edge_coverage()

# Train the model
final_model, history = hierfed.train()
```

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

