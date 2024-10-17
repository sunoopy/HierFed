# Hierarchical Federated Learning with Edge Servers and Client Distribution

This repository implements a hierarchical federated learning system where clients are distributed across multiple edge servers. The clients are assigned based on their proximity to edge servers in a grid-like area, mimicking a cellular network or 5G network environment. The system also allows you to simulate different scenarios, such as enabling or disabling overlapping areas among the edge servers.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Parameters](#parameters)
  - [Running the Simulation](#running-the-simulation)
- [Visualization](#visualization)
- [Future Improvements](#future-improvements)

## Introduction

In this project, we aim to simulate a distributed federated learning environment where:
- A defined number of clients are distributed over a grid in a Multi-Media (MM) pixel space.
- The clients are assigned to edge servers (which serve as cell base stations) based on proximity.
- Each edge server handles different numbers of clients, as would occur in a real cellular network.

The number of edge servers (base stations) can be defined as a parameter, and each edge server covers an equal-sized area. The distribution can either allow or disallow overlapping client assignments across edge servers.

The learning framework supports datasets such as MNIST, CIFAR-10, and CIFAR-100, and uses TensorFlow models for client and server updates.

## Features

- **Client Distribution**: Clients are distributed in a grid using a Dirichlet distribution to simulate non-IID data across the clients.
- **Edge Server Assignment**: Clients are assigned to edge servers based on proximity, with an option to enable/disable overlapping coverage between edge servers.
- **Hierarchical Federated Learning**: Supports training models with multiple clients assigned to each edge server. Regional updates are sent from edge servers to the central global server, mimicking a hierarchical learning structure.
- **Configurable Parameters**: You can adjust the number of clients, edge servers, the grid size, and other parameters to simulate various real-world scenarios.
- **Visualization**: Visualizes the client distribution and edge server coverage areas.

## Installation

To run this repository, you need to have the following libraries installed:

```bash
pip install tensorflow numpy scipy matplotlib
 ```
## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/sunoopy/HierFed.git
   cd HierFed
    ```
   
2. Install the required dependencies:

   - Ensure you have Python installed. 
   - Install the necessary libraries by running the following commands in your terminal or command prompt:
     - `pip install tensorflow`
     - `pip install numpy`
     - `pip install scipy`
     - `pip install matplotlib`

3. Run the main script to start the hierarchical federated learning experiment. You can customize the parameters directly in the script.

   - To run the script with default parameters, execute the main Python script: `python main.py`
   - The script is configured with the following default parameters:
     - `dataset = 'mnist'`: The dataset to use (`mnist`, `cifar10`, or `cifar100`)
     - `grid_size = 100`: The size of the grid (100x100 grid for data distribution)
     - `num_clients = 1000`: The number of clients
     - `num_edge_servers = 9`: The number of edge servers (3x3 grid of edge servers)
     - `alpha = 0.1`: Dirichlet concentration parameter for controlling non-IIDness (lower values lead to more non-IID distribution)
     - `samples_per_client = 100`: Number of samples for each client
     - `rounds = 5`: Number of training rounds
     - `allow_overlap = False`: Whether clients can be assigned to multiple edge servers (set to `True` to allow overlap)

4. Customize the parameters directly in the script if needed. For example, you can modify the following variables in the script to change the behavior:

   - `dataset`: Choose the dataset (`mnist`, `cifar10`, `cifar100`)
   - `grid_size`: Set the grid size for client distribution
   - `num_clients`: Define the number of clients in the simulation
   - `num_edge_servers`: Specify the number of edge servers (base stations)
   - `alpha`: Adjust the Dirichlet distribution parameter to control the data distribution (higher values lead to more uniform distribution)
   - `samples_per_client`: Set the number of samples each client holds
   - `rounds`: Set the number of communication rounds for the experiment
   - `allow_overlap`: Enable or disable client assignment overlap among edge servers

5. Visualize the results after running the simulation:

   - The `visualize_client_distribution` function generates a plot of client locations and the areas covered by each edge server.
   - Run the visualization function to see the client and edge server distribution after the experiment completes: 
     - This will create a plot showing how the clients are assigned to the edge servers.

