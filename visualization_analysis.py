# visualization_analysis.py

import numpy as np
import matplotlib.pyplot as plt

def analyze_client_data(client_data, dataset, clients_per_region):
    num_classes = 10 if dataset in ['mnist', 'cifar10'] else 100
    client_labels = []
    for x, y in client_data:
        unique_labels, counts = np.unique(y, return_counts=True)
        client_labels.append(dict(zip(unique_labels, counts)))
    
    print("Data distribution among clients:")
    client_idx = 0
    for region, region_clients in enumerate(clients_per_region):
        print(f"Edge Server {region + 1}:")
        for _ in range(region_clients):
            print(f"  Client {client_idx}: {client_labels[client_idx]}")
            client_idx += 1
    
    print("\nLabel distribution across all clients:")
    label_counts = {i: 0 for i in range(num_classes)}
    for labels in client_labels:
        for label, count in labels.items():
            label_counts[label] += count
    
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")

def visualize_class_probabilities(class_probs, grid_size, x, y):
    num_classes = class_probs.shape[0]
    probs = class_probs[:, x, y]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), probs)
    plt.title(f"Class Probabilities at Grid Location ({x}, {y})")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.xticks(range(num_classes))
    plt.ylim(0, 1)
    plt.show()
