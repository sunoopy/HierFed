import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import dirichlet

def plot_training_metrics(history, total_rounds):
    """Plot training metrics over rounds."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot loss
    ax1.plot(range(1, total_rounds + 1), history['losses'])
    ax1.set_title('Average Training Loss per Round')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(range(1, total_rounds + 1), history['accuracies'])
    ax2.set_title('Test Accuracy per Round')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    # Plot client and edge times
    ax3.plot(range(1, total_rounds + 1), history['client_times'], label='Client Training')
    ax3.plot(range(1, total_rounds + 1), history['edge_times'], label='Edge Aggregation')
    ax3.set_title('Average Times per Round')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend()
    ax3.grid(True)

    # Plot total round time
    ax4.plot(range(1, total_rounds + 1), history['total_times'])
    ax4.set_title('Total Round Time')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Time (seconds)')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


def analyze_edge_server_distribution(client_assignments, client_label_counts):
    """Analyze the distribution of labels across edge servers."""
    edge_label_distributions = defaultdict(lambda: defaultdict(int))

    for edge_idx, client_indices in client_assignments.items():
        for client_idx in client_indices:
            for label, count in client_label_counts[client_idx].items():
                edge_label_distributions[edge_idx][label] += count

    edge_distributions = {}
    for edge_idx, label_counts in edge_label_distributions.items():
        total_samples = sum(label_counts.values())
        edge_distributions[edge_idx] = {label: count / total_samples for label, count in label_counts.items()}

    return edge_distributions


def calculate_kl_divergence(p, q):
    """Calculate KL divergence between two distributions."""
    return sum(p[i] * np.log(p[i] / q[i]) for i in range(len(p)) if p[i] > 0 and q[i] > 0)


def visualize_label_distributions(edge_distributions, divergences, num_classes, num_edges):
    """Visualize the label distribution across edge servers."""
    fig, axes = plt.subplots(2, (num_edges + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()

    for edge_idx, dist in edge_distributions.items():
        labels = list(range(num_classes))
        values = [dist.get(label, 0) for label in labels]

        axes[edge_idx].bar(labels, values)
        axes[edge_idx].set_title(f'Edge Server {edge_idx}\nKL Div: {divergences[edge_idx]:.4f}')
        axes[edge_idx].set_xlabel('Class Label')
        axes[edge_idx].set_ylabel('Proportion')

    for idx in range(len(edge_distributions), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Label Distribution Across Edge Servers')
    plt.tight_layout()
    plt.show()


def visualize_dirichlet_distribution(label_distributions, grid_size, num_classes):
    """Visualize the Dirichlet distribution across the grid."""
    fig = plt.figure(figsize=(20, 4 * ((num_classes + 3) // 4)))
    gs = plt.GridSpec(((num_classes + 3) // 4), 4, figure=fig)

    for class_idx in range(num_classes):
        ax = fig.add_subplot(gs[class_idx // 4, class_idx % 4])

        grid_probs = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                grid_probs[i, j] = label_distributions[(i, j)][class_idx]

        im = ax.imshow(grid_probs, origin='lower', cmap='YlOrRd')
        ax.set_title(f'Class {class_idx} Distribution')
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'Spatial Distribution of Class Probabilities')
    plt.tight_layout()
    plt.show()