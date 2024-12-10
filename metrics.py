import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from collections import defaultdict

def calculate_kl_divergence(p, q):
    """
    Calculate the KL divergence between two distributions.
    """
    return sum(p[i] * np.log(p[i] / q[i]) for i in range(len(p)) if p[i] > 0 and q[i] > 0)

def analyze_edge_server_distribution(client_assignments):
    """
    Analyze the distribution of labels across edge servers.
    """
    label_distributions = defaultdict(lambda: defaultdict(int))
    for client_idx, client_data in client_assignments.items():
        y = client_data["y"]
        for label in y:
            label_distributions[client_idx][label] += 1

    edge_distributions = {}
    for edge_idx, label_count in label_distributions.items():
        total = sum(label_count.values())
        edge_distributions[edge_idx] = {label: count / total for label, count in label_count.items()}

    return edge_distributions

def calculate_noniid_metrics(edge_distributions):
    """
    Calculate non-IID metrics such as KL divergence and label diversity.

    Args:
        edge_distributions: Dictionary of label distributions across edge servers.

    Returns:
        Dictionary with non-IID metrics like average KL divergence, label diversity, etc.
    """
    global_dist = defaultdict(float)
    
    # Calculate global distribution as the average of all edge distributions
    for dist in edge_distributions.values():
        for label, prob in dist.items():
            global_dist[label] += prob
    global_dist = {label: prob / len(edge_distributions) for label, prob in global_dist.items()}

    # Calculate KL divergences between each edge server and the global distribution
    kl_divergences = {
        edge_idx: calculate_kl_divergence(dist, global_dist)
        for edge_idx, dist in edge_distributions.items()
    }

    # Calculate label diversity (number of labels with > 1% presence per edge server)
    label_diversity = {
        edge_idx: len([label for label, prob in dist.items() if prob > 0.01])
        for edge_idx, dist in edge_distributions.items()
    }

    # Calculate metrics
    metrics = {
        "avg_kl_divergence": np.mean(list(kl_divergences.values())),
        "max_kl_divergence": max(kl_divergences.values()),
        "min_kl_divergence": min(kl_divergences.values()),
        "avg_label_diversity": np.mean(list(label_diversity.values())),
        "max_label_diversity": max(label_diversity.values()),
        "min_label_diversity": min(label_diversity.values()),
        "std_kl_divergence": np.std(list(kl_divergences.values()))  # Added the standard deviation of KL divergences
    }

    return metrics

def visualize_dirichlet_distribution(label_distributions, grid_size, num_classes, alpha):
    """
    Visualize how the Dirichlet distribution affects label distribution across the grid.
    """
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

    plt.suptitle(f'Spatial Distribution of Class Probabilities (alpha={alpha})')
    plt.tight_layout()
    plt.show()


def analyze_spatial_iidness(label_distributions, grid_size, num_classes, alpha):
    """
    Analyze the IIDness of data distribution across the spatial grid.
    """
    global_dist = np.zeros(num_classes)
    for dist in label_distributions.values():
        global_dist += dist
    global_dist /= len(label_distributions)

    kl_divergences = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            local_dist = label_distributions[(i, j)]
            kl_div = sum(local_dist[k] * np.log(local_dist[k] / global_dist[k])
                         for k in range(num_classes) if local_dist[k] > 0 and global_dist[k] > 0)
            kl_divergences[i, j] = kl_div

    plt.figure(figsize=(10, 8))
    im = plt.imshow(kl_divergences, origin='lower', cmap='viridis')
    plt.colorbar(im, label='KL Divergence')
    plt.title(f'Spatial Distribution of Non-IIDness (alpha={alpha})')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.show()

    stats = {
        'mean_kl': np.mean(kl_divergences),
        'max_kl': np.max(kl_divergences),
        'min_kl': np.min(kl_divergences),
        'std_kl': np.std(kl_divergences)
    }

    print("\nSpatial IIDness Analysis:")
    print(f"Mean KL Divergence: {stats['mean_kl']:.4f}")
    print(f"Max KL Divergence: {stats['max_kl']:.4f}")
    print(f"Min KL Divergence: {stats['min_kl']:.4f}")
    print(f"Std KL Divergence: {stats['std_kl']:.4f}")

    return stats


def analyze_client_label_distribution(client_label_counts, num_classes):
    """
    Analyze and visualize the actual distribution of labels among clients.
    """
    client_distributions = []
    for client_idx in range(len(client_label_counts)):
        dist = np.zeros(num_classes)
        total_samples = sum(client_label_counts[client_idx].values())
        if total_samples > 0:
            for label, count in client_label_counts[client_idx].items():
                dist[label] = count / total_samples
        client_distributions.append(dist)

    client_distributions = np.array(client_distributions)

    global_dist = np.mean(client_distributions, axis=0)

    client_kl_divs = []
    for dist in client_distributions:
        kl_div = sum(dist[k] * np.log(dist[k] / global_dist[k])
                     for k in range(num_classes) if dist[k] > 0 and global_dist[k] > 0)
        client_kl_divs.append(kl_div)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    im1 = ax1.imshow(client_distributions.T, aspect='auto', cmap='YlOrRd')
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Class Label')
    ax1.set_title('Label Distribution Across Clients')
    plt.colorbar(im1, ax=ax1, label='Proportion')

    ax2.boxplot([client_distributions[:, i] for i in range(num_classes)])
    ax2.set_xlabel('Class Label')
    ax2.set_ylabel('Proportion')
    ax2.set_title('Distribution of Label Proportions')

    ax3.hist(client_kl_divs, bins=30)
    ax3.set_xlabel('KL Divergence')
    ax3.set_ylabel('Number of Clients')
    ax3.set_title('Distribution of Client KL Divergences')

    plt.suptitle(f'Analysis of Client Label Distributions')
    plt.tight_layout()
    plt.show()

    print("\nClient Label Distribution Analysis:")
    print(f"Mean KL Divergence: {np.mean(client_kl_divs):.4f}")
    print(f"Max KL Divergence: {np.max(client_kl_divs):.4f}")
    print(f"Min KL Divergence: {np.min(client_kl_divs):.4f}")
    print(f"Std KL Divergence: {np.std(client_kl_divs):.4f}")

    return {
        'client_distributions': client_distributions,
        'kl_divergences': client_kl_divs,
        'global_distribution': global_dist
    }


def analyze_dirichlet_effect(num_classes, num_samples=1000):
    """
    Analyze the theoretical effect of different alpha values on the Dirichlet distribution.
    """
    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
    samples = {}

    for alpha in alpha_values:
        samples[alpha] = dirichlet.rvs([alpha] * num_classes, size=num_samples)

    fig, axes = plt.subplots(2, len(alpha_values), figsize=(20, 8))

    for idx, alpha in enumerate(alpha_values):
        axes[0, idx].bar(range(num_classes), samples[alpha][0])
        axes[0, idx].set_title(f'α={alpha}')
        axes[0, idx].set_ylim(0, 1)
        if idx == 0:
            axes[0, idx].set_ylabel('Probability')

        axes[1, idx].hist(samples[alpha][:, 0], bins=30, density=True)
        axes[1, idx].set_ylim(0, 5)
        if idx == 0:
            axes[1, idx].set_ylabel('Density')

    plt.suptitle('Effect of Alpha on Dirichlet Distribution')
    axes[0, 0].set_ylabel('Class Probabilities')
    axes[1, 0].set_ylabel('Probability Density')
    plt.tight_layout()
    plt.show()

    concentration_metrics = {}
    for alpha in alpha_values:
        entropies = [-np.sum(s * np.log(s + 1e-10)) for s in samples[alpha]]
        concentration_metrics[alpha] = {
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies)
        }

    print("\nConcentration Analysis:")
    for alpha, metrics in concentration_metrics.items():
        print(f"\nAlpha = {alpha}:")
        print(f"Mean Entropy: {metrics['mean_entropy']:.4f}")
        print(f"Std Entropy: {metrics['std_entropy']:.4f}")

    return concentration_metrics
