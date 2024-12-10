import numpy as np
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
    """
    global_dist = defaultdict(float)
    for dist in edge_distributions.values():
        for label, prob in dist.items():
            global_dist[label] += prob
    global_dist = {label: prob / len(edge_distributions) for label, prob in global_dist.items()}

    kl_divergences = {
        edge_idx: calculate_kl_divergence(dist, global_dist)
        for edge_idx, dist in edge_distributions.items()
    }

    label_diversity = {
        edge_idx: len([label for label, prob in dist.items() if prob > 0.01])
        for edge_idx, dist in edge_distributions.items()
    }

    metrics = {
        "avg_kl_divergence": np.mean(list(kl_divergences.values())),
        "max_kl_divergence": max(kl_divergences.values()),
        "min_kl_divergence": min(kl_divergences.values()),
        "avg_label_diversity": np.mean(list(label_diversity.values())),
        "max_label_diversity": max(label_diversity.values()),
        "min_label_diversity": min(label_diversity.values()),
    }

    return metrics
