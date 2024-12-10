import numpy as np

def calculate_kl_divergence(p, q):
    kl_div = 0
    for i in range(len(p)):
        if p[i] > 0 and q[i] > 0:
            kl_div += p[i] * np.log(p[i] / q[i])
    return kl_div

def calculate_average_divergence(distributions):
    avg_divergence = 0
    num_distributions = len(distributions)
    for i in range(num_distributions):
        for j in range(i + 1, num_distributions):
            avg_divergence += calculate_kl_divergence(distributions[i], distributions[j])
    return avg_divergence / num_distributions