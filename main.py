from federated_learning import HierFedLearning

if __name__ == "__main__":
    hierfed = HierFedLearning(
        dataset_name="mnist",
        total_rounds=100,
        num_clients=100,
        sample_per_client=100,
        num_edge_servers=4,
        grid_size=10,
        alpha=100,
        coverage_radius=3.0,
        client_repetition=True
    )
    
    hierfed.calculate_noniid_metrics()
    hierfed.visualize_topology(show_grid=True, show_distances=True)
    hierfed.visualize_edge_coverage()
    final_model= hierfed.train()

