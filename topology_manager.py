import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
import random

class TopologyManager:
    def __init__(self, num_clients: int, num_edge_servers: int, grid_size: int):
        self.num_clients = num_clients
        self.num_edge_servers = num_edge_servers
        self.grid_size = grid_size
        
        self.edge_points = self.generate_edge_server_locations()
        self.client_locations = self.generate_client_locations()
        self.client_assignments = self.assign_clients_to_edges()

    def generate_edge_server_locations(self) -> List[Tuple[float, float]]:
        """Generate evenly distributed edge server locations"""
        edge_points = []
        rows = int(np.sqrt(self.num_edge_servers))
        cols = self.num_edge_servers // rows
        
        for i in range(rows):
            for j in range(cols):
                x = (i + 0.5) * (self.grid_size / rows)
                y = (j + 0.5) * (self.grid_size / cols)
                edge_points.append((x, y))
                
        return edge_points

    def generate_client_locations(self) -> List[Tuple[float, float]]:
        """Generate random client locations on the grid"""
        return [(random.uniform(0, self.grid_size), 
                random.uniform(0, self.grid_size)) 
                for _ in range(self.num_clients)]

    def assign_clients_to_edges(self) -> Dict[int, List[int]]:
        """Assign clients to nearest edge server based on Euclidean distance"""
        assignments = defaultdict(list)
        
        for client_idx, client_loc in enumerate(self.client_locations):
            distances = [np.sqrt((client_loc[0] - edge[0])**2 + 
                               (client_loc[1] - edge[1])**2) 
                       for edge in self.edge_points]
            nearest_edge = np.argmin(distances)
            assignments[nearest_edge].append(client_idx)
            
        return assignments

    def get_topology_info(self):
        """Return topology information"""
        return {
            'edge_points': self.edge_points,
            'client_locations': self.client_locations,
            'client_assignments': self.client_assignments
        }