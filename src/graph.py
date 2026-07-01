"""
Graph analysis module for contour tracking
"""
import numpy as np
import networkx as nx
from scipy import ndimage
from collections import defaultdict
import cv2
import gc  # Προσθήκη

class GraphAnalyzer:
    """Create and analyze graph from skeleton"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.pixel_to_node = {}
        self.node_to_pixel = {}
        self.skeleton = None
        self.endpoints = []
        self.junctions = []
        self.loops = []
        
    def build_graph(self, skeleton):
        """Build graph from skeleton image"""
        self.skeleton = skeleton > 0 if skeleton is not None else np.array([[]])
        self.pixel_to_node = {}
        self.node_to_pixel = {}
        self.graph = nx.Graph()
        self.loops = []
        
        if self.skeleton.size == 0 or not np.any(self.skeleton):
            print("Warning: Empty skeleton, cannot build graph")
            return self.graph
        
        pixels = np.argwhere(self.skeleton)
        print(f"Building graph with {len(pixels)} skeleton pixels")
        
        if len(pixels) == 0:
            return self.graph
        
        for i, (y, x) in enumerate(pixels):
            node_id = i
            self.pixel_to_node[(y, x)] = node_id
            self.node_to_pixel[node_id] = (y, x)
            self.graph.add_node(node_id, pos=(x, y))
        
        connections = 0
        for (y, x), node_id in self.pixel_to_node.items():
            neighbors = self._get_neighbors(y, x)
            for ny, nx_pos in neighbors:
                if (ny, nx_pos) in self.pixel_to_node:
                    neighbor_id = self.pixel_to_node[(ny, nx_pos)]
                    if not self.graph.has_edge(node_id, neighbor_id):
                        self.graph.add_edge(node_id, neighbor_id)
                        connections += 1
        
        print(f"Graph created: {len(self.graph.nodes())} nodes, {connections} edges")
        
        self._analyze_graph()
        
        # Εκκαθάριση μνήμης
        gc.collect()
        
        return self.graph
    
    def _get_neighbors(self, y, x):
        """Get 8-neighbors that are in the skeleton"""
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_pos = y + dy, x + dx
                if 0 <= ny < self.skeleton.shape[0] and 0 <= nx_pos < self.skeleton.shape[1]:
                    if self.skeleton[ny, nx_pos]:
                        neighbors.append((ny, nx_pos))
        return neighbors
    
    def _analyze_graph(self):
        """Analyze graph to find endpoints, junctions, and loops"""
        self.endpoints = []
        self.junctions = []
        self.loops = []
        
        if self.graph.number_of_nodes() == 0:
            return
        
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            if degree == 1:
                self.endpoints.append(node)
            elif degree > 2:
                self.junctions.append(node)
        
        print(f"Found {len(self.endpoints)} endpoints, {len(self.junctions)} junctions")
        
        try:
            if self.graph.number_of_nodes() > 0:
                cycles = nx.cycle_basis(self.graph)
                self.loops = cycles
                print(f"Found {len(self.loops)} loops")
            else:
                self.loops = []
        except Exception as e:
            print(f"Warning: Could not find cycles: {e}")
            self.loops = []
    
    def extract_contours(self):
        """Extract contour lines from graph"""
        contours = []
        used_nodes = set()
        
        if self.graph.number_of_nodes() == 0:
            return contours
        
        for start_node in self.endpoints + self.junctions:
            if start_node in used_nodes:
                continue
            
            contour = []
            current = start_node
            
            while current not in used_nodes and current not in self.junctions:
                if current in self.node_to_pixel:
                    contour.append(self.node_to_pixel[current])
                used_nodes.add(current)
                
                neighbors = list(self.graph.neighbors(current))
                next_node = None
                for neighbor in neighbors:
                    if neighbor not in used_nodes:
                        next_node = neighbor
                        break
                
                if next_node is None:
                    break
                current = next_node
            
            if len(contour) > 1:
                contours.append(contour)
        
        if self.loops:
            for loop in self.loops:
                if len(loop) > 3:
                    contour = []
                    for node in loop:
                        if node in self.node_to_pixel:
                            contour.append(self.node_to_pixel[node])
                    if len(contour) > 1:
                        contours.append(contour)
        
        print(f"Extracted {len(contours)} contours")
        return contours
    
    def get_graph_statistics(self):
        """Get statistics about the graph"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_endpoints': len(self.endpoints) if hasattr(self, 'endpoints') else 0,
            'num_junctions': len(self.junctions) if hasattr(self, 'junctions') else 0,
            'num_loops': len(self.loops) if hasattr(self, 'loops') else 0,
            'is_connected': nx.is_connected(self.graph) if self.graph.nodes() else False
        }
        
        if self.graph.nodes():
            try:
                degrees = [self.graph.degree(n) for n in self.graph.nodes()]
                stats['avg_degree'] = np.mean(degrees) if degrees else 0
            except:
                stats['avg_degree'] = 0
            try:
                stats['num_components'] = nx.number_connected_components(self.graph)
            except:
                stats['num_components'] = 0
        
        return stats
