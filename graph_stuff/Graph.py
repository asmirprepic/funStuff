class Graph:
    def __init__(self, directed=False):
        self.graph = nx.DiGraph() if directed else nx.Graph()
    
    def add_edge(self, u, v, weight=1):
        """Adds an edge between u and v with optional weight."""
        self.graph.add_edge(u, v, weight=weight)
    
    def remove_edge(self, u, v):
        """Removes the edge between u and v if it exists."""
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
    
    def bfs(self, start):
        """Performs Breadth-First Search (BFS) from the given start node."""
        return list(nx.bfs_tree(self.graph, start).nodes)
    
    def dfs(self, start):
        """Performs Depth-First Search (DFS) from the given start node."""
        return list(nx.dfs_tree(self.graph, start).nodes)
    
    def dijkstra(self, start):
        """Finds the shortest paths from start node using Dijkstra's algorithm."""
        return nx.single_source_dijkstra_path_length(self.graph, start)
    
    def topological_sort(self):
        """Performs topological sorting (only for Directed Acyclic Graphs)."""
        return list(nx.topological_sort(self.graph)) if nx.is_directed_acyclic_graph(self.graph) else []
    
    def __repr__(self):
        return '\n'.join(f'{node}: {list(self.graph.adj[node])}' for node in self.graph.nodes)
