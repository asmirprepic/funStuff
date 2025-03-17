import networkx as nx

class RelationalToGraph:
    def __init__(self):
        # These lists simulate relational tables
        self.persons = []         # List of person dictionaries: {'PersonID': int, 'Name': str, 'Age': int}
        self.friendships = []     # List of friendship dictionaries: {'PersonID': int, 'FriendID': int, 'FriendshipStrength': float}
        self.graph = nx.Graph()   # Using an undirected graph for a simple social network

    def load_persons(self, persons_data):
        """
        Load persons from a list of dictionaries.
        """
        self.persons = persons_data

    def load_friendships(self, friendships_data):
        """
        Load friendships from a list of dictionaries.
        """
        self.friendships = friendships_data
        
    def compute_degree_centrality(self):
        """
        Compute and return degree centrality for the graph.
        """
        return nx.degree_centrality(self.graph)

    def build_graph(self):
        """
        Build a graph from the relational data.
        """
        # Add person nodes with properties
        for person in self.persons:
            self.graph.add_node(
                person['PersonID'],
                name=person.get('Name'),
                age=person.get('Age')
            )

        # Add friendship edges with properties (if any)
        for friendship in self.friendships:
            self.graph.add_edge(
                friendship['PersonID'],
                friendship['FriendID'],
                friendship_strength=friendship.get('FriendshipStrength')
            )

    def get_person_friends(self, person_id):
        """
        Return a list of friend IDs for the given person.
        """
        return list(self.graph.neighbors(person_id))

    def get_isolated_nodes(self):
        """
        Return nodes that have no edges (i.e. isolated nodes).
        """
        return [node for node, degree in self.graph.degree() if degree == 0]

    def get_shortest_path(self, source, target):
        """
        Get the shortest path between two nodes.
        Provide source and target as the node keys (e.g., 'person_1').
        """
        try:
            path = nx.shortest_path(self.graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return None
    def detect_communities(self):
        """
        Detect communities using a greedy modularity approach.
        Returns a list of sets, each set being a community.
        """
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(self.graph))
        return communities

    def display_graph_info(self):
        """
        Print basic information about the graph.
        """
        print("Nodes in graph:")
        for node, data in self.graph.nodes(data=True):
            print(f"  PersonID: {node}, Data: {data}")
        print("\nEdges in graph:")
        for u, v, data in self.graph.edges(data=True):
            print(f"  {u} <--> {v}, Data: {data}")
    def detect_communities(self):
        """
        Detect communities using a greedy modularity approach.
        Returns a list of sets, each set being a community.
        """
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(self.graph))
        return communities

    def visualize_graph(self):
        """
        Visualize the graph using matplotlib.
        """
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph)
        node_labels = {node: data.get('name', node) for node, data in self.graph.nodes(data=True)}
        nx.draw(self.graph, pos, with_labels=True, labels=node_labels, node_color='lightblue', node_size=1500, edge_color='gray')
        edge_labels = {(u, v): data.get('friendship_strength', '') for u, v, data in self.graph.edges(data=True) if data.get('friendship_strength') is not None}
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title("Social Network Graph")
        plt.axis('off')
        plt.show()

# Example usage:
if __name__ == '__main__':
    # Sample relational data
    persons_data = [
        {'PersonID': 1, 'Name': 'Alice', 'Age': 30},
        {'PersonID': 2, 'Name': 'Bob', 'Age': 25},
        {'PersonID': 3, 'Name': 'Charlie', 'Age': 35},
        {'PersonID': 4, 'Name': 'Diana', 'Age': 28}  # This person may be isolated if not in friendships
    ]
    friendships_data = [
        {'PersonID': 1, 'FriendID': 2, 'FriendshipStrength': 0.8},
        {'PersonID': 2, 'FriendID': 3, 'FriendshipStrength': 0.5},
        {'PersonID': 1, 'FriendID': 3, 'FriendshipStrength': 0.9}
    ]

    # Create the converter
    converter = RelationalToGraph()
    converter.load_persons(persons_data)
    converter.load_friendships(friendships_data)
    converter.build_graph()

    # Display the graph information
    converter.display_graph_info()

    # Query: Find friends of Alice (PersonID = 1)
    alice_friends = converter.get_person_friends(1)
    print(f"\nFriends of Alice (PersonID 1): {alice_friends}")

    # Query: Find isolated nodes (e.g., persons with no relationships)
    isolated_nodes = converter.get_isolated_nodes()
    print(f"\nIsolated nodes: {isolated_nodes}")
