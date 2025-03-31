import networkx as nx
import random
import matplotlib.pyplot as plt
from faker import Faker


class RelationalToGraph:
    def __init__(self):
        # These lists simulate relational tables
        self.persons = []         # List of person dictionaries: {'PersonID': int, 'Name': str, 'Age': int}
        self.friendships = []     # List of friendship dictionaries: {'PersonID': int, 'FriendID': int, 'FriendshipStrength': float}
        self.graph = nx.Graph()   # Using an undirected graph for a simple social network
        self.fake = Faker()

    def generate_fake_poeple(self,num_people):
        """
        Generate fake people using faker.
        Each person will have a unique id a name and a random age
        
        """

        people = []
        for i in range(1,num_people +1):
            person = {
                'PersonId': i,
                'Name': self.fake.name(),
                'Age': random.randomint(18,80)
            }
            people.append(person)
        self.persons = people
        return self.persons
    
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
    def add_person(self, person):
        """
        Dynamically add a person node.
        """
        self.persons.append(person)
        self.graph.add_node(
            person['PersonID'],
            name=person.get('Name'),
            age=person.get('Age')
        )
    def add_friendship(self, friendship):
        """
        Dynamically add a friendship edge.
        """
        self.friendships.append(friendship)
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

    def recommend_friends(self, person_id):
        """
        Recommend friends for a given person based on mutual friends.
        Returns a dictionary with potential friend and the count of mutual friends.
        """
        current_friends = set(self.get_person_friends(person_id))
        recommendations = {}
        for friend in current_friends:
            for potential in self.get_person_friends(friend):
                if potential != person_id and potential not in current_friends:
                    recommendations[potential] = recommendations.get(potential, 0) + 1
        # Return recommendations sorted by number of mutual friends
        return dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True))
    
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

    def simulate_network_growth(self, num_new_nodes=1, max_new_edges=2):
        """
        Simulate network growth by randomly adding new persons and friendships.
        :param num_new_nodes: Number of new persons to add.
        :param max_new_edges: Maximum number of friendships for each new person.
        """
        max_person_id = max(self.graph.nodes) if self.graph.nodes else 0
        for i in range(1, num_new_nodes + 1):
            new_id = max_person_id + i
            # Create a new person with random age between 20 and 50
            new_person = {'PersonID': new_id, 'Name': f'Person_{new_id}', 'Age': random.randint(20, 50)}
            self.add_person(new_person)
            # Randomly connect this new person with some existing persons
            potential_friends = list(self.graph.nodes)
            # Exclude the new person itself if it's already in the list
            potential_friends = [pid for pid in potential_friends if pid != new_id]
            num_edges = random.randint(1, max_new_edges)
            friends = random.sample(potential_friends, min(num_edges, len(potential_friends)))
            for friend in friends:
                # Create a friendship with random strength
                friendship = {'PersonID': new_id, 'FriendID': friend, 'FriendshipStrength': round(random.uniform(0.1, 1.0), 2)}
                self.add_friendship(friendship)

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

    def compute_graph_statistics(self):
        """
        Compute and print graph-wide statistics such as:
         - Number of nodes
         - Number of edges
         - Graph density
         - Average clustering coefficient
        """
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        density = nx.density(self.graph)
        avg_clustering = nx.average_clustering(self.graph)
        stats = {
            'Number of Nodes': num_nodes,
            'Number of Edges': num_edges,
            'Density': density,
            'Average Clustering Coefficient': avg_clustering
        }
        return stats

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
