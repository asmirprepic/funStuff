import random
from faker import Faker

class DataGenerator:
    @staticmethod
    def generate_random_persons(num_persons,use_faker = True):
        """
        Generate a list of random persons.
        Each person gets a unique PersonID, a generated name, and a random age.
        If use_faker is True, use realistic fake names. 
        """
        persons = []
        faker = Faker() if use_faker else None:
        
        for i in range(1, num_persons + 1):
            person = {
                'PersonID': i,
                'Name': faker.name() if use_faker else f'Person_{i}',
                'Age': random.randint(18, 80)
            }
            persons.append(person)
        return persons

    @staticmethod
    def generate_random_friendships(num_persons, avg_friends=5):
        """
        Generate random friendships between persons.
        Ensures no duplicate friendships or self-friendships.
        """
        friendships = []
        added = set()  # Track added friendship pairs
        for person in range(1, num_persons + 1):
            num_friends = random.randint(1, avg_friends)
            potential_friends = [i for i in range(1, num_persons + 1) if i != person]
            friends = random.sample(potential_friends, min(num_friends, len(potential_friends)))
            for friend in friends:
                pair = frozenset({person, friend})
                if pair not in added:
                    added.add(pair)
                    friendship = {
                        'PersonID': person,
                        'FriendID': friend,
                        'FriendshipStrength': round(random.uniform(0.1, 1.0), 2)
                    }
                    friendships.append(friendship)
        return friendships

    @staticmethod
    def generate_network_growth(existing_graph,num_new_nodes = 1, max_new_edges = 2,use_faker = True):
        """    
        Generate new nodes and edges to simulate network growth. 
        Returns new persons and friendships to be added to an existing graph. 

        Parameters: 
        -------------
        existing_graph: NetworkX graph object to check existing nodes. 
        num_new_nodes: Number of new persons to add
        max_new_edges: Maximum number of friendships for each new person

        Returns:
        ------------
        tuple_of (new_persons, new_friendships)

        """

        max_person_id = max(existing_graph.nodes) if existing_graph.nodes else 0
        new_persons = []
        new_friendships = []

        for i in range(1, num_new_nodes +1):
            new_id = max_person_id
            new_person = {'PersonID': new_id, 'Name': faker.name() if use_faker is True}
        
