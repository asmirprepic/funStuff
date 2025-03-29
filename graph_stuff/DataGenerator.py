import random

class DataGenerator:
    @staticmethod
    def generate_random_persons(num_persons):
        """
        Generate a list of random persons.
        Each person gets a unique PersonID, a generated name, and a random age.
        """
        persons = []
        for i in range(1, num_persons + 1):
            person = {
                'PersonID': i,
                'Name': f'Person_{i}',
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
