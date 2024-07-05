from environment import Environment
import matplotlib.pyplot as plt

if __name__ == "__main__":
    size = 10  # Size of the environment
    num_actions = 9  # Number of possible actions
    learning_rate = 0.1
    discount = 0.95
    episodes = 10000
    move_penalty = 1
    enemy_penalty = 300
    food_reward = 25
    epsilon = 0.3  # Initial exploration rate
    show_every = 1000  # How often to display the environment

    # Create environment instance
    env = Environment(size, num_actions, learning_rate, discount, episodes, move_penalty, enemy_penalty, food_reward)

    # Train
    rewards_enemy,rewards_player = env.train(epsilon, show_every)


    plt.plot(rewards_player,label = 'players rewards')
    plt.plot(rewards_enemy,label = 'enemys rewards')
    plt.legend(loc = 4)
    plt.show()