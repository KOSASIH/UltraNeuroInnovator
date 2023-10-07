import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

# Define the neural network model
def create_model(input_shape, num_actions):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    return model

# Define the deep Q-network (DQN) agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = create_model(state_size, action_size)
        self.target_model = create_model(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights())
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.memory = []

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = np.random.choice(self.memory, self.batch_size, replace=False)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_full = self.model.predict(state)
            target_full[0][action] = target
            states.append(state[0])
            targets.append(target_full[0])
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define the preprocessing function
def preprocess_state(state):
    return np.expand_dims(state, axis=0)

# Create the environment
env = gym.make('Pong-v0')
state_size = (80, 80, 4)
action_size = env.action_space.n

# Create the DQN agent
agent = DQNAgent(state_size, action_size)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = preprocess_state(state)
    done = False
    total_reward = 0
    while not done:
        # Render the environment (optional)
        env.render()

        # Agent takes action
        action = agent.act(state)

        # Agent performs action
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)

        # Agent remembers experience
        agent.remember(state, action, reward, next_state, done)

        # Agent replays experiences and learns
        agent.replay()

        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_model()

        # Update current state
        state = next_state

        # Update total reward
        total_reward += reward

        # Decrease exploration rate
        agent.decrease_epsilon()

    # Print episode results
    print('Episode: {}, Total Reward: {}, Epsilon: {:.4f}'.format(episode, total_reward, agent.epsilon))

# Close the environment
env.close()
