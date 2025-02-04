import numpy as np
import gym

class NStepTDControl:
    def __init__(self, env, n=1, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))  # Initialize Q-table

    def epsilon_greedy_policy(self, Q, state, epsilon, nA):
        if np.random.rand() > epsilon:  # Exploit
            max_actions = np.flatnonzero(Q[state] == Q[state].max())  # Find all actions with max Q-value
            return np.random.choice(max_actions)  # Randomly choose among them to break ties
        else:  # Explore
            return np.random.choice(nA)


    def learn(self, num_episodes=500):
        rewards_per_episode = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()  # Handle tuple from reset
            action = self.epsilon_greedy_policy(self.Q, state, self.epsilon, self.env.action_space.n)
            total_reward = 0
            T = float('inf')
            t = 0
            states, rewards, actions = [state], [], [action]
            
            while True:
                if t < T:
                    # Take a step and handle tuple return values
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated  # Combine termination conditions
                    rewards.append(reward)
                    states.append(next_state)
                    total_reward += reward
                    action = self.epsilon_greedy_policy(self.Q, next_state, self.epsilon, self.env.action_space.n)
                    actions.append(action)
                    
                    if done:
                        T = t + 1
    
                tau = t - self.n + 1
                if tau >= 0:
                    G = sum([self.gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + self.n, T) + 1)])
                    if tau + self.n < T:
                        G += self.gamma ** self.n * np.max(self.Q[states[tau + self.n]])
                    
                    self.Q[states[tau], actions[tau]] += self.alpha * (G - self.Q[states[tau], actions[tau]])
    
                if tau == T - 1:
                    break
    
                # Move to next time step
                state = next_state
                t += 1
                
            rewards_per_episode.append(total_reward)
    
        return self.Q, rewards_per_episode


# Example usage:
# env = gym.make('CliffWalking-v0')
# agent = NStepTDControl(env, n=3, alpha=0.1, gamma=0.99, epsilon=0.1)
# Q, rewards = agent.learn(num_episodes=500)
