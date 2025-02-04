#harshaldafade(02125321)

import numpy as np
import itertools
import matplotlib.pyplot as plt
import gymnasium as gym
import time


def get_fourier_basis(order, dimensions):

    c = list(itertools.product(range(order + 1), repeat=dimensions))
    c = np.array(c)
    return c

def bound_state(s):
   
    s_min = np.array([-4.8, -3.0, -0.418, -3.5])
    s_max = np.array([4.8, 3.0, 0.418, 3.5])
    s_clipped = np.clip(s, s_min, s_max)
    s_bounded = (s_clipped - s_min) / (s_max - s_min)
    return s_bounded

def compute_features(s, c):
    cs = np.dot(c, s)
    phi = np.cos(np.pi * cs)
    return phi

def softmax_policy(theta, phi_s):

    preferences = np.dot(theta, phi_s)
    max_pref = np.max(preferences)
    stable_prefs = preferences - max_pref
    exp_preferences = np.exp(stable_prefs)
    
    sum_exp = np.sum(exp_preferences)
    if sum_exp == 0.0 or np.isnan(sum_exp):
        sum_exp = 1e-10 
    
    action_probs = exp_preferences / sum_exp
    if np.isnan(action_probs).any():
        print("Warning: action_probs contain NaN. Resetting to uniform probabilities.")
        action_probs = np.ones_like(action_probs) / len(action_probs)
    
    return action_probs

class ActorCritic:
    def __init__(self, env, alpha_w=0.001, alpha_theta=0.0001, gamma=1.0, lambd=0.9, order=3):
        self.env = env
        self.alpha_w = alpha_w
        self.alpha_theta = alpha_theta
        self.gamma = gamma
        self.lambd = lambd
        self.order = order
        
        self.num_actions = env.action_space.n
        self.dimensions = env.observation_space.shape[0]
        self.c = get_fourier_basis(order, self.dimensions)
        self.num_features = self.c.shape[0]
        
        self.w = np.zeros(self.num_features)  
        self.theta = np.zeros((self.num_actions, self.num_features))  
    
    def train(self, num_episodes):
       
        total_rewards = []
        steps_per_episode = []
        
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            s = observation
            phi_s = compute_features(bound_state(s), self.c)
            action_probs = softmax_policy(self.theta, phi_s)
            A = np.random.choice(self.num_actions, p=action_probs)
            
            z_w = np.zeros(self.num_features)
            z_theta = np.zeros((self.num_actions, self.num_features))
            
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                s_prime, reward, terminated, truncated, info = self.env.step(A)
                done = terminated or truncated
                R = reward
                phi_s_prime = compute_features(bound_state(s_prime), self.c)
                
                V_s = np.dot(self.w, phi_s)
                V_s_prime = np.dot(self.w, phi_s_prime) if not done else 0.0
                delta = R + self.gamma * V_s_prime - V_s
                
                z_w = self.gamma * self.lambd * z_w + phi_s
                self.w += self.alpha_w * delta * z_w
                
                one_hot = np.zeros(self.num_actions)
                one_hot[A] = 1
                grad_ln_pi = (one_hot - action_probs)[:, None] * phi_s[None, :]
                z_theta = self.gamma * self.lambd * z_theta + grad_ln_pi
                self.theta += self.alpha_theta * delta * z_theta
                
                s = s_prime
                phi_s = phi_s_prime
                action_probs = softmax_policy(self.theta, phi_s)
                A = np.random.choice(self.num_actions, p=action_probs)
                
                total_reward += R
                steps += 1
            
            total_rewards.append(total_reward)
            steps_per_episode.append(steps)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Steps: {steps}")
        
        return total_rewards, steps_per_episode
    
    def get_policy_action(self, s):
        
        phi_s = compute_features(bound_state(s), self.c)
        action_probs = softmax_policy(self.theta, phi_s)
        A = np.argmax(action_probs)
        return A
    
    def evaluate_policy(self, render=False, sleep_time=0.02):
       
        observation, info = self.env.reset()
        s = observation
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            if render:
                self.env.render()
                time.sleep(sleep_time)
            
            phi_s = compute_features(bound_state(s), self.c)
            action_probs = softmax_policy(self.theta, phi_s)
            A = np.argmax(action_probs)
            
            s_prime, reward, terminated, truncated, info = self.env.step(A)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            s = s_prime
        
        self.env.close()
        return total_reward, steps



def plot_rewards(avg_rewards, std_rewards, num_episodes):
   
    episodes = np.arange(1, num_episodes + 1)
    plt.figure(figsize=(12, 5))
    plt.plot(episodes, avg_rewards, label='Average Reward')
    plt.fill_between(episodes, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, label='Std Dev')
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.title('Actor-Critic with Eligibility Traces on CartPole: Average Total Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_steps(avg_steps, std_steps, num_episodes):
    
    episodes = np.arange(1, num_episodes + 1)
    plt.figure(figsize=(12, 5))
    plt.semilogy(episodes, avg_steps, label='Average Steps')
    plt.fill_between(episodes, avg_steps - std_steps, avg_steps + std_steps, alpha=0.2, label='Std Dev')
    plt.xlabel('Episode')
    plt.ylabel('Average Steps per Episode (Log Scale)')
    plt.title('Actor-Critic with Eligibility Traces on CartPole: Average Steps per Episode')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

def animate_policy(env, agent, sleep_time=0.02):
   
    observation, info = env.reset()
    s = observation
    total_reward = 0
    steps = 0
    done = False
    
    while not done:
        env.render()
        phi_s = compute_features(bound_state(s), agent.c)
        action_probs = softmax_policy(agent.theta, phi_s)
        A = np.argmax(action_probs)
        
        s_prime, reward, terminated, truncated, info = env.step(A)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        s = s_prime
        time.sleep(sleep_time)
    
    env.close()
    print(f"Animation complete. Total Reward: {total_reward}, Steps: {steps}")



if __name__ == "__main__":
   
    env = gym.make('CartPole-v1', max_episode_steps=500) 
    alpha_w = 0.001        
    alpha_theta = 0.0001   
    gamma = 1.0            
    lambd = 0.9            
    order = 3              
    num_episodes = 500     
    num_runs = 50          
    
  
    all_total_rewards = np.zeros((num_runs, num_episodes))
    all_steps_per_episode = np.zeros((num_runs, num_episodes))
    
    for run in range(num_runs):
        print(f"Starting Run {run + 1}/{num_runs}")
        env = gym.make('CartPole-v1', max_episode_steps=500)
        agent = ActorCritic(env, alpha_w, alpha_theta, gamma, lambd, order)
        total_rewards, steps_per_episode = agent.train(num_episodes)
        all_total_rewards[run, :] = total_rewards
        all_steps_per_episode[run, :] = steps_per_episode
        env.close()
        print(f"Completed Run {run + 1}/{num_runs}\n")
    

    avg_total_rewards = np.mean(all_total_rewards, axis=0)
    std_total_rewards = np.std(all_total_rewards, axis=0)
    avg_steps_per_episode = np.mean(all_steps_per_episode, axis=0)
    std_steps_per_episode = np.std(all_steps_per_episode, axis=0)
    

    plot_rewards(avg_total_rewards, std_total_rewards, num_episodes)
    

    plot_steps(avg_steps_per_episode, std_steps_per_episode, num_episodes)
    
  
    max_reward = np.max(avg_total_rewards)
    min_steps = np.min(avg_steps_per_episode)
    print(f"Optimal Values over {num_runs} runs:")
    print(f"Maximum Average Reward: {max_reward}")
    print(f"Minimum Average Steps per Episode: {min_steps}")
    
    
    print("Animating the final policy...")
    env = gym.make('CartPole-v1', render_mode='human')
    agent = ActorCritic(env, alpha_w, alpha_theta, gamma, lambd, order)
    agent.train(num_episodes=1)  
    animate_policy(env, agent)
    
    env.close()

