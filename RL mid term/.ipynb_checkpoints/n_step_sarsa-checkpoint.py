
import numpy as np
import matplotlib.pyplot as plt
import gym

def epsilon_greedy_policy(Q, state, epsilon, nA):
    if np.random.rand() > epsilon:
        max_actions = np.flatnonzero(Q[state] == Q[state].max())
        return np.random.choice(max_actions)
    else:
        return np.random.choice(nA)

def n_step_sarsa(env, num_episodes, alpha, gamma, epsilon_start, epsilon_min, decay_factor, n):
    Q = np.zeros((env.observation_space.n, env.action_space.n))  
    episode_rewards = np.zeros(num_episodes)
    
    for episode in range(num_episodes):
        epsilon = max(epsilon_min, epsilon_start * (decay_factor ** episode))  
        state, _ = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon, env.action_space.n)

        T = np.inf  
        t = 0  
        tau = 0  
        rewards = []  
        states = [state]  
        actions = [action]  

        while tau < (T - 1):
            if t < T:
                next_state, reward, done, _, _ = env.step(action)
                rewards.append(reward)
                states.append(next_state)

                if done:
                    T = t + 1  
                else:
                    next_action = epsilon_greedy_policy(Q, next_state, epsilon, env.action_space.n)
                    actions.append(next_action)

            tau = t - n + 1  

            if tau >= 0:
                if n == 1:  
                    Q[states[tau], actions[tau]] += alpha * (rewards[tau] + gamma * Q[next_state].max() - Q[states[tau], actions[tau]])
                else:
                    G = np.sum([gamma**(i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T))])
                    if tau + n < T:
                        G += gamma**n * Q[states[tau + n], actions[tau + n]]
        
                    Q[states[tau], actions[tau]] += alpha * (G - Q[states[tau], actions[tau]])

            t += 1  
            state = next_state  
            action = next_action if 'next_action' in locals() else action  

        episode_rewards[episode] = np.sum(rewards)  

    return Q, episode_rewards

def plot_results(nstep_rewards, env_name):
    episodes = len(nstep_rewards[0])
    avg_nstep_rewards = np.mean(nstep_rewards, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_nstep_rewards, label=f'n-step SARSA in {env_name}', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.title(f'Sum of Rewards over Episodes for n-step SARSA in {env_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

  
    plt.figure(figsize=(10, 6))
    plt.plot(avg_nstep_rewards[300:], label=f'n-step SARSA in {env_name}', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.title(f'Sum of Rewards over Episodes 300 to 500 for n-step SARSA in {env_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_optimal_policy(Q, env_name):
    action_mapping = {0: 'S', 1: 'N', 2: 'E', 3: 'W', 4: 'P', 5: 'D'}  
    
    print("Optimal Policy (Taxi Position, Passenger Location, Destination):")
    
    
    for taxi_row in range(5):
        for taxi_col in range(5):
            for passenger_location in range(5):  
                for destination in range(4):  
                    
                    state_index = (taxi_row * 5 + taxi_col) * 5 * 4 + passenger_location * 4 + destination
                    best_action = np.argmax(Q[state_index])  
                    action_str = action_mapping[best_action] 
                    print(f"Taxi at ({taxi_row}, {taxi_col}), Passenger Pos: {passenger_location}, Destination: {destination}, Best Action: {action_str}")

def run_experiments_nstep(env_name, num_runs=5, num_episodes=500, n=30, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.001, decay_factor=0.999):
    env = gym.make(env_name)
    nstep_rewards = []
    final_q_nstep = None

    for _ in range(num_runs):
        Q_nstep, rewards = n_step_sarsa(env, num_episodes, alpha, gamma, epsilon_start, epsilon_min, decay_factor, n)
        nstep_rewards.append(rewards)
        final_q_nstep = Q_nstep

    plot_results(nstep_rewards, env_name)  

 
    env = gym.make(env_name, render_mode='human') 
    state, _ = env.reset()  

    for _ in range(num_episodes):
        action = np.argmax(final_q_nstep[state])  
        state, reward, done, truncated, _ = env.step(action)  
        env.render()  
        if done or truncated:
            break
    
    env.close()  
    print("Optimal Policy (n-step SARSA):")
    print_optimal_policy(final_q_nstep, env_name)

'''
def print_optimal_policy(Q, env_name):
    action_mapping = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    
    optimal_policy = np.empty((4, 12), dtype=str)
    
    for state in range(Q.shape[0]):
        row = state // 12
        col = state % 12
        best_action = np.argmax(Q[state])
        optimal_policy[row, col] = action_mapping[best_action]
    
    for row in optimal_policy:
        print(' '.join(row)) '''
