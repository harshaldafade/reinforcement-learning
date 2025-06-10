# Reinforcement Learning Journey

A comprehensive exploration of Reinforcement Learning concepts through practical implementations and theoretical foundations.

## Homework 1: Multi-armed Bandit Algorithms

### Overview
This homework focuses on implementing and analyzing various multi-armed bandit algorithms, exploring the fundamental trade-off between exploration and exploitation in reinforcement learning.

### Key Components

#### 1. Bandit Class Implementation
- Custom `Bandit` class with configurable parameters:
  - Number of arms (K)
  - Mean rewards for each arm
  - Standard deviations for reward distributions
- Methods:
  - `K()`: Returns number of arms
  - `pull(k)`: Returns reward from pulling arm k using normal distribution

#### 2. Policy Implementations

##### Random Policy
- Completely random arm selection
- Maintains estimates of action values (Q) and visit counts (N)
- Uses incremental averaging for value updates

##### ε-Greedy Policy
- Balances exploration and exploitation
- With probability ε: random exploration
- With probability 1-ε: greedy exploitation
- Implemented with different ε values (0.1, 0.01)

##### Upper Confidence Bound (UCB) Policy
- Uses optimistic estimates to encourage exploration
- Incorporates uncertainty in value estimates
- Implemented with different exploration parameters (c=1, c=2)

### Experimental Analysis

#### Testbed Implementation
- 500 randomly generated 7-armed bandits
- True arm values sampled from N(0,1)
- Reward distributions: N(Q*(a),1)
- 1000 steps per bandit problem

#### Performance Metrics
- Average reward over time
- Comparison across different policies:
  1. Random policy
  2. Greedy policy (ε=0)
  3. ε-greedy (ε=0.1)
  4. ε-greedy (ε=0.01)
  5. UCB (c=1)
  6. UCB (c=2)

### Visualizations and Animations

#### 1. Policy Comparison
```python
# Example visualization code
plt.figure(figsize=(14, 8))
plt.plot(avg_rewards_random, label='Random')
plt.plot(avg_rewards_greedy, label='Greedy')
plt.plot(avg_rewards_epsilon_01, label='ε-greedy (0.1)')
plt.plot(avg_rewards_epsilon_01, label='ε-greedy (0.01)')
plt.plot(avg_rewards_ucb_1, label='UCB (c=1)')
plt.plot(avg_rewards_ucb_2, label='UCB (c=2)')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Comparison of Bandit Algorithms')
plt.legend()
plt.grid(True)
```

#### 2. Learning Curves
- Shows convergence of different policies
- Demonstrates exploration vs exploitation trade-off
- Illustrates impact of different parameter settings

#### 3. Interactive Animations

##### Bandit Arm Selection Visualization
```python
def animate_bandit_selection(bandit, policy, steps=100):
    rewards = []
    selected_arms = []
    
    for t in range(steps):
        arm = policy.choose_arm(t)
        reward = bandit.pull(arm)
        policy.update_estimate(arm, reward)
        rewards.append(reward)
        selected_arms.append(arm)
        
    return rewards, selected_arms
```

##### Real-time Policy Performance
```python
def animate_policy_performance(policies, bandit, steps=1000):
    plt.figure(figsize=(12, 6))
    for policy_name, policy in policies.items():
        rewards = []
        for t in range(steps):
            arm = policy.choose_arm(t)
            reward = bandit.pull(arm)
            policy.update_estimate(arm, reward)
            rewards.append(reward)
            plt.plot(rewards, label=policy_name)
            plt.pause(0.01)
    plt.legend()
    plt.show()
```

### Key Findings
1. Random policy serves as baseline performance
2. ε-greedy shows better performance with appropriate ε values
3. UCB demonstrates strong performance without explicit exploration parameter
4. Parameter tuning significantly impacts algorithm performance
5. Trade-off between exploration and exploitation is crucial

### Technical Implementation Details
- Pure NumPy implementation for efficiency
- Vectorized operations for better performance
- Proper statistical analysis of results
- Comprehensive visualization of learning curves

### Interactive Demonstrations
1. **Bandit Arm Selection**
   - Visual representation of arm selection process
   - Real-time updates of value estimates
   - Color-coded arms based on their true values

2. **Policy Comparison**
   - Side-by-side comparison of different policies
   - Real-time reward accumulation
   - Dynamic visualization of exploration vs exploitation

3. **Parameter Impact**
   - Interactive sliders for ε and c parameters
   - Real-time updates of policy performance
   - Visual feedback on parameter effects

## Homework 2: Finite Markov Decision Processes (MDPs)

### Overview
This homework focuses on the mathematical foundations and implementation of Finite Markov Decision Processes, exploring key concepts in reinforcement learning such as state-space representation, transition dynamics, and value functions.

### Key Components

#### 1. Mathematical Foundations
- State-space and action-space representation
- Transition dynamics and reward functions
- Bellman equations and optimality principles
- Value functions and policy evaluation
- Policy iteration and value iteration algorithms

#### 2. Core Concepts Implementation

##### State-Space Representation
- Discrete state representation
- Action space definition
- Transition probability matrices
- Reward function implementation

##### Value Functions
- State-value function V(s)
- Action-value function Q(s,a)
- Bellman equations implementation
- Value iteration convergence analysis

##### Policy Evaluation
- Policy evaluation algorithms
- Value function approximation
- Convergence criteria
- Performance metrics

### Experimental Analysis

#### Testbed Implementation
- Custom MDP environment
- State transition dynamics
- Reward structure
- Terminal state handling

#### Performance Metrics
- Value function convergence
- Policy optimality
- Computational efficiency
- Solution quality assessment

### Visualizations and Analysis

#### 1. Value Function Visualization
```python
def plot_value_function(V, title="Value Function"):
    plt.figure(figsize=(10, 6))
    plt.plot(V)
    plt.xlabel('States')
    plt.ylabel('Value')
    plt.title(title)
    plt.grid(True)
    plt.show()
```

#### 2. Policy Visualization
```python
def plot_policy(policy, title="Optimal Policy"):
    plt.figure(figsize=(10, 6))
    plt.plot(policy)
    plt.xlabel('States')
    plt.ylabel('Actions')
    plt.title(title)
    plt.grid(True)
    plt.show()
```

#### 3. Convergence Analysis
- Value iteration convergence plots
- Policy improvement visualization
- Performance comparison charts

### Key Findings
1. Value iteration converges to optimal value function
2. Policy iteration finds optimal policy efficiently
3. Discount factor (γ) significantly impacts solution
4. State-space size affects computational complexity
5. Reward structure influences optimal policy

### Technical Implementation Details
- Pure NumPy implementation
- Vectorized operations for efficiency
- Proper convergence criteria
- Comprehensive error handling

### Interactive Demonstrations
1. **Value Iteration Process**
   - Step-by-step value updates
   - Convergence visualization
   - Policy extraction demonstration

2. **Policy Evaluation**
   - Real-time policy assessment
   - Value function updates
   - Performance metrics tracking

3. **Parameter Impact**
   - Discount factor effects
   - Reward scaling analysis
   - Convergence rate comparison

## Homework 3: Dynamic Programming

### Overview
This homework focuses on implementing and analyzing dynamic programming algorithms through a practical case study of a cleaning robot problem. The implementation covers both deterministic and stochastic environments, demonstrating the application of value iteration and policy optimization techniques.

### Key Components

#### 1. Deterministic Cleaning Robot Problem
- State space: S = {0,...,9} (robot position in corridor)
- Action space: A = {-1,1} (move left or right)
- Terminal states: 0 and 9
- Discount factor: γ = 0.9

##### Implementation Details
- Deterministic transition function
- Custom reward structure:
  - +5 reward for being at s=8 and taking action a=1
  - +1 reward for being at s=1 and taking action a=-1
  - 0 reward otherwise
- Value iteration algorithm implementation
- Policy extraction from value functions

#### 2. Stochastic Cleaning Robot Problem
- Same state and action spaces as deterministic version
- Stochastic transitions:
  - 0.8 probability of intended movement
  - 0.15 probability of staying in place
  - 0.05 probability of moving in opposite direction
- Modified reward structure for stochastic environment

##### Implementation Details
- Stochastic transition function
- Probability table generation
- Modified value iteration for stochastic environment
- Policy optimization under uncertainty

### Technical Implementation

#### 1. Value Iteration Algorithm
```python
def value_iteration_deterministic(transition_fcn, reward_fcn, V0, gamma=0.9, theta=1e-6):
    V = np.copy(V0)
    P = np.zeros(len(V), dtype=int)
    
    while True:
        delta = 0
        for s in range(len(V)):
            if s == 0 or s == 9:
                continue
            
            v = V[s]
            action_values = []
            for a in [-1, 1]:
                next_state = transition_fcn(s, a)
                action_values.append(reward_fcn(s, a) + gamma * V[next_state])
            
            V[s] = max(action_values)
            P[s] = -1 if action_values[0] > action_values[1] else 1
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
            
    return V, P
```

#### 2. Stochastic Value Iteration
```python
def value_iteration_stochastic(transition_fcn, reward_fcn, V0, P_dynamics, gamma=0.9, theta=1e-6):
    V = np.copy(V0)
    P = np.zeros(len(V), dtype=int)
    
    while True:
        delta = 0
        for s in range(len(V)):
            if s == 0 or s == 9:
                continue
            
            v = V[s]
            action_values = []
            for a in [-1, 1]:
                expected_value = 0
                for sp in range(len(V)):
                    expected_value += P_dynamics[s][0 if a == -1 else 1][sp] * (
                        reward_fcn(s, a, sp) + gamma * V[sp])
                action_values.append(expected_value)
            
            V[s] = max(action_values)
            P[s] = -1 if action_values[0] > action_values[1] else 1
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
            
    return V, P
```

### Visualizations and Analysis

#### 1. Value Function Visualization
```python
def plot_value_function(V, title="Value Function"):
    plt.figure(figsize=(10, 6))
    plt.plot(V)
    plt.xlabel('States')
    plt.ylabel('Value')
    plt.title(title)
    plt.grid(True)
    plt.show()
```

#### 2. Policy Visualization
```python
def plot_policy(policy, title="Optimal Policy"):
    plt.figure(figsize=(10, 6))
    plt.plot(policy)
    plt.xlabel('States')
    plt.ylabel('Actions')
    plt.title(title)
    plt.grid(True)
    plt.show()
```

### Key Findings
1. Value iteration converges to optimal value function in both deterministic and stochastic environments
2. Stochastic environment requires more iterations to converge
3. Optimal policy in deterministic environment is more straightforward
4. Stochastic transitions lead to more conservative policies
5. Terminal states significantly influence the value function shape

### Technical Implementation Details
- Pure NumPy implementation for efficiency
- Vectorized operations for better performance
- Proper convergence criteria implementation
- Comprehensive error handling
- Modular design for easy testing and modification

### Interactive Demonstrations
1. **Value Iteration Process**
   - Step-by-step value updates
   - Convergence visualization
   - Policy extraction demonstration

2. **Policy Evaluation**
   - Real-time policy assessment
   - Value function updates
   - Performance metrics tracking

3. **Environment Comparison**
   - Deterministic vs stochastic behavior
   - Transition probability effects
   - Reward structure impact

## Homework 4: Monte Carlo Methods

### Overview
This homework focuses on implementing and analyzing Monte Carlo methods in reinforcement learning, using the classic Blackjack environment as a case study. The implementation covers both on-policy and off-policy Monte Carlo methods, including prediction and control algorithms.

### Key Components

#### 1. Blackjack Environment Implementation
- State representation: (player_sum, dealer_card, usable_ace)
- Action space: {hit, stick}
- Reward structure: {+1, -1, 0}
- Terminal state handling
- State-value and action-value estimation

#### 2. Monte Carlo Prediction
- First-visit Monte Carlo prediction
- Every-visit Monte Carlo prediction
- State-value function estimation
- Return calculation and averaging
- Convergence analysis

#### 3. Monte Carlo Control
- On-policy Monte Carlo control
- Off-policy Monte Carlo control
- Weighted importance sampling
- Policy improvement
- Optimal policy extraction

### Technical Implementation

#### 1. First-visit Monte Carlo Prediction
```python
def first_visit_mc_prediction(env, policy, num_episodes):
    V = defaultdict(float)
    returns = defaultdict(list)
    
    for episode in range(num_episodes):
        episode_states = []
        episode_rewards = []
        state = env.reset()
        
        while True:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_states.append(state)
            episode_rewards.append(reward)
            
            if done:
                break
            state = next_state
            
        G = 0
        for t in range(len(episode_states)-1, -1, -1):
            G = gamma * G + episode_rewards[t]
            state = episode_states[t]
            if state not in episode_states[:t]:
                returns[state].append(G)
                V[state] = np.mean(returns[state])
                
    return V
```

#### 2. Monte Carlo Control with Exploring Starts
```python
def mc_control_es(env, num_episodes):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = defaultdict(list)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        while True:
            action = policy[state]
            next_state, reward, done, _ = env.step(action)
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            if done:
                break
            state = next_state
            
        G = 0
        for t in range(len(episode_states)-1, -1, -1):
            G = gamma * G + episode_rewards[t]
            state = episode_states[t]
            action = episode_actions[t]
            sa_pair = (state, action)
            
            if sa_pair not in zip(episode_states[:t], episode_actions[:t]):
                returns[sa_pair].append(G)
                Q[state][action] = np.mean(returns[sa_pair])
                policy[state] = np.argmax(Q[state])
                
    return Q, policy
```

### Visualizations and Analysis

#### 1. State-Value Function Visualization
```python
def plot_state_value_3d(V, title="State-Value Function"):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    player_sums = range(12, 22)
    dealer_cards = range(1, 11)
    X, Y = np.meshgrid(player_sums, dealer_cards)
    Z = np.zeros_like(X)
    
    for i, ps in enumerate(player_sums):
        for j, dc in enumerate(dealer_cards):
            Z[j, i] = V.get((ps, dc, True), 0)
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.colorbar(surf)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Card')
    ax.set_zlabel('Value')
    plt.title(title)
    plt.show()
```

#### 2. Policy Visualization
```python
def plot_policy_2d(policy, title="Optimal Policy"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for usable ace
    for ps in range(12, 22):
        for dc in range(1, 11):
            action = policy.get((ps, dc, True), 0)
            ax1.scatter(ps, dc, c='g' if action == 1 else 'r', marker='s')
    
    ax1.set_xlabel('Player Sum')
    ax1.set_ylabel('Dealer Card')
    ax1.set_title('With Usable Ace')
    
    # Plot for no usable ace
    for ps in range(12, 22):
        for dc in range(1, 11):
            action = policy.get((ps, dc, False), 0)
            ax2.scatter(ps, dc, c='g' if action == 1 else 'r', marker='s')
    
    ax2.set_xlabel('Player Sum')
    ax2.set_ylabel('Dealer Card')
    ax2.set_title('Without Usable Ace')
    
    plt.suptitle(title)
    plt.show()
```

### Key Findings
1. First-visit MC prediction converges to true state values
2. Monte Carlo control with exploring starts finds optimal policy
3. Off-policy methods with importance sampling provide unbiased estimates
4. State-value function shows higher values for favorable states
5. Optimal policy demonstrates risk-averse behavior in certain states

### Technical Implementation Details
- Efficient state representation using tuples
- Dictionary-based value function storage
- Vectorized operations for better performance
- Proper handling of terminal states
- Comprehensive error handling

### Interactive Demonstrations
1. **Monte Carlo Prediction Process**
   - Episode generation and analysis
   - Return calculation visualization
   - State-value function updates

2. **Policy Evaluation**
   - Real-time policy assessment
   - Value function updates
   - Performance metrics tracking

3. **Control Methods Comparison**
   - On-policy vs off-policy behavior
   - Importance sampling effects
   - Policy improvement visualization

## Homework 5: Temporal Difference Learning

### Overview
This homework focuses on implementing and analyzing Temporal Difference (TD) learning algorithms, using the Cliff Walking environment as a case study. The implementation covers both on-policy (SARSA) and off-policy (Q-learning) TD control methods, with a focus on performance comparison and policy optimization.

### Key Components

#### 1. Cliff Walking Environment
- 4x12 grid world environment
- Start state: [3, 0] (bottom-left)
- Goal state: [3, 11] (bottom-right)
- Cliff states: [3, 1..10] (bottom-center)
- Actions: {up, right, down, left}
- Rewards: -1 per step, -100 for falling off cliff

#### 2. TD Control Algorithms
- SARSA (on-policy TD control)
- Q-learning (off-policy TD control)
- ε-greedy policy implementation
- Decaying ε exploration strategy

#### 3. Performance Analysis
- Reward comparison
- Policy visualization
- Convergence analysis
- Exploration vs exploitation trade-off

### Technical Implementation

#### 1. ε-Greedy Policy
```python
def epsilon_greedy_policy(Q, state, epsilon, nA):
    if np.random.rand() > epsilon:
        max_actions = np.flatnonzero(Q[state] == Q[state].max())
        return np.random.choice(max_actions)
    else:
        return np.random.choice(nA)
```

#### 2. SARSA Implementation
```python
def sarsa(env, num_episodes=500, alpha=0.1, gamma=1, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon, env.action_space.n)
        total_rewards = 0

        done = False
        while not done:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, env.action_space.n)
            
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            
            state = next_state
            action = next_action
            total_rewards += reward
            
        rewards_per_episode.append(total_rewards)
    
    return Q, rewards_per_episode
```

#### 3. Q-Learning Implementation
```python
def qlearning(env, num_episodes=500, alpha=0.1, gamma=1, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_per_episode = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_rewards = 0

        done = False
        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon, env.action_space.n)
            next_state, reward, done, truncated, _ = env.step(action)
            
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            state = next_state
            total_rewards += reward
            
        rewards_per_episode.append(total_rewards)

    return Q, rewards_per_episode
```

### Visualizations and Analysis

#### 1. Policy Visualization
```python
def print_optimal_policy(Q):
    action_mapping = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    
    optimal_policy = np.empty((4, 12), dtype=str)
    
    for state in range(Q.shape[0]):
        row = state // 12
        col = state % 12
        best_action = np.argmax(Q[state])
        optimal_policy[row, col] = action_mapping[best_action]
    
    for row in optimal_policy:
        print(' '.join(row))
```

#### 2. Performance Comparison
```python
def plot_results(sarsa_rewards, qlearning_rewards):
    episodes = len(sarsa_rewards[0])
    plt.figure(figsize=(12, 6))
    
    plt.plot(np.mean(sarsa_rewards, axis=0), label='SARSA')
    plt.plot(np.mean(qlearning_rewards, axis=0), label='Q-Learning')
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Performance Comparison: SARSA vs Q-Learning')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### Key Findings
1. Q-learning generally converges faster than SARSA
2. Both algorithms find optimal policies for the Cliff Walking task
3. Decaying ε strategy improves final performance
4. SARSA is more conservative in cliff regions
5. Q-learning takes more direct paths to goal

### Technical Implementation Details
- Pure NumPy implementation
- Vectorized operations for efficiency
- Proper handling of terminal states
- Comprehensive error handling
- Modular design for easy testing

### Interactive Demonstrations
1. **Learning Process**
   - Real-time policy updates
   - Reward accumulation visualization
   - Exploration vs exploitation balance

2. **Policy Evaluation**
   - Optimal path visualization
   - Action selection analysis
   - Performance metrics tracking

3. **Algorithm Comparison**
   - SARSA vs Q-learning behavior
   - Exploration strategy effects
   - Convergence rate analysis

## Homework 6: Function Approximation

### Overview
This homework focuses on implementing and analyzing function approximation methods in reinforcement learning, using the Mountain Car environment as a case study. The implementation covers both linear and non-linear function approximation techniques, with a focus on solving continuous state-space problems.

### Key Components

#### 1. Mountain Car Environment
- Continuous state space: (position, velocity)
- Action space: {0: accelerate left, 1: do nothing, 2: accelerate right}
- Transition dynamics:
  - Position: x_{t+1} = bound[x_t + ẋ_{t+1}]
  - Velocity: ẋ_{t+1} = bound[ẋ_t + 0.001(A_t-1) - 0.0025cos(3x_t)]
- Reward: -1 per step until goal reached
- Goal: Reach position ≥ 0.5
- Initial state: Random position in [-0.6, -0.4) with zero velocity

#### 2. Function Approximation Methods
- Linear function approximation
- Fourier basis functions
- Tile coding
- State aggregation
- Feature engineering

#### 3. Learning Algorithms
- Semi-gradient TD(0)
- Semi-gradient SARSA
- Semi-gradient Q-learning
- Gradient descent optimization
- Parameter tuning

### Technical Implementation

#### 1. Fourier Basis Functions
```python
def fourier_basis(state, order):
    """
    Generate Fourier basis features for state representation
    state: (position, velocity)
    order: maximum order of Fourier basis
    """
    features = []
    for i in range(order + 1):
        for j in range(order + 1):
            if i + j <= order:
                features.append(np.cos(np.pi * (i * state[0] + j * state[1])))
    return np.array(features)
```

#### 2. Semi-gradient TD(0)
```python
def semi_gradient_td(env, feature_fn, alpha=0.1, gamma=1.0, num_episodes=1000):
    """
    Semi-gradient TD(0) with function approximation
    """
    n_features = len(feature_fn(env.reset()[0], order=3))
    weights = np.zeros(n_features)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        features = feature_fn(state, order=3)
        
        done = False
        while not done:
            action = epsilon_greedy_policy(weights, features)
            next_state, reward, done, truncated, _ = env.step(action)
            next_features = feature_fn(next_state, order=3)
            
            # TD update
            td_target = reward + gamma * np.max(np.dot(next_features, weights))
            td_error = td_target - np.dot(features, weights)
            weights += alpha * td_error * features
            
            state = next_state
            features = next_features
            
    return weights
```

#### 3. Semi-gradient SARSA
```python
def semi_gradient_sarsa(env, feature_fn, alpha=0.1, gamma=1.0, epsilon=0.1, num_episodes=1000):
    """
    Semi-gradient SARSA with function approximation
    """
    n_features = len(feature_fn(env.reset()[0], order=3))
    weights = np.zeros(n_features)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        features = feature_fn(state, order=3)
        action = epsilon_greedy_policy(weights, features, epsilon)
        
        done = False
        while not done:
            next_state, reward, done, truncated, _ = env.step(action)
            next_features = feature_fn(next_state, order=3)
            next_action = epsilon_greedy_policy(weights, next_features, epsilon)
            
            # SARSA update
            td_target = reward + gamma * np.dot(next_features, weights)
            td_error = td_target - np.dot(features, weights)
            weights += alpha * td_error * features
            
            state = next_state
            features = next_features
            action = next_action
            
    return weights
```

### Visualizations and Analysis

#### 1. Learning Curves
```python
def plot_learning_curves(episode_rewards, title="Learning Curves"):
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.grid(True)
    plt.show()
```

#### 2. Value Function Visualization
```python
def plot_value_function(weights, feature_fn, title="Value Function"):
    positions = np.linspace(-1.2, 0.6, 100)
    velocities = np.linspace(-0.07, 0.07, 100)
    X, Y = np.meshgrid(positions, velocities)
    Z = np.zeros_like(X)
    
    for i in range(len(positions)):
        for j in range(len(velocities)):
            state = np.array([positions[i], velocities[j]])
            features = feature_fn(state, order=3)
            Z[j, i] = np.max(np.dot(features, weights))
    
    plt.figure(figsize=(12, 8))
    plt.contourf(X, Y, Z)
    plt.colorbar(label='Value')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title(title)
    plt.show()
```

### Key Findings
1. Function approximation successfully handles continuous state spaces
2. Fourier basis functions provide good generalization
3. Higher-order basis functions capture more complex value functions
4. Learning rate and basis function order significantly impact performance
5. Semi-gradient methods effectively solve the Mountain Car problem

### Technical Implementation Details
- Efficient feature computation
- Vectorized operations for better performance
- Proper handling of continuous states
- Comprehensive error handling
- Modular design for easy testing

### Interactive Demonstrations
1. **Learning Process**
   - Real-time value function updates
   - Policy improvement visualization
   - Performance metrics tracking

2. **Function Approximation**
   - Basis function visualization
   - Feature space exploration
   - Parameter impact analysis

3. **Environment Interaction**
   - Mountain Car simulation
   - Policy execution
   - Performance comparison

## Mathematical Foundations

Throughout these assignments, I've worked with:
- Probability theory and statistics
- Linear algebra for state-space representation
- Dynamic programming principles
- Stochastic processes
- Optimization theory
- Convergence analysis

## Key Algorithms Implemented

1. **Bandit Algorithms**
   - ε-greedy
   - UCB
   - Thompson Sampling

2. **Dynamic Programming**
   - Value Iteration
   - Policy Iteration
   - Policy Evaluation

3. **Monte Carlo Methods**
   - First-visit MC
   - Every-visit MC
   - On-policy MC control
   - Off-policy MC control

# Homework 7: Eligibility Traces and TD(λ)

## Key Concepts

### Eligibility Traces
- Eligibility traces are a mechanism to track which states or state-action pairs are eligible for learning
- They help bridge the gap between TD(0) and Monte Carlo methods
- Each state or state-action pair has an eligibility value that decays over time
- The decay rate is controlled by the parameter λ (lambda)

### TD(λ)
- TD(λ) is a family of algorithms that combines the advantages of TD(0) and Monte Carlo methods
- λ = 0: Pure TD(0) learning
- λ = 1: Monte Carlo method
- 0 < λ < 1: Combines both approaches
- The λ parameter controls how much weight is given to future rewards

## Algorithm Components

1. **Eligibility Trace Update**
   - For state-value prediction:
     ```
     e(s) = γλe(s) + 1 if s is current state
     e(s) = γλe(s) otherwise
     ```
   - For action-value prediction:
     ```
     e(s,a) = γλe(s,a) + 1 if (s,a) is current state-action pair
     e(s,a) = γλe(s,a) otherwise
     ```

2. **Value Function Update**
   - For state-value prediction:
     ```
     V(s) = V(s) + αδe(s)
     ```
   - For action-value prediction:
     ```
     Q(s,a) = Q(s,a) + αδe(s,a)
     ```
   - where δ is the TD error

## Advantages
- Faster learning than TD(0)
- More efficient than Monte Carlo methods
- Can handle both episodic and continuing tasks
- Provides a smooth transition between TD(0) and Monte Carlo methods

## Implementation Considerations
- Need to maintain eligibility traces for all states/state-action pairs
- Memory requirements increase with the number of states/actions
- Need to choose appropriate values for:
  - λ (eligibility trace decay)
  - α (learning rate)
  - γ (discount factor)

## Common Applications
- Policy evaluation
- Policy improvement
- Model-free control
- Function approximation 


## Homework 8: Policy Gradient Methods

### Overview
This homework focuses on implementing and analyzing Policy Gradient methods, specifically Actor-Critic with Eligibility Traces, using the Mountain Car environment as a case study. The implementation demonstrates how to solve continuous control problems using policy-based methods.

### Key Components

#### 1. Mountain Car Environment
- Continuous state space: (position, velocity)
- Action space: {0: accelerate left, 1: do nothing, 2: accelerate right}
- Transition dynamics:
  - Position: x_{t+1} = bound[x_t + ẋ_{t+1}]
  - Velocity: ẋ_{t+1} = bound[ẋ_t + 0.001(A_t-1) - 0.0025cos(3x_t)]
- Reward: -1 per step until goal reached
- Goal: Reach position ≥ 0.5
- Initial state: Random position in [-0.6, -0.4) with zero velocity

#### 2. Actor-Critic with Eligibility Traces
- Actor: Policy network using softmax function
- Critic: Value function approximation
- Eligibility traces for both actor and critic
- Fourier basis function approximation
- Parameter tuning for optimal performance

#### 3. Implementation Details
```python
def actor_critic_with_eligibility_traces(env, alpha_w, alpha_theta, gamma, lambd, order, num_episodes):
    # Actor-Critic implementation with eligibility traces
    # Parameters:
    # - alpha_w: critic learning rate
    # - alpha_theta: actor learning rate
    # - gamma: discount factor
    # - lambd: eligibility trace decay
    # - order: Fourier basis order
    # - num_episodes: number of training episodes
```

### Technical Implementation

#### 1. Fourier Basis Functions
```python
def get_fourier_basis(order, dimensions):
    # Generate Fourier basis functions for state representation
    c = list(itertools.product(range(order + 1), repeat=dimensions))
    return np.array(c)
```

#### 2. Policy Implementation
```python
def softmax_policy(theta, phi_s):
    # Softmax policy for action selection
    preferences = np.dot(theta, phi_s)
    max_pref = np.max(preferences)
    exp_preferences = np.exp(preferences - max_pref)
    return exp_preferences / np.sum(exp_preferences)
```

### Performance Analysis

#### 1. Learning Curves
- Reward per episode vs. number of episodes
- Steps per episode vs. number of episodes
- Averaged over multiple runs (50-100)

#### 2. Policy Visualization
- Animation of learned policy
- Analysis of policy optimality
- Comparison of episode lengths

### Key Findings
1. Actor-Critic with eligibility traces provides stable learning
2. Fourier basis functions effectively represent continuous state space
3. Parameter tuning significantly impacts performance
4. Policy gradient methods can solve complex control problems
5. Eligibility traces improve learning efficiency

### Interactive Demonstrations
1. **Policy Learning Process**
   - Real-time visualization of policy updates
   - Action selection demonstration
   - Value function approximation

2. **Performance Metrics**
   - Learning curve visualization
   - Step count analysis
   - Reward accumulation

3. **Parameter Impact**
   - Learning rate effects
   - Eligibility trace decay analysis
   - Basis function order comparison

## Midterm Project: A Unifying RL Algorithm

### Overview
This project focuses on implementing and analyzing n-step SARSA, a unifying reinforcement learning algorithm that bridges the gap between one-step TD methods and Monte Carlo methods. The implementation is tested in two distinct environments: Cliff Walking and Taxi, demonstrating the algorithm's versatility and effectiveness.

### Key Components

#### 1. n-step SARSA Algorithm
- Combines advantages of TD(0) and Monte Carlo methods
- Updates Q-values based on n-step returns
- Flexible parameter tuning for optimal performance
- Efficient learning through balanced exploration and exploitation

#### 2. Cliff Walking Environment
- 4x12 grid world environment
- State space: Grid positions
- Action space: {up, right, down, left}
- Reward structure:
  - -1 per step
  - -100 for falling off cliff
  - +1 for reaching goal
- Goal: Navigate to goal while avoiding cliff

#### 3. Taxi Environment
- Complex state space: (taxi position, passenger location, destination)
- Action space: {south, north, east, west, pickup, dropoff}
- Reward structure:
  - +20 for successful dropoff
  - -10 for illegal actions
  - -1 per step
- Goal: Efficiently pick up and drop off passengers

### Technical Implementation

#### 1. n-step SARSA Algorithm
```python
def n_step_sarsa(env, n, alpha, gamma, epsilon, num_episodes):
    """
    n-step SARSA implementation
    Parameters:
    - n: number of steps for return calculation
    - alpha: learning rate
    - gamma: discount factor
    - epsilon: exploration rate
    - num_episodes: number of training episodes
    """
```

#### 2. Policy Implementation
```python
def epsilon_greedy_policy(Q, state, epsilon, nA):
    """
    Epsilon-greedy policy for action selection
    Parameters:
    - Q: action-value function
    - state: current state
    - epsilon: exploration rate
    - nA: number of actions
    """
```

### Performance Analysis

#### 1. Cliff Walking Results
- 50 separate runs with 500 episodes each
- n-step return: 3
- Learning rate: 0.1
- Discount factor: 0.99
- Initial epsilon: 0.1
- Minimum epsilon: 0.01
- Epsilon decay: 0.99

#### 2. Taxi Environment Results
- 5 runs with 10,000 episodes each
- n-step return: 5
- Learning rate: 0.2
- Discount factor: 0.99
- Initial epsilon: 1.0
- Minimum epsilon: 0.01
- Epsilon decay: 0.999

### Key Findings
1. n-step SARSA successfully learns optimal policies in both environments
2. Parameter tuning significantly impacts learning performance
3. Higher n-step returns improve learning in complex environments
4. Exploration strategy crucial for optimal policy discovery
5. Algorithm demonstrates versatility across different problem domains

### Interactive Demonstrations
1. **Learning Process**
   - Real-time policy updates
   - Action selection visualization
   - Reward accumulation tracking

2. **Policy Evaluation**
   - Optimal path visualization
   - Performance metrics analysis
   - Convergence rate comparison

3. **Environment Comparison**
   - Cliff Walking vs Taxi behavior
   - Parameter impact analysis
   - Learning efficiency comparison

## Final Project: Actor-Critic with Eligibility Traces

### Overview
This project focuses on implementing and analyzing an Actor-Critic algorithm with Eligibility Traces to solve the CartPole-v1 environment. The implementation demonstrates how to effectively balance exploration and exploitation while learning a stable policy for continuous control tasks.

### Key Components

#### 1. CartPole-v1 Environment
- State space: 4-dimensional continuous vector
  - Cart Position (x): [-4.8, 4.8]
  - Cart Velocity (ẋ): [-3.0, 3.0]
  - Pole Angle (θ): [-0.418, 0.418] radians
  - Pole Angular Velocity (θ̇): [-3.5, 3.5] radians/sec
- Action space: Discrete with two actions
  - Push Left (0)
  - Push Right (1)
- Reward: +1 for each time step the pole remains balanced
- Termination conditions:
  - Pole angle exceeds ±12° (±0.209 radians)
  - Cart position exceeds ±2.4 units
  - Episode reaches 500 time steps

#### 2. Actor-Critic Implementation
- Actor: Policy network using softmax function
- Critic: Value function approximation using Fourier basis
- Eligibility traces for both actor and critic
- Parameter tuning for optimal performance

#### 3. Technical Implementation
```python
class ActorCritic:
    def __init__(self, env, alpha_w=0.001, alpha_theta=0.0001, gamma=1.0, lambd=0.9, order=3):
        # Initialize parameters
        self.env = env
        self.alpha_w = alpha_w  # Critic learning rate
        self.alpha_theta = alpha_theta  # Actor learning rate
        self.gamma = gamma  # Discount factor
        self.lambd = lambd  # Eligibility trace decay
        self.order = order  # Fourier basis order
```

### Performance Analysis

#### 1. Training Configuration
- 50 separate runs
- 500 episodes per run
- Hyperparameters:
  - Critic learning rate (α_w): 0.0001
  - Actor learning rate (α_θ): 0.000001
  - Discount factor (γ): 1.0
  - Eligibility trace decay (λ): 0.9
  - Fourier basis order: 4

#### 2. Results
- Maximum Average Reward: 500.0
- Minimum Average Steps per Episode: 19.98
- Stable learning across multiple runs
- Consistent policy convergence

### Key Findings
1. Actor-Critic with eligibility traces provides stable learning in the CartPole environment
2. Fourier basis functions effectively represent the continuous state space
3. Careful parameter tuning is crucial for optimal performance
4. Eligibility traces improve learning efficiency by maintaining temporal credit assignment
5. The algorithm successfully learns to balance the pole for extended periods

### Interactive Demonstrations
1. **Learning Process**
   - Real-time visualization of policy updates
   - Action selection demonstration
   - Value function approximation

2. **Performance Metrics**
   - Learning curve visualization
   - Step count analysis
   - Reward accumulation tracking

3. **Policy Evaluation**
   - Real-time pole balancing demonstration
   - Action selection analysis
   - Performance metrics tracking

### Technical Implementation Details
- Efficient state representation using Fourier basis
- Vectorized operations for better performance
- Proper handling of continuous states
- Comprehensive error handling
- Modular design for easy testing and modification

## Author

Harshal Dafade
