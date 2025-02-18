import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
import time

#1. Fix the mse_scaling_2.py code
from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
from sklearn.metrics import mean_squared_error as sk_mse
import timeit as it

observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]
karg = {'observed': observed, 'predicted': predicted}

def sk_mse_interface(observed, predicted):
    return sk_mse(observed, predicted)

factory = {
    'mse_vanilla': vanilla_mse,
    'mse_numpy': numpy_mse,
    'mse_sk': sk_mse_interface
}

mse_results = {}
for name, func in factory.items():
    exec_time = it.timeit('{func(**karg)}', globals=globals(), number=100) / 100
    mse_val = func(**karg)
    mse_results[name] = mse_val
    print(f"Mean Squared Error, {name}: {mse_val} | Avg exec time: {exec_time:.6f} sec")

if mse_results['mse_vanilla'] == mse_results['mse_numpy'] == mse_results['mse_sk']:
    print("Test successful")
else:
    print("Test FAILED: MSE values do not match")

#2. 1d oscillatory function with and without noise
def generate_oscillatory_data(x_range=(0, 1), num_points=100, amplitude=1.0,
                              frequency=1.0, phase=0.0, add_noise=False, noise_level=0.1):
    
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = amplitude * np.sin(2 * np.pi * frequency * x + phase)
    if add_noise:
        y += np.random.normal(0, noise_level, num_points)
    return x, y

# Parameters
n_points = 200
data_range = (0, 1)
amplitude = 1.0
frequency = 2.0
phase = 0.0
noise_level = 0.2

x, y = generate_oscillatory_data(x_range=data_range, num_points=n_points,
                                 amplitude=amplitude, frequency=frequency,
                                 phase=phase, add_noise=True, noise_level=noise_level)
other_info = f"amplitude={amplitude}, frequency={frequency}, phase={phase}, noise_level={noise_level}"
print(f"Data generated: {n_points}, {data_range}, {other_info}")

# Plot
plt.figure(figsize=(8, 4))
plt.scatter(x, y, label="Noisy Data", color="blue", alpha=0.6)
plt.plot(x, amplitude * np.sin(2*np.pi*frequency*x + phase), 'k--', label="True Function")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Oscillatory Data (with Noise)")
plt.legend()
plt.show()

#3. Clustering (KMeans)
def clustering_variance(data, max_clusters=10):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        inertia = kmeans.inertia_
        inertias.append(inertia)
        print(f"Clustering: k={k}, Inertia (Variance): {inertia:.4f}")
    return inertias

data_2d = np.column_stack((x, y))
inertias = clustering_variance(data_2d, max_clusters=10)
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Variance)")
plt.title("Elbow Plot: Variance vs. Number of Clusters")
plt.show()
print("Clustering method: KMeans (scikit-learn), default parameters with random_state=0")

#Use LR,  NN and PINNS to make a regression of such data.

# LR
class LRModel(nn.Module):
    def __init__(self):
        super(LRModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

# NN
class NNModel(nn.Module):
    def __init__(self, hidden_size=50):
        super(NNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.model(x)

# PINN
class PINNModel(nn.Module):
    def __init__(self, hidden_size=50):
        super(PINNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.model(x)


# Training Function for Regression
def train_model(model, x_train, y_train, num_epochs=1000, lr=1e-3, use_physics=False, frequency=1.0, lambda_phy=1.0):
    """
    Trains the model using MSE loss.
    If use_physics=True, adds a physics loss enforcing: 
         u''(x) + (2*pi*frequency)**2 * u(x) = 0
    Returns:
      error_history: list of loss values per epoch.
      solution_history: dictionary mapping epoch -> predicted solution (on a fixed grid).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    error_history = []
    solution_history = {}
    # Fixed test grid for monitoring solution evolution
    test_x = torch.linspace(x_train.min(), x_train.max(), 100).unsqueeze(1)
    if use_physics:
        x_colloc = torch.linspace(x_train.min(), x_train.max(), 200).unsqueeze(1)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        loss_data = loss_fn(pred, y_train)
        if use_physics:
            x_colloc_tensor = x_colloc.clone().detach().requires_grad_(True)
            u = model(x_colloc_tensor)
            u_x = torch.autograd.grad(u, x_colloc_tensor, torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_colloc_tensor, torch.ones_like(u_x), create_graph=True)[0]
            physics_residual = u_xx + (2*np.pi*frequency)**2 * u
            loss_phy = loss_fn(physics_residual, torch.zeros_like(u))
            loss = loss_data + lambda_phy * loss_phy
        else:
            loss = loss_data
        loss.backward()
        optimizer.step()
        error_history.append(loss.item())
        if epoch % 100 == 0:
            with torch.no_grad():
                solution_history[epoch] = model(test_x).detach().numpy().flatten()
    return error_history, solution_history

# Prepare data as torch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Train LR
lr_model = LRModel()
lr_error_history, lr_solution_history = train_model(lr_model, x_tensor, y_tensor,
                                                    num_epochs=2000, lr=1e-2, use_physics=False)
print("Task completed LR")

# Train NN
nn_model = NNModel(hidden_size=50)
nn_error_history, nn_solution_history = train_model(nn_model, x_tensor, y_tensor,
                                                    num_epochs=2000, lr=1e-3, use_physics=False)
print("Task completed NN")

# Train PINN
pinn_model = PINNModel(hidden_size=50)
pinn_error_history, pinn_solution_history = train_model(pinn_model, x_tensor, y_tensor,
                                                        num_epochs=2000, lr=1e-3, use_physics=True,
                                                        frequency=frequency, lambda_phy=1.0)
print("Task completed PINN")

#5. Plot the regression function NN PINN
def animate_solution(solution_history, test_x, title="Solution Evolution"):
   
    fig, ax = plt.subplots()
    init_epoch = sorted(solution_history.keys())[0]
    line, = ax.plot(test_x, solution_history[init_epoch], 'b-', lw=2)
    true_curve = amplitude * np.sin(2*np.pi*frequency*test_x + phase)
    ax.plot(test_x, true_curve, 'k--', lw=2, label="True Function")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Predicted y")
    ax.legend()
    plt.ion()
    plt.show()
    for epoch in sorted(solution_history.keys()):
        line.set_ydata(solution_history[epoch])
        ax.set_title(f"{title} at epoch {epoch}")
        plt.draw()
        plt.pause(0.2)
    plt.ioff()
    plt.show()

test_x = np.linspace(x_tensor.min().item(), x_tensor.max().item(), 100)
print("Animating NN regression evolution:")
animate_solution(nn_solution_history, test_x, title="NN Regression Evolution")
print("Animating PINN regression evolution:")
animate_solution(pinn_solution_history, test_x, title="PINN Regression Evolution")

#6. Plot the error as a function of iteration number 
true_y = amplitude * np.sin(2*np.pi*frequency*test_x + phase)
epochs_recorded = sorted(nn_solution_history.keys())  # Assume same epochs recorded for all methods
lr_errors = []
nn_errors = []
pinn_errors = []
for ep in epochs_recorded:
    pred_lr = lr_solution_history[ep]
    pred_nn = nn_solution_history[ep]
    pred_pinn = pinn_solution_history[ep]
    lr_errors.append(np.mean((pred_lr - true_y)**2))
    nn_errors.append(np.mean((pred_nn - true_y)**2))
    pinn_errors.append(np.mean((pred_pinn - true_y)**2))

plt.figure(figsize=(10, 5))
plt.plot(epochs_recorded, lr_errors, 'r-', label="LR Error")
plt.plot(epochs_recorded, nn_errors, 'b-', label="NN Error")
plt.plot(epochs_recorded, pinn_errors, 'g-', label="PINN Error")
plt.xlabel("Iteration (Epoch)")
plt.ylabel("MSE Error")
plt.title("Error vs. Iterations (w.r.t. True Function)")
plt.legend()
plt.show()

#7. No truth
plt.figure(figsize=(10, 5))
plt.plot(range(len(lr_error_history)), lr_error_history, 'r-', label="LR Training Loss")
plt.plot(range(len(nn_error_history)), nn_error_history, 'b-', label="NN Training Loss")
plt.plot(range(len(pinn_error_history)), pinn_error_history, 'g-', label="PINN Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Training Loss vs. Iterations (Progress Monitor)")
plt.legend()
plt.show()

# Bonus
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
for ep in epochs_recorded:
    plt.plot(test_x, nn_solution_history[ep], 'b-', alpha=0.2)
plt.plot(test_x, true_y, 'k--', lw=2, label="True Function")
plt.title("NN Regression Evolution")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(epochs_recorded, nn_errors, 'b-', label="NN Error")
plt.title("NN Error vs. Iterations")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(range(len(nn_error_history)), nn_error_history, 'b-', label="NN Training Loss")
plt.title("NN Training Loss vs. Iterations")
plt.xlabel("Iteration")
plt.legend()

plt.tight_layout()
plt.show()

#8. RL script
print("Running Reinforcement Learning script...")

# --- GridWorld Environment ---
class GridWorld:
    """
    GridWorld environment.
    The agent starts at (0,0) and must reach the goal at (size-1, size-1).
    Obstacles are predefined.
    """
    def __init__(self, size=5, num_obstacles=5):
        self.size = size
        self.num_obstacles = num_obstacles
        # Hard-coded obstacles for demonstration
        self.obstacles = [(0, 4), (4, 3), (1, 3), (1, 0), (3, 2)]
        self.state_space = np.zeros((self.size, self.size))
        self.state = (0, 0)
        self.goal = (self.size-1, self.size-1)
    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            x = max(0, x-1)
        elif action == 1:  # right
            y = min(self.size-1, y+1)
        elif action == 2:  # down
            x = min(self.size-1, x+1)
        elif action == 3:  # left
            y = max(0, y-1)
        self.state = (x, y)
        if self.state in self.obstacles:
            return self.state, -1, True  # end episode if hit obstacle
        if self.state == self.goal:
            return self.state, 1, True  # reward at goal
        return self.state, -0.1, False
    def reset(self):
        self.state = (0, 0)
        return self.state

# --- Q-Learning Agent ---
class QLearning:
    """
    Q-Learning agent for the GridWorld environment.
    """
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1, episodes=10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((self.env.size, self.env.size, 4))
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2, 3])
        else:
            return np.argmax(self.q_table[state])
    def update_q_table(self, state, action, reward, new_state):
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
            self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]))
    def train(self):
        rewards = []
        states = []
        starts = []
        steps_per_episode = []
        steps = 0
        episode = 0
        while episode < self.episodes:
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                total_reward += reward
                states.append(state)
                steps += 1
                if done:
                    starts.append(len(states))
                    rewards.append(total_reward)
                    steps_per_episode.append(steps)
                    steps = 0
                    episode += 1
        return rewards, states, starts, steps_per_episode

env = GridWorld(size=5, num_obstacles=5)
agent = QLearning(env, episodes=10)
rewards, states, starts, steps_per_episode = agent.train()

# Animate agent movement in the grid
fig, ax = plt.subplots(figsize=(8, 6))
def update(i):
    ax.clear()
    # Determine current episode and step
    current_episode = next((j for j, s in enumerate(starts) if s > i), len(starts)) - 1
    if current_episode < 0:
        steps = i + 1
    else:
        steps = i - starts[current_episode] + 1
    ax.set_title(f"Episode: {current_episode+1}, Steps: {steps}")
    grid = np.zeros((env.size, env.size))
    for obs in env.obstacles:
        grid[obs] = -1
    grid[env.goal] = 1
    grid[states[i]] = 0.5
    ax.imshow(grid, cmap='magma')
ani = animation.FuncAnimation(fig, update, frames=range(len(states)), repeat=False)
plt.show()
print("Reinforcement Learning script completed.")
