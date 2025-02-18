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
def generate_oscillatory_data(n_points=100, x_range=(0, 1), amplitude=1.0, frequency=1.0, phase=0.0, add_noise=False, noise_level=0.1):

    x = np.linspace(x_range[0], x_range[1], n_points)
    y = amplitude * np.sin(2 * np.pi * frequency * x + phase)
    
    if add_noise:
        y += np.random.normal(0, noise_level, n_points)
    other_info = f"amplitude={amplitude}, frequency={frequency}, phase={phase}, noise_level={noise_level if add_noise else 0}"
    
    print(f"Data generated: {n_points}, {x_range}, {other_info}")
    
    return x, y
if __name__ == "__main__":
    # Generate data without noise
    x_clean, y_clean = generate_oscillatory_data(n_points=150, x_range=(0, 2*np.pi), amplitude=1.0, frequency=1.0, phase=0.0, add_noise=False)
    
    # Generate data with noise
    x_noisy, y_noisy = generate_oscillatory_data(n_points=150, x_range=(0, 2*np.pi), amplitude=1.0, frequency=1.0, phase=0.0, add_noise=True, noise_level=0.2)

#3. Clustering (KMeans)
def clustering_variance(data, max_clusters=10):
    
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia = kmeans.inertia_
        inertias.append(inertia)
        print(f"Number of clusters: {k}, Inertia (Variance): {inertia:.4f}")
    return inertias

# Example: Create some 2D data
# 1D oscillatory function and then combine it with noise to create 2D data.
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)
data = np.column_stack((x, y))

# Print information about the clustering method
print("Clustering method: KMeans from scikit-learn")
print("Parameters: random_state=42 (for reproducibility), default n_init and max_iter")

# Compute inertia values for cluster numbers 1 to 10
max_clusters = 10
inertias = clustering_variance(data, max_clusters=max_clusters)

# Plot the variance (inertia) versus the number of clusters
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_clusters+1), inertias, marker='o', linestyle='-')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Variance)")
plt.title("Elbow Plot: Variance vs. Number of Clusters")
plt.grid(True)
plt.show()

#4.Use LR,  NN and PINNS to make a regression of such data

# Convert x and y to torch tensors (shape: [100, 1])
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Define Regression Models
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
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

# PINN
# We use the same architecture as NN but include an additional physics loss in training.
class PINNModel(nn.Module):
    def __init__(self, hidden_size=50):
        super(PINNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

# Training Functions
def train_model(model, x_train, y_train, num_epochs=1000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
    return model

def train_pinn(model, x_train, y_train, num_epochs=1000, lr=1e-3, lambda_phy=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    # Collocation points for computing the physics loss:
    x_colloc = torch.linspace(x_train.min(), x_train.max(), 100).unsqueeze(1)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Data loss:
        pred = model(x_train)
        loss_data = loss_fn(pred, y_train)
        # Physics loss:
        x_colloc_var = x_colloc.clone().detach().requires_grad_(True)
        u = model(x_colloc_var)
        # First derivative:
        u_x = torch.autograd.grad(u, x_colloc_var, torch.ones_like(u), create_graph=True)[0]
        # Second derivative:
        u_xx = torch.autograd.grad(u_x, x_colloc_var, torch.ones_like(u_x), create_graph=True)[0]
        # For sin(x), the equation is: u''(x) + u(x) = 0.
        physics_residual = u_xx + u
        loss_phy = loss_fn(physics_residual, torch.zeros_like(u))
        loss = loss_data + lambda_phy * loss_phy
        loss.backward()
        optimizer.step()
    return model

# Training
# LR
lr_model = LRModel()
lr_model = train_model(lr_model, x_tensor, y_tensor, num_epochs=1000, lr=1e-2)
print("Task completed LR")

# NN
nn_model = NNModel(hidden_size=50)
nn_model = train_model(nn_model, x_tensor, y_tensor, num_epochs=1000, lr=1e-3)
print("Task completed NN")

# PINN
pinn_model = PINNModel(hidden_size=50)
pinn_model = train_pinn(pinn_model, x_tensor, y_tensor, num_epochs=1000, lr=1e-3, lambda_phy=1.0)
print("Task completed PINN")

# 5 6 7

# Define a training function that records history
def train_with_history(model, x_train, y_train, num_epochs=1000, lr=1e-3, use_physics=False, lambda_phy=1.0):
    """
    Trains the given model while recording:
      - error_history: training loss at each epoch.
      - solution_history: model predictions on a fixed test grid (recorded every 100 epochs).
    
    If use_physics is True, adds a physics loss enforcing: u''(x) + u(x) = 0.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    error_history = []
    solution_history = {}
    # Fixed test grid for monitoring predictions:
    test_x = torch.linspace(x_train.min(), x_train.max(), 100).unsqueeze(1)
    if use_physics:
        x_colloc = torch.linspace(x_train.min(), x_train.max(), 100).unsqueeze(1)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        loss_data = loss_fn(pred, y_train)
        if use_physics:
            x_colloc_var = x_colloc.clone().detach().requires_grad_(True)
            u = model(x_colloc_var)
            u_x = torch.autograd.grad(u, x_colloc_var, torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_colloc_var, torch.ones_like(u_x), create_graph=True)[0]
            physics_residual = u_xx + u  # For sin(x), u'' + u = 0
            loss_phy = loss_fn(physics_residual, torch.zeros_like(u))
            loss = loss_data + lambda_phy * loss_phy
        else:
            loss = loss_data
        loss.backward()
        optimizer.step()
        error_history.append(loss.item())
        if epoch % 100 == 0:
            # Record the model's predictions on the test grid
            with torch.no_grad():
                solution_history[epoch] = model(test_x).detach().numpy().flatten()
    return error_history, solution_history

# For tasks 5,6,7 we assume the "true" function is sin(x)
true_function = lambda x: np.sin(x)
test_x_np = np.linspace(x_tensor.min().item(), x_tensor.max().item(), 100)

# ---------------------------
# Train LR with history as well.
# ---------------------------
lr_model_hist = LRModel()
# Using a slightly higher learning rate for LR:
lr_error_history, lr_solution_history = train_with_history(lr_model_hist, x_tensor, y_tensor, num_epochs=1000, lr=1e-2, use_physics=False)
#print("Training history recorded for LR.")

# ---------------------------
# Train NN with history (already trained above without history; retrain with history):
# ---------------------------
nn_model_hist = NNModel(hidden_size=50)
nn_error_history, nn_solution_history = train_with_history(nn_model_hist, x_tensor, y_tensor, num_epochs=1000, lr=1e-3, use_physics=False)
#print("Training history recorded for NN.")

# ---------------------------
# Train PINN with history:
# ---------------------------
pinn_model_hist = PINNModel(hidden_size=50)
pinn_error_history, pinn_solution_history = train_with_history(pinn_model_hist, x_tensor, y_tensor, num_epochs=1000, lr=1e-3, use_physics=True, lambda_phy=1.0)
#print("Training history recorded for PINN.")


#5. Animate the Regression Function Evolution

def animate_solution(solution_history, test_x, title="Solution Evolution"):
    """
    Animates the evolution of the predicted regression function.
    Uses a loop with plt.pause() to update the plot.
    """
    fig, ax = plt.subplots()
    epochs = sorted(solution_history.keys())
    line, = ax.plot(test_x, solution_history[epochs[0]], 'b-', lw=2)
    ax.plot(test_x, true_function(test_x), 'k--', lw=2, label="True Function")
    ax.set_title(f"{title} at epoch {epochs[0]}")
    ax.set_xlabel("x")
    ax.set_ylabel("Predicted y")
    ax.legend()
    plt.ion()
    plt.show()
    for epoch in epochs:
        line.set_ydata(solution_history[epoch])
        ax.set_title(f"{title} at epoch {epoch}")
        plt.draw()
        plt.pause(0.3)
    plt.ioff()
    plt.show()

print("Animating NN regression evolution:")
animate_solution(nn_solution_history, test_x_np, title="NN Regression Evolution")
print("Animating PINN regression evolution:")
animate_solution(pinn_solution_history, test_x_np, title="PINN Regression Evolution")


#6. Plot Error vs. Iteration
def compute_errors(solution_history, test_x, true_func):
    errors = {}
    for epoch, sol in solution_history.items():
        errors[epoch] = np.mean((sol - true_func(test_x))**2)
    return errors

lr_errors = compute_errors(lr_solution_history, test_x_np, true_function)
nn_errors = compute_errors(nn_solution_history, test_x_np, true_function)
pinn_errors = compute_errors(pinn_solution_history, test_x_np, true_function)

plt.figure(figsize=(10,5))
plt.plot(sorted(lr_errors.keys()), [lr_errors[k] for k in sorted(lr_errors.keys())], 'r-', label="LR Error")
plt.plot(sorted(nn_errors.keys()), [nn_errors[k] for k in sorted(nn_errors.keys())], 'b-', label="NN Error")
plt.plot(sorted(pinn_errors.keys()), [pinn_errors[k] for k in sorted(pinn_errors.keys())], 'g-', label="PINN Error")
plt.xlabel("Epoch")
plt.ylabel("MSE Error")
plt.title("Error vs. Iteration (w.r.t. True Function)")
plt.legend()
plt.show()


#7. Plot Progress Monitoring Variable (Training Loss vs. Iteration)

plt.figure(figsize=(10,5))
plt.plot(range(len(lr_error_history)), lr_error_history, 'r-', label="LR Training Loss")
plt.plot(range(len(nn_error_history)), nn_error_history, 'b-', label="NN Training Loss")
plt.plot(range(len(pinn_error_history)), pinn_error_history, 'g-', label="PINN Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Training Loss vs. Iteration (Progress Monitor)")
plt.legend()
plt.show()

#Bonus 
fig, axs = plt.subplots(3, 1, figsize=(12, 12))
# Subplot 1: Final Regression Function vs. x (for LR, NN, PINN)
final_epoch_lr = sorted(lr_solution_history.keys())[-1]
final_epoch_nn = sorted(nn_solution_history.keys())[-1]
final_epoch_pinn = sorted(pinn_solution_history.keys())[-1]
axs[0].plot(test_x_np, lr_solution_history[final_epoch_lr], 'r-', lw=2, label="LR Prediction")
axs[0].plot(test_x_np, nn_solution_history[final_epoch_nn], 'b-', lw=2, label="NN Prediction")
axs[0].plot(test_x_np, pinn_solution_history[final_epoch_pinn], 'g-', lw=2, label="PINN Prediction")
axs[0].plot(test_x_np, true_function(test_x_np), 'k--', lw=2, label="True Function")
axs[0].set_title("Final Regression Functions")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()

# Subplot 2: Error vs. Iteration (for all methods)
axs[1].plot(sorted(lr_errors.keys()), [lr_errors[k] for k in sorted(lr_errors.keys())], 'r-', label="LR Error")
axs[1].plot(sorted(nn_errors.keys()), [nn_errors[k] for k in sorted(nn_errors.keys())], 'b-', label="NN Error")
axs[1].plot(sorted(pinn_errors.keys()), [pinn_errors[k] for k in sorted(pinn_errors.keys())], 'g-', label="PINN Error")
axs[1].set_title("Error vs. Iteration")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("MSE Error")
axs[1].legend()

# Subplot 3: Training Loss vs. Iteration (for all methods)
axs[2].plot(range(len(lr_error_history)), lr_error_history, 'r-', label="LR Training Loss")
axs[2].plot(range(len(nn_error_history)), nn_error_history, 'b-', label="NN Training Loss")
axs[2].plot(range(len(pinn_error_history)), pinn_error_history, 'g-', label="PINN Training Loss")
axs[2].set_title("Training Loss vs. Iteration")
axs[2].set_xlabel("Iteration")
axs[2].set_ylabel("Training Loss")
axs[2].legend()

plt.tight_layout()
plt.show()



#8. RL script
# GridWorld Environment
class GridWorld:
    
    def __init__(self, size=5, num_obstacles=5):
        self.size = size
        self.num_obstacles = num_obstacles
        self.obstacles = [(0, 4), (4, 3), (1, 3), (1, 0), (3, 2)]
        self.state_space = np.zeros((self.size, self.size))
        self.state = (0, 0)
        self.goal = (self.size-1, self.size-1)

    def step(self, action):
        """
        Take a step in the environment.
        The agent takes a step in the environment based on the action it chooses.

        Args:
            action (int): The action the agent takes.
                0: up
                1: right
                2: down
                3: left

        Returns:
            state (tuple): The new state of the agent.
            reward (float): The reward the agent receives.
            done (bool): Whether the episode is done or not.
        """
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
         #   self.state = (0, 0)
            return self.state, -1, True
        if self.state == self.goal:
            return self.state, 1, True
        return self.state, -0.1, False

    def reset(self):
        """
        Reset the environment.
        The agent is placed back at the top-left corner of the grid.

        Args:
            None

        Returns:
            state (tuple): The new state of the agent.
        """
        self.state = (0, 0)
        return self.state
# Q-Learning
class QLearning:
    
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1, episodes=10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((self.env.size, self.env.size, 4))

    def choose_action(self, state):
       
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2, 3])  # exploration
        else:
            return np.argmax(self.q_table[state])  # exploitation

    def update_q_table(self, state, action, reward, new_state):
        
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
            self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]))

    def train(self):
        
        rewards = []
        states = []  # Store states at each step
        starts = []  # Store the start of each new episode
        steps_per_episode = []  # Store the number of steps per episode
        steps = 0  # Initialize the step counter outside the episode loop
        episode = 0
        #print(self.q_table)
        while episode < self.episodes:
            state = self.env.reset()
            total_reward = 0
            done = False
            #print(f"Episode {episode+1}")
            #print(self.q_table)
            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                total_reward += reward
                states.append(state)  # Store state
                steps += 1  # Increment the step counter
                if done and state == self.env.goal:  # Check if the agent has reached the goal
                    starts.append(len(states))  # Store the start of the new episode
                    rewards.append(total_reward)
                    steps_per_episode.append(steps)  # Store the number of steps for this episode
                    steps = 0  # Reset the step counter
                    episode += 1
        return rewards, states, starts, steps_per_episode
for i in range(1):
    env = GridWorld(size=5, num_obstacles=5)
    agent = QLearning(env)


    # Train the agent and get rewards
    rewards, states, starts, steps_per_episode = agent.train()  # Get starts and steps_per_episode as well

    # Visualize the agent moving in the grid
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(i):
        """
        Update the grid with the agent's movement.

        Args:
            i (int): The current step.

        Returns:
            None
        """
        ax.clear()
        # Calculate the cumulative reward up to the current step
        #print(rewards)
        cumulative_reward = sum(rewards[:i+1])
        #print(rewards[:i+1])
        # Find the current episode
        current_episode = next((j for j, start in enumerate(starts) if start > i), len(starts)) - 1
        # Calculate the number of steps since the start of the current episode
        if current_episode < 0:
            steps = i + 1
        else:
            steps = i - starts[current_episode] + 1
        ax.set_title(f"Episode: {current_episode+1}, Number of Steps to Reach Target: {steps}")
        grid = np.zeros((env.size, env.size))
        for obstacle in env.obstacles:

            grid[obstacle] = -1
        grid[env.goal] = 1
        grid[states[i]] = 0.5  # Use states[i] instead of env.state
        ax.imshow(grid, cmap='magma')

    ani = animation.FuncAnimation(fig, update, frames=range(len(states)), repeat=False)
    print(f"Environment number {i+1}")
    for i, steps in enumerate(steps_per_episode, 1):
        print(f"Episode {i}: {steps} Number of Steps to Reach Target ")
    #print(f"Total reward: {sum(rewards):.2f}")
    print()
    #ani.save('gridworld_lin.gif', writer='pillow', dpi=100)

    plt.show()
    #plt.close()
print("Reinforcement Learning script completed.")
