import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 1. A class to generate 2D datasets
class DataGenerator2D:
    def __init__(self, dataset_type="random", num_points=100, x_range=(0, 10), y_range=(0, 10),
                 noise_level=0.0, function=None):
        self.dataset_type = dataset_type
        self.num_points = num_points
        self.x_range = x_range
        self.y_range = y_range
        self.noise_level = noise_level
        self.function = function
        self.data = None
        self.metadata = {}
        
    def generate(self):
        if self.dataset_type == "random":
            # Generate uniformly random points in the specified x and y ranges.
            x = np.random.uniform(self.x_range[0], self.x_range[1], self.num_points)
            y = np.random.uniform(self.y_range[0], self.y_range[1], self.num_points)
            self.data = np.column_stack((x, y))
            self.metadata = {
                "type": "random",
                "num_points": self.num_points,
                "x_range": self.x_range,
                "y_range": self.y_range
            }
        elif self.dataset_type == "function":
            # Generate points along a function f(x) in the x_range.
            x = np.linspace(self.x_range[0], self.x_range[1], self.num_points)
            if self.function is None:
                # Default function is sine.
                y_true = np.sin(x)
                func_name = "sin"
            else:
                y_true = self.function(x)
                func_name = self.function.__name__
            # Add noise if requested.
            noise = np.random.normal(0, self.noise_level, self.num_points)
            y = y_true + noise
            self.data = np.column_stack((x, y))
            self.metadata = {
                "type": "function",
                "num_points": self.num_points,
                "x_range": self.x_range,
                "function": func_name,
                "noise_level": self.noise_level
            }
        elif self.dataset_type == "custom":
            # Generate points approximately on a circle.
            angles = np.linspace(0, 2 * np.pi, self.num_points)
            radius = 5  # Define a fixed radius
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            # Optionally add noise to both coordinates.
            if self.noise_level > 0:
                x += np.random.normal(0, self.noise_level, self.num_points)
                y += np.random.normal(0, self.noise_level, self.num_points)
            self.data = np.column_stack((x, y))
            self.metadata = {
                "type": "custom",
                "num_points": self.num_points,
                "shape": "circle",
                "radius": radius,
                "noise_level": self.noise_level
            }
        else:
            raise ValueError("Unknown dataset_type. Choose 'random', 'function', or 'custom'.")
        return self.data, self.metadata

# Example usage for Task 1:
if __name__ == "__main__":
    # Create an instance to generate a random 2D dataset:
    generator = DataGenerator2D(dataset_type="random", num_points=100, x_range=(0, 10), y_range=(0, 10))
    data, metadata = generator.generate()
    print("Generated 2D dataset (random):")
    print(metadata)

# 2. Generate Different 2D Datasets Using DataGenerator2D

# 2.1: Generate a 2D random dataset.
random_gen = DataGenerator2D(dataset_type="random", num_points=100, x_range=(0, 10), y_range=(0, 10))
data_random, meta_random = random_gen.generate()
print("2D Random Dataset Metadata:")
print(meta_random)

# 2.2: Generate a dataset with noise around a function.
# Here we use the sine function as our base.
function_gen = DataGenerator2D(dataset_type="function", num_points=100, x_range=(0, 2*np.pi),
                               noise_level=0.2, function=np.sin)
data_function, meta_function = function_gen.generate()
print("2D Function Dataset Metadata:")
print(meta_function)

# 2.3: Generate a "custom" dataset (your truth, not a line). In this example, points on a circle.
custom_gen = DataGenerator2D(dataset_type="custom", num_points=100, noise_level=0.3)
data_custom, meta_custom = custom_gen.generate()
print("2D Custom Dataset Metadata:")
print(meta_custom)

# Optionally, plot the three datasets to visualize:
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].scatter(data_random[:,0], data_random[:,1], color='blue', alpha=0.6)
axs[0].set_title("2D Random Dataset")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

axs[1].scatter(data_function[:,0], data_function[:,1], color='green', alpha=0.6)
axs[1].set_title("Function Dataset (with Noise)")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

axs[2].scatter(data_custom[:,0], data_custom[:,1], color='red', alpha=0.6)
axs[2].set_title("Custom Dataset (Circle)")
axs[2].set_xlabel("x")
axs[2].set_ylabel("y")

plt.tight_layout()
plt.show()

# 3. Generate Two Datasets and Append Them

# Generate a dataset from a function (e.g., a noisy sine curve)
function_gen = DataGenerator2D(
    dataset_type="function", 
    num_points=150, 
    x_range=(0, 2*np.pi), 
    noise_level=0.2, 
    function=np.sin
)
data_function, meta_function = function_gen.generate()

# Generate a "custom" dataset (points on a circle)
custom_gen = DataGenerator2D(
    dataset_type="custom", 
    num_points=150, 
    noise_level=0.3
)
data_custom, meta_custom = custom_gen.generate()

# Append (combine) the two datasets (vertical stacking)
combined_data = np.vstack((data_function, data_custom))
combined_metadata = {
    "dataset1": meta_function,
    "dataset2": meta_custom,
    "total_points": combined_data.shape[0]
}

print("Combined Dataset Metadata:")
print(combined_metadata)

# # Plot the combined dataset
# plt.figure(figsize=(8,6))
# plt.scatter(combined_data[:, 0], combined_data[:, 1], c='purple', alpha=0.6)
# plt.title("Combined 2D Dataset")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid(True)
# plt.show()

# 4. Write a File with the Data and Save the Plots

output_dir = os.path.join(",,", "data")
os.makedirs(output_dir, exist_ok=True)

# Save the combined dataset as a CSV file.
data_file = os.path.join(output_dir, "combined_data.csv")
np.savetxt(data_file, combined_data, delimiter=",", header="x,y", comments="")

# Save a plot of the combined data.
plot_file = os.path.join(output_dir, "combined_data.png")
plt.figure(figsize=(6,6))
plt.scatter(combined_data[:, 0], combined_data[:, 1], c='purple', alpha=0.6)
plt.title("Combined 2D Dataset")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.savefig(plot_file)
plt.close()

# 5. Save Metadata to json

metadata_file = os.path.join(output_dir, "combined_metadata.json")
with open(metadata_file, "w") as f:
    json.dump(combined_metadata, f, indent=4)

print("Task 3, 4, and 5 completed.")
print("Combined data saved to:", data_file)
print("Plot saved to:", plot_file)
print("Metadata saved to:", metadata_file)

