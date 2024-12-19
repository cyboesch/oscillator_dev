# %%
import math
import matplotlib
import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Node:
    def __init__(self, label, time):
        self.label = label
        self.time = time
        self.children = []
        self.parent = None
        self.latent_value = None
        self.v_rho = None


def kingman_coalescent(n):
    """Generate a Kingman's n-coalescent process."""
    nodes = [Node({i}, 0) for i in range(n)]
    active_nodes = nodes.copy()
    coalescent_events = []
    t = 0

    while len(active_nodes) > 1:
        m = len(active_nodes)
        rate = m * (m - 1) / 2
        delta_t = expon.rvs(scale=1 / rate)
        t += delta_t

        i, j = np.random.choice(m, 2, replace=False)
        node_i, node_j = active_nodes[i], active_nodes[j]

        new_node = Node(node_i.label.union(node_j.label), t)
        new_node.children = [node_i, node_j]
        node_i.parent = new_node
        node_j.parent = new_node

        active_nodes = [node for k, node in enumerate(active_nodes) if k not in (i, j)]
        active_nodes.append(new_node)

        coalescent_events.append((t, new_node))

    return nodes, coalescent_events, active_nodes[0]


def forward_time_process(root, transition_func, leaf_observation_func, initial_value, verbose=False):
    """Simulate the forward-time Markov process on the tree."""

    def recurse(node):
        if not node.children:  # Leaf node
            node.latent_value = leaf_observation_func(node.latent_value)
            if verbose:
                print(f"Leaf node: {node.label}, value: {node.latent_value}")
            return

        for child in node.children:
            child.latent_value = transition_func(
                node.latent_value, node.time, child.time
            )
            if verbose:
                print(
                    f"Internal node: {node.label} -> {child.label}, value: {child.latent_value}"
            )
            recurse(child)

    root.latent_value = initial_value
    if verbose:
        print(f"Root node: {root.label}, initial value: {root.latent_value}")
    recurse(root)


def conditional_brownian_bridge(start, end, T, n_steps=100000):
    """
    Simulate a Brownian bridge between two points.

    :param start: Starting point
    :param end: Ending point
    :param T: Total time
    :param n_steps: Number of time steps
    :return: Time array and array for the Brownian bridge
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    # Generate a standard Brownian motion
    dB = np.random.normal(0, np.sqrt(dt), n_steps)
    B = np.cumsum(dB)
    B = np.insert(B, 0, 0)

    # Convert to Brownian bridge
    X = start + (end - start) * (t / T) + B - t * (B[-1] / T)

    return t, X

sample_path_style = {"alpha": 1.0, "linewidth": 0.5}
coalescent_event_style = {
    "s": 100,
    "zorder": 4,
    "marker": "o",
    "facecolors": "none",
    "linewidth": 3,
}
end_marker_style = {
    "s": 50,
    "zorder": 4,
    "marker": "o",
    "facecolors": "none",
    "linewidth": 2,
}
leaves_style = {
    "s":100,
    "zorder": 5,
    "marker": "v",
    "linewidth": 3,
    "facecolors": "none",
}
root_style = {
    "s": 300,
    "zorder": 5,
    "marker": ".",
    "facecolors": "none",
    "edgecolors": "r",
    "linewidth": 3,
}

def plot_1d_coalescent(leaves, events, root):
    plt.figure(figsize=(10, 10))

    # Create a color map
    color_map = cm.get_cmap('tab20')
    color_index = 0

    def plot_sample_path(node):
        nonlocal color_index
        if node.parent:
            T = node.parent.time - node.time
            t, X = conditional_brownian_bridge(
                node.latent_value, node.parent.latent_value, T
            )
            plt.plot(node.time + t, X, color=color_map(color_index % 10), **sample_path_style)
            # plot the marker for a coalescent event
            if node.children:
                plt.scatter(node.time, node.latent_value, edgecolor=color_map(color_index % 10), **coalescent_event_style)

        
        if node.children:
            for child in node.children:
                plot_sample_path(child)
        else:
            # plot the marker for a leaf
            plt.scatter(node.time, node.latent_value, edgecolor=color_map(color_index % 10), **leaves_style)

        color_index += 1
    # Plot sample paths
    plot_sample_path(root)
    
    # plot root marker
    
    plt.scatter(root.time, root.latent_value, **root_style)

    # # Plot leaf nodes
    # leaf_times = [leaf.time for leaf in leaves]
    # leaf_values = [leaf.latent_value for leaf in leaves]
    # plt.scatter(leaf_times, leaf_values, **leaves_style)

    # Plot coalescent events
    # event_times = [t for t, _ in events]
    # event_values = [node.latent_value for _, node in events]
    # plt.scatter(event_times, event_values, **coalescent_event_style)

    plt.xlabel("Time")
    plt.ylabel("Latent Value")
    plt.title("1D Brownian Motion Coalescent Process")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_2d_coalescent(leaves, observed_data):
    plt.figure(figsize=(8, 8))
    x, y = zip(*observed_data)
    plt.scatter(x, y)

    for i, leaf in enumerate(leaves):
        plt.annotate(str(i), (x[i], y[i]), xytext=(5, 5), textcoords="offset points")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Observed Data from Coalescent Process")
    plt.grid(True)
    plt.show()


# Example usage
n = 10
leaves, events, root = kingman_coalescent(n)


# Define transition and observation functions for 1D
def transition_func_1d(parent_value, parent_time, child_time):
    delta_t = abs(child_time - parent_time)
    return parent_value + np.random.normal(0, np.sqrt(delta_t))


def leaf_observation_func_1d(latent_value):
    return latent_value + np.random.normal(0, 0.1)


# Run the forward-time process for 1D
initial_value_1d = np.random.normal(0, 1)
forward_time_process(
    root, transition_func_1d, leaf_observation_func_1d, initial_value_1d
)

# Extract observed data for 1D
observed_data_1d = [leaf.latent_value for leaf in leaves]

# Print results for 1D
print("1D Coalescent Results:")
print(f"Observed data: {observed_data_1d}")
print(f"Number of coalescent events: {len(events)}")
for t, node in events:
    print(f"Time {t:.4f}: Merged {node.children[0].label} and {node.children[1].label}")

# Plot 1D coalescent
plot_1d_coalescent(leaves, events, root)


# %%
# Define transition and observation functions for 2D
def transition_func_2d(parent_value, parent_time, child_time):
    delta = np.sqrt(max(child_time - parent_time, 0))  # Ensure non-negative
    return np.array(
        [
            parent_value[0] + np.random.normal(0, delta),
            parent_value[1] + np.random.normal(0, delta),
        ]
    )


def leaf_observation_func_2d(latent_value):
    return np.array(
        [
            latent_value[0] + np.random.normal(0, 0.1),
            latent_value[1] + np.random.normal(0, 0.1),
        ]
    )


# Run the forward-time process for 2D
initial_value_2d = np.random.multivariate_normal([0, 0], np.eye(2))
forward_time_process(
    root, transition_func_2d, leaf_observation_func_2d, initial_value_2d
)

# Extract observed data for 2D
observed_data_2d = [leaf.latent_value for leaf in leaves]

# Print results for 2D
print("\n2D Coalescent Results:")
print(f"Observed data: {observed_data_2d}")

# Plot 2D coalescent
plot_2d_coalescent(leaves, observed_data_2d)


# Example usage
n = 5
leaves, events, root = kingman_coalescent(n)

# 1D Coalescent
initial_value_1d = np.random.normal(0, 1)
forward_time_process(
    root, transition_func_1d, leaf_observation_func_1d, initial_value_1d
)
plot_1d_coalescent(leaves, events, root)


# %%
