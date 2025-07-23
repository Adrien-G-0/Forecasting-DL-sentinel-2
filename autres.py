import numpy as np
import matplotlib.pyplot as plt
import weightwatcher as ww
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import torch
import torch.nn.functional as F

def plt_alpha_ww(details, savefig=None):
    '''
    This function visualizes the alpha values from a weight watcher analysis.
    It creates a histogram and a bar plot of alpha values for each layer.
    Problematic layers (overfitted or underfitted) are identified and printed.
    '''
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # One row, two columns
    # Histogram on the left axis
    details.alpha.plot.hist(bins=20, ax=axes[0], title="")
    axes[0].set_xlabel("Weights Watcher Layer Quality Metrics Alpha")
    axes[0].axvline(x=2, color="r")
    axes[0].axvline(x=6, color="orange")
    # Bar graph on the right axis
    layer_names = details.index  # or details.layer_id depending on the version
    alpha_values = details.alpha
    axes[1].bar(range(len(alpha_values)), alpha_values)
    axes[1].set_xlabel("Layers")
    axes[1].set_ylabel("Alpha (Weight Watcher)")
    axes[1].set_title("Layer quality by alpha")
    axes[1].axhline(y=2, color="r", linestyle="--", label="Overfitted (α < 2)")
    axes[1].axhline(y=6, color="orange", linestyle="--", label="Underfitted (α > 6)")
    axes[1].set_xticks(range(len(alpha_values)))
    axes[1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[1].legend()
    plt.tight_layout()
    # Save the graph if a path is provided
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')  # bbox_inches='tight' avoids unwanted cropping
    plt.show()
    # Identify problematic layers
    problematic_layers = details[(details.alpha < 2) | (details.alpha > 6)]
    print("Overfitted layers (α < 2):")
    for idx, row in problematic_layers[problematic_layers.alpha < 2].iterrows():
        print(f"  {idx}: α = {row.alpha:.3f}")
    print("\nUnderfitted layers (α > 6):")
    for idx, row in problematic_layers[problematic_layers.alpha > 6].iterrows():
        print(f"  {idx}: α = {row.alpha:.3f}")

def visualize_weights(model, savefig=None):
    '''
    This function visualizes the weights of a model using WeightWatcher.
    It creates a WeightWatcher instance, analyzes the model, and saves the results.
    '''
    # Create the WeightWatcher
    watcher = ww.WeightWatcher(model=model)

    # Analyze the model
    details = watcher.analyze()

    # Save the results
    plt_alpha_ww(details, savefig=savefig)

def ww_model(model, savefig=None):
    '''
    This function visualizes the alpha values for a model, distinguishing between middle fusion and Unet layers.
    It creates a bar plot of alpha values and identifies problematic layers.
    '''
    alpha_values=[]
    color=[]
    if model.conf['method'] == 'middle_fusion':
        watcher = ww.WeightWatcher(model=model.fusion_en)
        details = watcher.analyze()
        alpha_values.extend(details.alpha.tolist())
        color.extend(['blue'] * len(details.alpha))

    watcher = ww.WeightWatcher(model=model.net)
    details = watcher.analyze()
    alpha_values.extend(details.alpha.tolist())
    color.extend(['green'] * len(details.alpha))
    plt.subplots(figsize=(12, 6))  # One row, two columns
    # Bar graph on the right axis
    plt.bar(range(len(alpha_values)), alpha_values, color=color)
    plt.xlabel("Layers")
    plt.ylabel("Alpha (Weight Watcher)")
    plt.title("Layer quality by alpha")

    plt.xticks(range(len(alpha_values)), rotation=45, ha='right')
    plt.legend(handles=[
        Patch(color='blue', label='Middle Fusion'),
        Patch(color='green', label='Unet'),
        Line2D([0], [0], color="r", linestyle="--", label="over-fitted (α < 2)"),
        Line2D([0], [0], color="orange", linestyle="--", label="Under fitted (α > 6)")
    ])
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    else:
        plt.show()

def calculate_tensor_gradient(image_tensor):
    '''
    This function calculates the gradient magnitude of an image tensor using Sobel filters.
    It normalizes the gradient magnitude for each channel separately.
    '''
    # Keep all channels of the image
    gray_tensor = image_tensor
    # Define Sobel kernels for gradients in x and y
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_x = sobel_x.to(image_tensor.device)  # Move the kernel to the same device as the image
    sobel_x = sobel_x.expand(gray_tensor.size(1), 1, -1, -1)  # Adjust for the number of channels
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = sobel_y.to(image_tensor.device)  # Move the kernel to the same device as the image
    sobel_y = sobel_y.expand(gray_tensor.size(1), 1, -1, -1)  # Adjust for the number of channels
    # Calculate gradients in x and y for each channel
    grad_x = F.conv2d(gray_tensor, sobel_x, padding=1, groups=gray_tensor.size(1))
    grad_y = F.conv2d(gray_tensor, sobel_y, padding=1, groups=gray_tensor.size(1))
    # Calculate the gradient magnitude for each channel
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    # Normalize the gradient magnitude for each channel
    # We will normalize each channel separately
    min_vals = grad_magnitude.min()
    max_vals = grad_magnitude.max()
    # Avoid division by zero
    epsilon = 1e-8
    grad_magnitude_normalized = (grad_magnitude - min_vals) / (max_vals-min_vals + epsilon)
    return grad_magnitude_normalized
