import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def plot_images(images, n=10):
    fig, axs = plt.subplots(1, n, figsize=(n, 1))
    for i, ax in enumerate(axs):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    plt.show()


# Function to add arrows along a line
def add_arrows(line, segment_index, color=None):
    """Add arrow to a specific line segment."""
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    # Get the start and end points of the specified segment
    start = np.array([xdata[segment_index], ydata[segment_index]])
    end = np.array([xdata[segment_index + 1], ydata[segment_index + 1]])

    # Calculate the midpoint for arrow placement
    mid = (start + end) / 2

    # Calculate the direction vector
    direction = end - start

    # Normalize the direction vector
    direction = direction / np.sqrt(np.sum(direction**2))

    # Create and add the arrowhead
    arrow = FancyArrowPatch(
        posA=mid,  # Start and end at same point for just the head
        posB=mid + direction * 0.001,  # Tiny offset to control direction
        arrowstyle='->',
        color=color,
        mutation_scale=20,
        shrinkA=0,  # Don't shrink the arrowhead
        shrinkB=0
    )
    plt.gca().add_patch(arrow)