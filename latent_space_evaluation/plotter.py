import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Union

class EmbeddingPlotter:
    """
    A class for visualizing high-dimensional embeddings after dimensionality reduction.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        palette: str = "tab10",
        fraction: float = 1.0,
        figsize: tuple = (10, 8),
        point_size: float = 50,
        alpha: float = 0.7,
        random_state: int = 42
    ):
        """
        Initializes the EmbeddingPlotter with configurable parameters.

        Parameters:
            output_dir (str or Path): Directory to save visualization images.
            palette (str): Color palette for different classes (default: 'tab10').
            fraction (float): Fraction of embeddings to sample for visualization (0 < fraction <= 1.0).
            figsize (tuple): Size of the plot figure (default: (10, 8)).
            point_size (float): Size of the points in the scatter plot.
            alpha (float): Transparency of the points (0: fully transparent, 1: fully opaque).
            random_state (int): Seed for reproducibility of sampling.
        """
        self.output_dir = Path(output_dir)
        self.palette = palette
        self.fraction = fraction
        self.figsize = figsize
        self.point_size = point_size
        self.alpha = alpha
        self.random_state = random_state

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: str = "Embedding Visualization",
        file_name: str = "embedding_plot.png"
    ):
        """
        Creates and saves a scatter plot of the embeddings.

        Parameters:
            embeddings (np.ndarray): 2D embeddings to plot (shape: [n_samples, 2]).
            labels (np.ndarray): Class labels corresponding to each embedding (shape: [n_samples]).
            title (str): Title of the plot (default: "Embedding Visualization").
            file_name (str): Name of the file to save the plot (default: "embedding_plot.png").
        """
        # Validate inputs
        if embeddings.shape[1] != 2:
            raise ValueError("Embeddings must be 2-dimensional for visualization.")
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError("Embeddings and labels must have the same number of samples.")

        # Sample a fraction of data
        sampled_embeddings, sampled_labels = self._sample_data(embeddings, labels)

        plt.figure(figsize=self.figsize)
        sns.scatterplot(
            x=sampled_embeddings[:, 0],
            y=sampled_embeddings[:, 1],
            hue=sampled_labels,
            palette=self.palette,
            s=self.point_size,
            alpha=self.alpha,
            edgecolor="k",
            linewidth=0.5,
            legend="full"
        )
        plt.title(title, fontsize=16)
        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)
        plt.legend(title="Classes", loc="best", fontsize=10, title_fontsize=12)
        plt.tight_layout()

        save_path = self.output_dir / file_name
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Plot saved to {save_path}")

    def _sample_data(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Samples a fraction of the embeddings and labels for visualization.
        """
        if self.fraction >= 1.0:
            return embeddings, labels

        np.random.seed(self.random_state)
        unique_labels = np.unique(labels)
        sampled_indices = []

        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            sample_size = max(1, int(len(label_indices) * self.fraction))
            sampled = np.random.choice(label_indices, size=sample_size, replace=False)
            sampled_indices.extend(sampled)

        sampled_indices = np.array(sampled_indices)
        return embeddings[sampled_indices], labels[sampled_indices]