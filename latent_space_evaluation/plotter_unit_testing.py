import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from reducers import PCAReducer, TSNEReducer, UMAPReducer  # Ensure reducers are accessible
from plotter import EmbeddingPlotter  # Ensure plotter.py is accessible
import os

class TestEmbeddingPlotter(unittest.TestCase):
    def setUp(self):
        """
        Set up a temporary directory and synthetic data for testing.
        """
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Generate synthetic high-dimensional data
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 50
        self.embeddings = np.random.rand(self.n_samples, self.n_features)

        # Generate synthetic labels (e.g., 5 classes)
        self.n_classes = 5
        self.labels = np.repeat(np.arange(self.n_classes), self.n_samples // self.n_classes)
        # If n_samples not divisible by n_classes, append remaining labels
        remaining = self.n_samples % self.n_classes
        if remaining > 0:
            self.labels = np.concatenate([self.labels, np.arange(remaining)])

    def tearDown(self):
        """
        Remove the temporary directory after tests.
        """
        shutil.rmtree(self.test_dir)

    def test_pca_plot(self):
        """
        Test PCA reduction followed by plotting.
        """
        # Initialize PCA Reducer
        pca_reducer = PCAReducer(n_components=2, random_state=42)
        reduced_embeddings = pca_reducer.fit_transform(self.embeddings)

        # Initialize EmbeddingPlotter
        plotter = EmbeddingPlotter(
            output_dir=self.test_dir,
            palette="tab10",
            fraction=1.0,  # Use all data
            figsize=(10, 8),
            point_size=50,
            alpha=0.7,
            random_state=42
        )

        # Plot
        plotter.plot(
            embeddings=reduced_embeddings,
            labels=self.labels,
            title="PCA Embedding Visualization",
            file_name="pca_plot.png"
        )

        # Check if plot file exists
        plot_path = Path(self.test_dir) / "pca_plot.png"
        self.assertTrue(plot_path.exists(), "PCA plot file was not created.")

        # Check if the plot file is not empty
        self.assertGreater(os.path.getsize(plot_path), 0, "PCA plot file is empty.")

    def test_tsne_plot_fraction(self):
        """
        Test t-SNE reduction with sampling fraction followed by plotting.
        """
        # Initialize t-SNE Reducer
        tsne_reducer = TSNEReducer(n_components=2, perplexity=30, random_state=42, n_iter=500)
        reduced_embeddings = tsne_reducer.fit_transform(self.embeddings)

        # Initialize EmbeddingPlotter with fraction=0.5
        plotter = EmbeddingPlotter(
            output_dir=self.test_dir,
            palette="tab10",
            fraction=0.5,  # Sample 50% of data
            figsize=(10, 8),
            point_size=30,
            alpha=0.6,
            random_state=42
        )

        # Plot
        plotter.plot(
            embeddings=reduced_embeddings,
            labels=self.labels,
            title="t-SNE Embedding Visualization",
            file_name="tsne_plot.png"
        )

        # Check if plot file exists
        plot_path = Path(self.test_dir) / "tsne_plot.png"
        self.assertTrue(plot_path.exists(), "t-SNE plot file was not created.")

        # Check if the plot file is not empty
        self.assertGreater(os.path.getsize(plot_path), 0, "t-SNE plot file is empty.")

    def test_umap_plot_edge_cases(self):
        """
        Test UMAP reduction with edge cases followed by plotting.
        """
        # Initialize UMAP Reducer
        umap_reducer = UMAPReducer(n_components=2, n_neighbors=10, min_dist=0.1, random_state=42)
        reduced_embeddings = umap_reducer.fit_transform(self.embeddings)

        # Initialize EmbeddingPlotter with fraction=0.1
        plotter = EmbeddingPlotter(
            output_dir=self.test_dir,
            palette="viridis",
            fraction=0.1,  # Sample 10% of data
            figsize=(12, 10),
            point_size=20,
            alpha=0.8,
            random_state=42
        )

        # Plot
        plotter.plot(
            embeddings=reduced_embeddings,
            labels=self.labels,
            title="UMAP Embedding Visualization",
            file_name="umap_plot.png"
        )

        # Check if plot file exists
        plot_path = Path(self.test_dir) / "umap_plot.png"
        self.assertTrue(plot_path.exists(), "UMAP plot file was not created.")

        # Check if the plot file is not empty
        self.assertGreater(os.path.getsize(plot_path), 0, "UMAP plot file is empty.")

    def test_invalid_embeddings_dimension(self):
        """
        Test plotting with invalid embedding dimensions (not 2D).
        """
        # Create invalid embeddings (3D)
        invalid_embeddings = np.random.rand(self.n_samples, 3)

        # Initialize EmbeddingPlotter
        plotter = EmbeddingPlotter(
            output_dir=self.test_dir
        )

        # Attempt to plot and expect a ValueError
        with self.assertRaises(ValueError):
            plotter.plot(
                embeddings=invalid_embeddings,
                labels=self.labels,
                title="Invalid Embedding Visualization",
                file_name="invalid_plot.png"
            )

    def test_mismatched_embeddings_labels(self):
        """
        Test plotting with mismatched number of embeddings and labels.
        """
        # Create mismatched labels
        mismatched_labels = self.labels[:-1]  # One less label

        # Initialize PCA Reducer and reduce to 2D
        pca_reducer = PCAReducer(n_components=2, random_state=42)
        reduced_embeddings = pca_reducer.fit_transform(self.embeddings)

        # Initialize EmbeddingPlotter
        plotter = EmbeddingPlotter(
            output_dir=self.test_dir
        )

        # Attempt to plot and expect a ValueError
        with self.assertRaises(ValueError):
            plotter.plot(
                embeddings=reduced_embeddings,
                labels=mismatched_labels,
                title="Mismatched Embeddings and Labels",
                file_name="mismatch_plot.png"
            )

    def test_plot_fraction_zero(self):
        """
        Test plotting with fraction=0, which should sample at least one point per class.
        """
        # Initialize PCA Reducer
        pca_reducer = PCAReducer(n_components=2, random_state=42)
        reduced_embeddings = pca_reducer.fit_transform(self.embeddings)

        # Initialize EmbeddingPlotter with fraction=0 (invalid, should sample at least one per class)
        plotter = EmbeddingPlotter(
            output_dir=self.test_dir,
            fraction=0.0,  # Edge case
            random_state=42
        )

        # Attempt to plot and expect at least one sample per class
        plotter.plot(
            embeddings=reduced_embeddings,
            labels=self.labels,
            title="PCA Embedding Visualization with Fraction=0",
            file_name="pca_fraction_zero_plot.png"
        )

        # Check if plot file exists
        plot_path = Path(self.test_dir) / "pca_fraction_zero_plot.png"
        self.assertTrue(plot_path.exists(), "PCA fraction=0 plot file was not created.")

        # Check if the plot file is not empty
        self.assertGreater(os.path.getsize(plot_path), 0, "PCA fraction=0 plot file is empty.")

    def test_invalid_fraction(self):
        """
        Test initializing EmbeddingPlotter with an invalid fraction value.
        """
        with self.assertRaises(ValueError):
            EmbeddingPlotter(
                output_dir=self.test_dir,
                fraction=1.5  # Invalid fraction >1
            )

    def test_plot_with_all_data_fraction(self):
        """
        Test plotting with fraction=1.0, ensuring all data is plotted.
        """
        # Initialize UMAP Reducer
        umap_reducer = UMAPReducer(n_components=2, random_state=42)
        reduced_embeddings = umap_reducer.fit_transform(self.embeddings)

        # Initialize EmbeddingPlotter with fraction=1.0
        plotter = EmbeddingPlotter(
            output_dir=self.test_dir,
            fraction=1.0,
            random_state=42
        )

        # Plot
        plotter.plot(
            embeddings=reduced_embeddings,
            labels=self.labels,
            title="UMAP Embedding Visualization with Full Data",
            file_name="umap_full_plot.png"
        )

        # Check if plot file exists
        plot_path = Path(self.test_dir) / "umap_full_plot.png"
        self.assertTrue(plot_path.exists(), "UMAP full data plot file was not created.")

        # Check if the plot file is not empty
        self.assertGreater(os.path.getsize(plot_path), 0, "UMAP full data plot file is empty.")

if __name__ == '__main__':
    unittest.main()