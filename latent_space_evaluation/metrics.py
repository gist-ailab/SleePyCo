from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)
from sklearn.manifold import trustworthiness
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy


class BaseMetric(ABC):
    """
    Abstract base class for metrics.
    """

    @abstractmethod
    def compute(
        self,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
        original_embeddings: Optional[np.ndarray] = None,
        predicted_labels: Optional[np.ndarray] = None
    ) -> float:
        """
        Computes the metric.

        Parameters:
            embeddings_2d (np.ndarray): 2D embeddings (shape: [n_samples, 2]).
            labels (np.ndarray): True class labels (shape: [n_samples]).
            n_clusters (int): Number of expected clusters.
            original_embeddings (np.ndarray, optional): High-dimensional embeddings,
                required by some metrics like trustworthiness.
            predicted_labels (np.ndarray, optional): Precomputed cluster labels from KMeans.
                If None and needed, this metric will compute them.

        Returns:
            float: Computed metric value.
        """
        pass

    @staticmethod
    def cluster_data(embeddings_2d: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Runs K-Means to get cluster labels.
        This can be called by metrics that need cluster-based evaluations.
        """
        return KMeans(n_clusters=n_clusters, n_init='auto', random_state=42).fit_predict(embeddings_2d)


class SilhouetteScoreMetric(BaseMetric):
    """
    Silhouette Score:
    Measures how similar an object is to its own cluster compared to other clusters.
    Range: -1 to 1, higher is better.
    """

    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        if predicted_labels is None:
            predicted_labels = self.cluster_data(embeddings_2d, n_clusters)
        return silhouette_score(embeddings_2d, predicted_labels)


class DaviesBouldinMetric(BaseMetric):
    """
    Davies-Bouldin Index:
    Measures the average similarity between clusters,
    lower is better.
    """

    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        if predicted_labels is None:
            predicted_labels = self.cluster_data(embeddings_2d, n_clusters)
        return davies_bouldin_score(embeddings_2d, predicted_labels)


class AdjustedRandIndexMetric(BaseMetric):
    """
    Adjusted Rand Index (ARI):
    Measures similarity between predicted clusters and true labels.
    Range: -1 to 1, higher is better.
    """

    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        if predicted_labels is None:
            predicted_labels = self.cluster_data(embeddings_2d, n_clusters)
        return adjusted_rand_score(labels, predicted_labels)


class AdjustedMutualInfoMetric(BaseMetric):
    """
    Adjusted Mutual Information (AMI):
    Measures mutual information between predicted clusters and true labels,
    adjusted for chance. Ranges from 0 to 1, higher is better.
    """

    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        if predicted_labels is None:
            predicted_labels = self.cluster_data(embeddings_2d, n_clusters)
        return adjusted_mutual_info_score(labels, predicted_labels)


class PurityMetric(BaseMetric):
    """
    Purity Score:
    Assigns each cluster the most frequent true label, and computes
    the overall proportion of correctly assigned points.
    Higher is better, max 1.0.
    """

    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        if predicted_labels is None:
            predicted_labels = self.cluster_data(embeddings_2d, n_clusters)
        return self._purity_score(labels, predicted_labels)

    @staticmethod
    def _purity_score(true_labels, predicted_labels):
        unique_true = np.unique(true_labels)
        unique_pred = np.unique(predicted_labels)
        contingency = np.zeros((len(unique_true), len(unique_pred)))

        for i, t_label in enumerate(unique_true):
            for j, p_label in enumerate(unique_pred):
                contingency[i, j] = np.sum((true_labels == t_label) & (predicted_labels == p_label))

        row_ind, col_ind = linear_sum_assignment(-contingency)
        purity = contingency[row_ind, col_ind].sum() / np.sum(contingency)
        return purity


class AverageEntropyMetric(BaseMetric):
    """
    Average Entropy:
    Measures how pure clusters are in terms of label distribution.
    Lower entropy means more pure clusters.
    """

    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        if predicted_labels is None:
            predicted_labels = self.cluster_data(embeddings_2d, n_clusters)
        return self._compute_average_entropy(labels, predicted_labels)

    @staticmethod
    def _compute_average_entropy(true_labels, predicted_labels):
        cluster_labels = defaultdict(list)
        for t_label, p_label in zip(true_labels, predicted_labels):
            cluster_labels[p_label].append(t_label)

        total_entropy = 0
        total_samples = len(true_labels)
        for labels_in_cluster in cluster_labels.values():
            label_counts = np.bincount(labels_in_cluster)
            probabilities = label_counts / len(labels_in_cluster)
            probabilities = probabilities[probabilities > 0]
            cluster_entropy = entropy(probabilities, base=2)
            total_entropy += cluster_entropy * len(labels_in_cluster)
        average_entropy = total_entropy / total_samples
        return average_entropy


class TrustworthinessMetric(BaseMetric):
    """
    Trustworthiness:
    Requires original_embeddings.
    Score close to 1 is better.
    """

    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        if original_embeddings is None:
            raise ValueError("original_embeddings must be provided for trustworthiness.")
        return trustworthiness(original_embeddings, embeddings_2d, n_neighbors=5)


# Below are examples of metrics that do not require clustering.
# They rely only on the embeddings and/or the ground truth labels.

class IntraClassCompactnessMetric(BaseMetric):
    """
    Intra-Class Compactness:
    Measures how close points of the same class are to each other.
    Lower values indicate more compactness.
    """

    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        # Example: Compute average intra-class distance
        unique_labels = np.unique(labels)
        intra_distances = []
        for lbl in unique_labels:
            class_points = embeddings_2d[labels == lbl]
            if len(class_points) > 1:
                dists = np.sum((class_points[:, None] - class_points[None, :])**2, axis=-1)
                # average distance in this class
                mean_dist = np.mean(dists[dists > 0])
                intra_distances.append(mean_dist)
            else:
                # Only one point in this class, no intra-distance
                intra_distances.append(0.0)
        return np.mean(intra_distances)


class InterClassSeparationMetric(BaseMetric):
    """
    Inter-Class Separation:
    Measures how far points of different classes are from each other.
    Higher values indicate better separation.
    """

    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        # Example: compute average distance between class centroids
        unique_labels = np.unique(labels)
        centroids = []
        for lbl in unique_labels:
            class_points = embeddings_2d[labels == lbl]
            centroid = np.mean(class_points, axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)

        # Compute mean distance between all pairs of centroids
        dists = np.sum((centroids[:, None] - centroids[None, :])**2, axis=-1)
        # Exclude diagonal (distance to itself)
        dists = dists[dists > 0]
        return np.mean(dists)


class CompactnessToSeparationRatio(BaseMetric):
    """
    Compactness-to-Separation Ratio:
    Ratio of intra-class compactness to inter-class separation.
    Lower is better (indicates more compact classes relative to how separated they are).
    """

    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        intra_metric = IntraClassCompactnessMetric().compute(embeddings_2d, labels, n_clusters, original_embeddings)
        inter_metric = InterClassSeparationMetric().compute(embeddings_2d, labels, n_clusters, original_embeddings)
        if inter_metric == 0:
            return np.inf  # Avoid division by zero
        return intra_metric / inter_metric


class MutualInformationMetric(BaseMetric):
    """
    Mutual Information:
    Measures the dependency between discretized embeddings and true labels.
    """
    
    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        from sklearn.metrics import mutual_info_score
        # Discretize embeddings per sample
        discretized_embeddings = self._discretize_embeddings(embeddings_2d)
        # Compute mutual information
        mi = mutual_info_score(labels, discretized_embeddings)
        return mi

    @staticmethod
    def _discretize_embeddings(embeddings, bins=10):
        # Discretize each dimension and combine them into a single label per sample
        discretized = np.floor((embeddings - embeddings.min(axis=0)) / (embeddings.ptp(axis=0) + 1e-8) * bins).astype(int)
        combined = discretized[:, 0] * bins + discretized[:, 1]
        return combined

class UniformityMetric(BaseMetric):
    """
    Uniformity:
    Measures how uniformly the embeddings are distributed using pairwise distances.
    Lower values indicate more uniform distribution.
    """

    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        import torch
        embeddings = torch.from_numpy(embeddings_2d)
        # Compute pairwise squared distances
        sq_pdist = torch.pdist(embeddings, p=2).pow(2)
        # Compute uniformity metric
        uniformity = torch.log(torch.exp(-2 * sq_pdist).mean() + 1e-8)
        return uniformity.item()

class AlignmentMetric(BaseMetric):
    """
    Alignment:
    Measures how well the embeddings preserve pairwise distances from the original space.
    Requires original_embeddings.
    """

    def compute(self, embeddings_2d, labels, n_clusters, original_embeddings=None, predicted_labels=None):
        if original_embeddings is None:
            raise ValueError("original_embeddings must be provided for alignment.")
        import torch
        x = torch.from_numpy(original_embeddings)
        y = torch.from_numpy(embeddings_2d)
        # Compute pairwise distances
        sq_pdist_x = torch.pdist(x, p=2).pow(2)
        sq_pdist_y = torch.pdist(y, p=2).pow(2)
        # Compute alignment metric
        alignment = (sq_pdist_x * sq_pdist_y).sum() / (sq_pdist_x.norm() * sq_pdist_y.norm())
        return alignment.item()

# Example usage (pseudocode):
# Suppose we have a set of metrics we want to compute:
# metrics = [SilhouetteScoreMetric(), DaviesBouldinMetric(), AdjustedRandIndexMetric(), PurityMetric()]
# embeddings_2d, labels, n_clusters = ... # obtained from pipeline
#
# # Run KMeans once:
# predicted_labels = BaseMetric.cluster_data(embeddings_2d, n_clusters)
# # Now compute each metric using the precomputed predicted_labels:
# results = {}
# for m in metrics:
#     results[m.__class__.__name__] = m.compute(embeddings_2d, labels, n_clusters, predicted_labels=predicted_labels)