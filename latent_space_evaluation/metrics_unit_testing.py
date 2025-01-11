import unittest
import numpy as np
from metrics import (
    SilhouetteScoreMetric,
    DaviesBouldinMetric,
    AdjustedRandIndexMetric,
    AdjustedMutualInfoMetric,
    PurityMetric,
    AverageEntropyMetric,
    TrustworthinessMetric,
    IntraClassCompactnessMetric,
    InterClassSeparationMetric,
    CompactnessToSeparationRatio,
    MutualInformationMetric,
    UniformityMetric,
    AlignmentMetric,
    BaseMetric
)

class TestMetrics(unittest.TestCase):

    def setUp(self):
        # High-dimensional dataset
        self.embeddings_2d = np.random.rand(100, 2)
        self.labels = np.random.randint(0, 5, 100)
        self.n_clusters = 5
        self.original_embeddings = np.random.rand(100, 50)  # High-dimensional data

        # Run KMeans once to get predicted labels
        self.predicted_labels = BaseMetric.cluster_data(self.embeddings_2d, self.n_clusters)

    def test_silhouette_score(self):
        metric = SilhouetteScoreMetric()
        score = metric.compute(self.embeddings_2d, self.labels, self.n_clusters, predicted_labels=self.predicted_labels)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)

    def test_davies_bouldin_score(self):
        metric = DaviesBouldinMetric()
        score = metric.compute(self.embeddings_2d, self.labels, self.n_clusters, predicted_labels=self.predicted_labels)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_adjusted_rand_index(self):
        metric = AdjustedRandIndexMetric()
        score = metric.compute(self.embeddings_2d, self.labels, self.n_clusters, predicted_labels=self.predicted_labels)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)

    def test_adjusted_mutual_info(self):
        metric = AdjustedMutualInfoMetric()
        score = metric.compute(self.embeddings_2d, self.labels, self.n_clusters, predicted_labels=self.predicted_labels)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_purity_score(self):
        metric = PurityMetric()
        score = metric.compute(self.embeddings_2d, self.labels, self.n_clusters, predicted_labels=self.predicted_labels)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_average_entropy(self):
        metric = AverageEntropyMetric()
        score = metric.compute(self.embeddings_2d, self.labels, self.n_clusters, predicted_labels=self.predicted_labels)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_trustworthiness(self):
        metric = TrustworthinessMetric()
        score = metric.compute(
            self.embeddings_2d,
            self.labels,
            self.n_clusters,
            original_embeddings=self.original_embeddings
        )
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_intra_class_compactness(self):
        metric = IntraClassCompactnessMetric()
        score = metric.compute(self.embeddings_2d, self.labels, self.n_clusters)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_inter_class_separation(self):
        metric = InterClassSeparationMetric()
        score = metric.compute(self.embeddings_2d, self.labels, self.n_clusters)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)

    def test_compactness_to_separation_ratio(self):
        metric = CompactnessToSeparationRatio()
        score = metric.compute(
            self.embeddings_2d,
            self.labels,
            self.n_clusters,
            original_embeddings=self.original_embeddings
        )
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_mutual_information(self):
        metric = MutualInformationMetric()
        score = metric.compute(self.embeddings_2d, self.labels, self.n_clusters)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_uniformity(self):
        metric = UniformityMetric()
        score = metric.compute(self.embeddings_2d, self.labels, self.n_clusters)
        self.assertIsInstance(score, float)
        # Uniformity can vary; just check it's a finite number
        self.assertTrue(np.isfinite(score))

    def test_alignment(self):
        metric = AlignmentMetric()
        score = metric.compute(
            self.embeddings_2d,
            self.labels,
            self.n_clusters,
            original_embeddings=self.original_embeddings
        )
        self.assertIsInstance(score, float)
        # Alignment can theoretically range; check it's finite
        self.assertTrue(np.isfinite(score))

    def test_label_cluster_mapping(self):
        """
        Test metrics when cluster labels do not match true labels.
        For example, true labels: [0,0,1,1], cluster labels: [1,1,0,0]
        For example, true labels: [0,0,1,1], cluster labels: [1,1,0,0]
        """
        # Define embeddings (not used in some metrics)
        embeddings_2d = np.random.rand(4, 2)
        labels = np.array([0, 0, 1, 1])
        n_clusters = 2
        original_embeddings = None  # Not required for all metrics
        
        # Define predicted_labels with mismatched cluster labels
        predicted_labels = np.array([1, 1, 0, 0])
        
        # Initialize metrics
        metrics = [
            SilhouetteScoreMetric(),
            DaviesBouldinMetric(),
            AdjustedRandIndexMetric(),
            AdjustedMutualInfoMetric(),
            PurityMetric(),
            AverageEntropyMetric(),
            IntraClassCompactnessMetric(),
            InterClassSeparationMetric(),
            CompactnessToSeparationRatio(),
            MutualInformationMetric(),
            UniformityMetric(),
            AlignmentMetric()
        ]
        
        # Compute and verify metrics
        results = {}
        for metric in metrics:
            if isinstance(metric, (TrustworthinessMetric, AlignmentMetric)):
                # These metrics require original_embeddings
                with self.assertRaises(ValueError):
                    metric.compute(
                        embeddings_2d,
                        labels,
                        n_clusters,
                        original_embeddings=original_embeddings,
                        predicted_labels=predicted_labels
                    )
                continue  # Skip metrics that require original_embeddings
            score = metric.compute(
                embeddings_2d,
                labels,
                n_clusters,
                predicted_labels=predicted_labels
            )
            self.assertIsInstance(score, float)
        
        # Specifically test PurityMetric with known mapping
        purity_metric = PurityMetric()
        purity_score = purity_metric.compute(embeddings_2d, labels, n_clusters, predicted_labels=predicted_labels)
        self.assertEqual(purity_score, 1.0)  # Perfect purity after optimal mapping

if __name__ == '__main__':
    unittest.main()

        
#         # Define embeddings (not used in some metrics)
#         embeddings_2d = np.random.rand(4, 2)
#         labels = np.array([0, 0, 1, 1])
#         n_clusters = 2
#         original_embeddings = None  # Not required for all metrics
        
#         # Define predicted_labels with mismatched cluster labels
#         predicted_labels = np.array([1, 1, 0, 0])
        
#         # Initialize metrics
#         metrics = [
#             SilhouetteScoreMetric(),
#             DaviesBouldinMetric(),
#             AdjustedRandIndexMetric(),
#             AdjustedMutualInfoMetric(),
#             PurityMetric(),
#             AverageEntropyMetric(),
#             IntraClassCompactnessMetric(),
#             InterClassSeparationMetric(),
#             CompactnessToSeparationRatio(),
#             MutualInformationMetric(),
#             UniformityMetric(),
#             AlignmentMetric()
#         ]
        
#         # Compute and verify metrics
#         results = {}
#         for metric in metrics:
#             if isinstance(metric, (TrustworthinessMetric, AlignmentMetric)):
#                 # These metrics require original_embeddings
#                 with self.assertRaises(ValueError):
#                     metric.compute(
#                         embeddings_2d,
#                         labels,
#                         n_clusters,
#                         original_embeddings=original_embeddings,
#                         predicted_labels=predicted_labels
#                     )
#                 continue  # Skip metrics that require original_embeddings
#             score = metric.compute(
#                 embeddings_2d,
#                 labels,
#                 n_clusters,
#                 predicted_labels=predicted_labels
#             )
#             self.assertIsInstance(score, float)
        
#         # Specifically test PurityMetric with known mapping
#         purity_metric = PurityMetric()
#         purity_score = purity_metric.compute(embeddings_2d, labels, n_clusters, predicted_labels=predicted_labels)
#         self.assertEqual(purity_score, 1.0)  # Perfect purity after optimal mapping

# if __name__ == '__main__':
#     unittest.main()
