import unittest
import numpy as np
from latent_space_evaluation import *
import pandas as pd
from tqdm import tqdm

class TestLatentSpaceEvaluation(unittest.TestCase):
    def setUp(self):
        # Generate random 128-dimensional embeddings
        self.embeddings = np.random.rand(1000, 128)
        self.labels = np.random.randint(0, 10, size=1000)  # Assume 10 classes
        self.n_clusters = 10
        self.output_dir = "./plots"
        self.kmeans_labels = {}

    def test_reducers_and_plotting(self):
        # Initialize reducers with verbose arguments
        pca = PCAReducer(n_components=2, random_state=42)
        tsne = TSNEReducer(
            n_components=2,
            perplexity=30.0,
            learning_rate='auto',
            n_iter=1000,
            random_state=42
        )
        umap = UMAPReducer(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )

        # Fit and transform embeddings
        pca_embeddings = pca.fit_transform(self.embeddings)
        tsne_embeddings = tsne.fit_transform(self.embeddings)
        umap_embeddings = umap.fit_transform(self.embeddings)

        # Initialize plotter with verbose arguments
        plotter = EmbeddingPlotter(
            output_dir=self.output_dir,
            palette="tab10",
            fraction=1.0,
            figsize=(10, 8),
            point_size=50,
            alpha=0.7,
            random_state=42
        )

        # Plot reduced embeddings
        plotter.plot(
            embeddings=pca_embeddings,
            labels=self.labels,
            title="PCA Reduction of Embeddings",
            file_name="pca_plot.png"
        )
        plotter.plot(
            embeddings=tsne_embeddings,
            labels=self.labels,
            title="t-SNE Reduction of Embeddings",
            file_name="tsne_plot.png"
        )
        plotter.plot(
            embeddings=umap_embeddings,
            labels=self.labels,
            title="UMAP Reduction of Embeddings",
            file_name="umap_plot.png"
        )

    def test_metrics(self):
        # Initialize reducers
        reducers = {
            'PCA': PCAReducer(n_components=2, random_state=42),
            't-SNE': TSNEReducer(
                n_components=2,
                perplexity=30.0,
                learning_rate='auto',
                n_iter=1000,
                random_state=42
            ),
            'UMAP': UMAPReducer(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=42
            )
        }

        # Initialize metrics
        metrics = [
            SilhouetteScoreMetric(),
            DaviesBouldinMetric(),
            AdjustedRandIndexMetric(),
            AdjustedMutualInfoMetric(),
            PurityMetric(),
            AverageEntropyMetric(),
            TrustworthinessMetric(),
            IntraClassCompactnessMetric(),
            InterClassSeparationMetric(),
            CompactnessToSeparationRatio(),
            MutualInformationMetric(),
            UniformityMetric(),
            AlignmentMetric()
        ]

        results = {}

        for name, reducer in reducers.items():
            # Fit and transform embeddings
            reduced_embeddings = reducer.fit_transform(self.embeddings)

            # Perform K-Means clustering once
            predicted_labels = BaseMetric.cluster_data(reduced_embeddings, self.n_clusters)
            self.kmeans_labels[name] = predicted_labels

            # Initialize dictionary for this reducer's metrics
            results[name] = {}

            for metric in metrics:
                if isinstance(metric, TrustworthinessMetric) or isinstance(metric, AlignmentMetric):
                    metric_value = metric.compute(
                        embeddings_2d=reduced_embeddings,
                        labels=self.labels,
                        n_clusters=self.n_clusters,
                        original_embeddings=self.embeddings
                    )
                else:
                    metric_value = metric.compute(
                        embeddings_2d=reduced_embeddings,
                        labels=self.labels,
                        n_clusters=self.n_clusters,
                        predicted_labels=self.kmeans_labels[name]
                    )
                results[name][metric.__class__.__name__] = metric_value

        # Print all metric results
        for reducer_name, metric_results in results.items():
            print(f"\nMetrics for {reducer_name}:")
            for metric_name, value in metric_results.items():
                print(f"{metric_name}: {value}")

def evaluate_embeddings(embeddings, labels, output_dir="./plots"):
    print("[INFO] Starting evaluation of embeddings...")
    # Initialize reducers
    reducers = {
        'PCA': PCAReducer(n_components=2, random_state=42),
        't-SNE': TSNEReducer(
            n_components=2,
            perplexity=30.0,
            learning_rate='auto',
            n_iter=1000,
            random_state=42
        ),
        'UMAP': UMAPReducer(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
    }

    # Initialize metrics
    metrics = [
        SilhouetteScoreMetric(),
        DaviesBouldinMetric(),
        AdjustedRandIndexMetric(),
        AdjustedMutualInfoMetric(),
        PurityMetric(),
        AverageEntropyMetric(),
        TrustworthinessMetric(),
        IntraClassCompactnessMetric(),
        InterClassSeparationMetric(),
        CompactnessToSeparationRatio(),
        MutualInformationMetric(),
        UniformityMetric(),
        AlignmentMetric()
    ]

    results = {}
    kmeans_labels = {}

    for name, reducer in tqdm(reducers.items(), desc="[INFO] Applying reducers"):
        # Fit and transform embeddings
        reduced_embeddings = reducer.fit_transform(embeddings)

        # Perform K-Means clustering once
        predicted_labels = BaseMetric.cluster_data(reduced_embeddings, 10)
        kmeans_labels[name] = predicted_labels

        # Initialize dictionary for this reducer's metrics
        results[name] = {}

        for metric in tqdm(metrics, desc=f"[INFO] Computing metrics for {name}", leave=False):
            if isinstance(metric, TrustworthinessMetric) or isinstance(metric, AlignmentMetric):
                metric_value = metric.compute(
                    embeddings_2d=reduced_embeddings,
                    labels=labels,
                    n_clusters=10,
                    original_embeddings=embeddings
                )
            else:
                metric_value = metric.compute(
                    embeddings_2d=reduced_embeddings,
                    labels=labels,
                    n_clusters=10,
                    predicted_labels=kmeans_labels[name]
                )
            results[name][metric.__class__.__name__] = metric_value

    print("[INFO] Creating DataFrame of metric results...")
    df = pd.DataFrame.from_dict({k: v for k, v in results.items()}, orient='index')
    df.to_csv(f"{output_dir}/metrics.csv")
    print("[INFO] Metrics saved to metrics.csv.")

    # Print all metric results
    for reducer_name, metric_results in results.items():
        print(f"\nMetrics for {reducer_name}:")
        for metric_name, value in metric_results.items():
            print(f"{metric_name}: {value}")

    print("[INFO] Starting plotting of embeddings...")
    plotter = EmbeddingPlotter(
        output_dir=output_dir,
        palette="tab10",
        fraction=1.0,
        figsize=(10, 8),
        point_size=50,
        alpha=0.7,
        random_state=42
    )

    for name, reduced_embeddings in zip(reducers.keys(), [reducer.fit_transform(embeddings) for reducer in reducers.values()]):
        print(f"[INFO] Plotting {name} Reduction of Embeddings...")
        plotter.plot(
            embeddings=reduced_embeddings,
            labels=labels,
            title=f"{name} Reduction of Embeddings",
            file_name=f"{name.lower()}_plot.png"
        )

if __name__ == '__main__':
    unittest.main()