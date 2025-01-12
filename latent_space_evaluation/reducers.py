from abc import ABC, abstractmethod
import numpy as np
from sklearn.manifold import TSNE
import umap.umap_ as umap
import warnings
from sklearn.decomposition import PCA

class BaseReducer(ABC):
    
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        pass

class TSNEReducer(BaseReducer):
    """
    t-SNE Reducer for dimensionality reduction.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 'auto',
        n_iter: int = 1000,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            random_state=self.random_state,
            init='random',
            verbose=0
        )
        
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        reduced_embeddings = self.tsne.fit_transform(embeddings)
        return reduced_embeddings


class UMAPReducer(BaseReducer):

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        random_state: int = 42,
    ):

        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.umap_reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            init='random',
            verbose=False
        )

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="umap")
            reduced_embeddings = self.umap_reducer.fit_transform(embeddings)
        return reduced_embeddings
    

class PCAReducer(BaseReducer):
    """
    PCA Reducer for dimensionality reduction.
    """

    def __init__(self, n_components: int = 2, random_state: int = 42):

        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        reduced_embeddings = self.pca.fit_transform(embeddings)
        return reduced_embeddings