import unittest
import numpy as np
from reducers import PCAReducer, TSNEReducer, UMAPReducer

class TestReducers(unittest.TestCase):
    def setUp(self):
        self.embeddings = np.random.rand(100, 50)  # Sample data

    def test_pca_reducer(self):
        reducer = PCAReducer(n_components=2)
        reduced = reducer.fit_transform(self.embeddings)
        self.assertEqual(reduced.shape[1], 2)

    def test_tsne_reducer(self):
        reducer = TSNEReducer(n_components=2, n_iter=500)
        reduced = reducer.fit_transform(self.embeddings)
        self.assertEqual(reduced.shape[1], 2)

    def test_umap_reducer(self):
        reducer = UMAPReducer(n_components=2, n_neighbors=10)
        reduced = reducer.fit_transform(self.embeddings)
        self.assertEqual(reduced.shape[1], 2)

if __name__ == "__main__":
    unittest.main()