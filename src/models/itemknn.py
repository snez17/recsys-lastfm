import pickle
import numpy as np
from scipy.sparse import csr_matrix
from implicit.nearest_neighbours import ItemItemRecommender

class ItemKNNRecommender:
    def __init__(self, K=50):
        self.K = K
        self.model = None
        self.matrix = None

    def fit(self, data):
        self.matrix = csr_matrix(
            (data['count'].values.astype(np.float64),
             (data['user_index'].values, data['track_index'].values))
        )
        self.model = ItemItemRecommender(K=self.K)
        self.model.fit(self.matrix)

    def recommend(self, user_id, n=10):
        if self.model is None or self.matrix is None:
            return np.array([])
        
        user_idx = int(str(user_id).replace('user_', ''))
        if user_idx >= self.matrix.shape[0]:
            return np.array([])
        
        # Передаём вектор пользователя
        user_vector = self.matrix[user_idx]
        ids, _ = self.model.recommend(user_idx, user_vector, N=n)
        return ids


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'K': self.K,
                'model': self.model,
                'matrix': self.matrix
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(K=data['K'])
        obj.model = data['model']
        obj.matrix = data['matrix']
        return obj