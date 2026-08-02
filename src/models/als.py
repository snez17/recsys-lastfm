import pickle
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

class ALSRecommender:
    def __init__(self, factors=50, iterations=15, regularization=0.01):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.model = None
        self.matrix = None 

    def fit(self, data):
        """data: pd.DataFrame с колонками user_index, track_index, count"""
        self.matrix = csr_matrix(
            (data['count'].values.astype(np.float64),
             (data['user_index'].values, data['track_index'].values))
        )
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization
        )
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
                'factors': self.factors,
                'iterations': self.iterations,
                'regularization': self.regularization,
                'model': self.model,
                'matrix': self.matrix
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(
            factors=data['factors'],
            iterations=data['iterations'],
            regularization=data['regularization']
        )
        obj.model = data['model']
        obj.matrix = data['matrix']
        return obj