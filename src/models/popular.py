import pickle
import numpy as np
import pandas as pd

class PopularRecommender:
    def __init__(self):
        self.popular = np.array([])

    def fit(self, data):
        """data: pd.DataFrame с колонками user_index, track_index, count"""
        self.popular = data.groupby('track_index')['count'].sum().sort_values(ascending=False).index.values

    def recommend(self, user_id=None, n=10):
        return self.popular[:n] if len(self.popular) > 0 else np.array([])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'popular': self.popular}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls()
        obj.popular = data['popular']
        return obj