import pickle
import numpy as np
import pandas as pd

class UserTopRecommender:
    def __init__(self, top_n=10):
        self.top_n = top_n
        self.user_tracks = {}

    def fit(self, data):
        sorted_data = data.sort_values(['user_index', 'count'], ascending=[True, False])
        self.user_tracks = (
            sorted_data.groupby('user_index')
            .head(self.top_n)
            .groupby('user_index')['track_index']
            .apply(list)
            .to_dict()
        )

    def recommend(self, user_id, n=10):
        user_idx = int(str(user_id).replace('user_', ''))
        tracks = self.user_tracks.get(user_idx, [])
        return np.array(tracks[:n])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'user_tracks': self.user_tracks, 'top_n': self.top_n}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(top_n=data['top_n'])
        model.user_tracks = data['user_tracks']
        return model