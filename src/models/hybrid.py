import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from models.popular import PopularRecommender
from models.usertop import UserTopRecommender
from collections import Counter


class HybridRecommender:
    """
    Гибридный рекомендатель:
    - UserKNN: если есть похожие пользователи (кеширует матрицу сходства)
    - UserTop: если пользователь переслушивает треки
    - Popular: иначе
    """

    def __init__(self, threshold=0.1, max_neighbors=3, repeated_ratio=0.3):
        self.threshold = threshold
        self.max_neighbors = max_neighbors
        self.repeated_ratio = repeated_ratio
        self.user_artist_matrix = None
        self.user_ids = None
        self.df_train = None
        self.user_id_col = None
        self.similarity_matrix = None  # ← кеш

    def fit(self, data):
        """data: pd.DataFrame с колонками user_index, track_index, count"""
        self.df_train = data

        if 'user_id' in data.columns:
            self.user_id_col = 'user_id'
            user_artist = data.groupby(['user_id', 'track_index'])['count'].sum().unstack(fill_value=0)
            self.user_ids = user_artist.index.tolist()
            self.user_artist_matrix = user_artist.values
        elif 'user_index' in data.columns:
            self.user_id_col = 'user_index'
            self.user_ids = data['user_index'].unique().tolist()
            user_artist = data.groupby(['user_index', 'track_index'])['count'].sum().unstack(fill_value=0)
            self.user_artist_matrix = user_artist.values
        else:
            self.user_ids = None
            self.user_artist_matrix = None

        # ← КЕШ: считаем матрицу сходства ОДИН РАЗ
        if self.user_artist_matrix is not None and len(self.user_artist_matrix) > 0:
            self.similarity_matrix = cosine_similarity(self.user_artist_matrix)
        else:
            self.similarity_matrix = None

    def recommend(self, user_id, n=10):
        """Гибридная рекомендация."""
        # 1. Пробуем UserKNN
        neighbors = self._find_neighbors(user_id)
        if neighbors:
            return self._get_neighbor_tracks(neighbors, n)

        # 2. Пробуем UserTop
        if self._is_repeated_user(user_id):
            usertop = UserTopRecommender(top_n=n)
            usertop.fit(self.df_train)
            return usertop.recommend(user_id, n=n)

        # 3. Иначе Popular
        popular = PopularRecommender()
        popular.fit(self.df_train)
        return popular.recommend(user_id, n=n)

    def _find_neighbors(self, user_id):
        """Находит похожих пользователей (из кеша)."""
        if self.similarity_matrix is None or self.user_ids is None:
            return []

        # Преобразуем user_id в индекс
        try:
            if isinstance(user_id, str):
                user_idx = int(user_id.replace('user_', ''))
            else:
                user_idx = int(user_id)
        except:
            return []

        try:
            idx = self.user_ids.index(user_idx) if user_idx in self.user_ids else -1
            if idx == -1:
                return []
        except:
            return []

        # ← БЕРЁМ ГОТОВУЮ СТРОКУ ИЗ КЕША
        similarities = self.similarity_matrix[idx]

        neighbors = []
        for i, sim in enumerate(similarities):
            if i == idx:
                continue
            if sim > self.threshold:
                neighbors.append((self.user_ids[i], sim))
                if len(neighbors) >= self.max_neighbors:
                    break
        return neighbors

    def _get_neighbor_tracks(self, neighbors, n=10):
        """Собирает треки соседей."""
        if self.df_train is None:
            return []

        all_tracks = []
        user_col = self.user_id_col or 'user_index'

        for neighbor_id, _ in neighbors:
            tracks = self.df_train[self.df_train[user_col] == neighbor_id]['track_index'].values
            all_tracks.extend(tracks)

        counter = Counter(all_tracks)
        return [track for track, _ in counter.most_common(n)]

    def _is_repeated_user(self, user_id):
        """Проверяет, переслушивает ли пользователь треки."""
        if self.df_train is None:
            return False

        user_col = self.user_id_col or 'user_index'

        if isinstance(user_id, str):
            try:
                user_idx = int(user_id.replace('user_', ''))
            except:
                return False
        else:
            user_idx = int(user_id)

        user_tracks = self.df_train[self.df_train[user_col] == user_idx]

        if len(user_tracks) == 0:
            return False

        repeated = user_tracks.groupby('track_index').size()
        repeated_ratio = (repeated > 1).sum() / len(repeated) if len(repeated) > 0 else 0
        return repeated_ratio > self.repeated_ratio

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'threshold': self.threshold,
                'max_neighbors': self.max_neighbors,
                'repeated_ratio': self.repeated_ratio,
                'user_ids': self.user_ids,
                'user_artist_matrix': self.user_artist_matrix,
                'df_train': self.df_train,
                'user_id_col': self.user_id_col,
                'similarity_matrix': self.similarity_matrix
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(
            threshold=data['threshold'],
            max_neighbors=data['max_neighbors'],
            repeated_ratio=data['repeated_ratio']
        )
        obj.user_ids = data['user_ids']
        obj.user_artist_matrix = data['user_artist_matrix']
        obj.df_train = data['df_train']
        obj.user_id_col = data.get('user_id_col', 'user_index')
        obj.similarity_matrix = data.get('similarity_matrix', None)
        return obj