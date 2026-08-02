import pickle
from .popular import PopularRecommender
from .kmeans import KMeansRecommender


class HybridRecommender:
    """
    Гибридная модель: смесь Popular (глобальные тренды) и KMeans (персонализация).
    
    Параметры:
        alpha: float — доля популярного (0.0 = только KMeans, 1.0 = только Popular)
    """
    
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.popular = PopularRecommender()
        self.kmeans = KMeansRecommender()
    
    def fit(self, data):
        self.popular.fit(data)
        self.kmeans.fit(data)
    
    def recommend(self, user_id, n=10):
        n_pop = int(n * self.alpha)
        n_pers = n - n_pop
        
        pop = self.popular.recommend(user_id, n)[:n_pop]
        pers = self.kmeans.recommend(user_id, n)[:n_pers]
        
        return pop + pers
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'popular': self.popular,
                'kmeans': self.kmeans
            }, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(alpha=data['alpha'])
        model.popular = data['popular']
        model.kmeans = data['kmeans']
        return model