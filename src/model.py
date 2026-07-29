import pickle
import pandas as pd
from typing import List, Dict, Optional, Tuple

# ========================
# ЗАГРУЗКА МОДЕЛИ
# ========================

def load_model(model_path: str = 'data/lastfm_model.pkl') -> Dict:
    """
    Загружает сохранённую модель из файла.
    
    Args:
        model_path: Путь к файлу модели
        
    Returns:
        Словарь с компонентами модели
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return {
        'user_clusters': model_data['user_clusters'],
        'cluster_recs': model_data['cluster_recs'],
        'kmeans': model_data['kmeans'],
        'scaler': model_data['scaler'],
        'user_artist': model_data['user_artist'],
        'popular_artists': model_data['popular_artists']
    }

# ========================
# РАБОТА С ПОЛЬЗОВАТЕЛЯМИ
# ========================

def get_user_cluster(user_id: str, user_clusters: pd.DataFrame) -> Optional[int]:
    """
    Возвращает кластер пользователя.
    
    Args:
        user_id: ID пользователя
        user_clusters: DataFrame с кластерами
        
    Returns:
        Номер кластера или None, если пользователь не найден
    """
    if user_id not in user_clusters['user_id'].values:
        return None
    return int(user_clusters[user_clusters['user_id'] == user_id]['cluster'].values[0])


def get_recommendations(
    user_id: str, 
    user_clusters: pd.DataFrame, 
    cluster_recs: Dict, 
    top_n: int = 10
) -> Tuple[Optional[int], Optional[List[Dict[str, str]]]]:
    """
    Возвращает рекомендации для существующего пользователя.
    
    Args:
        user_id: ID пользователя
        user_clusters: DataFrame с кластерами
        cluster_recs: Словарь с рекомендациями по кластерам
        top_n: Количество рекомендаций
        
    Returns:
        Tuple (cluster, recommendations) или (None, None) если пользователь не найден
    """
    cluster = get_user_cluster(user_id, user_clusters)
    if cluster is None:
        return None, None
    
    recs = cluster_recs[cluster][:top_n]
    recommendations = [{"artist": artist, "track": track} for artist, track in recs]
    return cluster, recommendations

def get_popular_tracks(df, week_start, top_n=10):
    """
    Возвращает топ-N самых популярных треков за указанную неделю.
    
    Args:
        df: DataFrame с данными
        week_start: начало недели (например, '2006-01-01')
        top_n: количество треков
    
    Returns:
        Список рекомендаций [{"artist": "...", "track": "..."}, ...]
    """
    week_end = pd.to_datetime(week_start) + pd.Timedelta(days=6)
    
    # Фильтруем по неделе
    df_week = df[(df['timestamp'] >= week_start) & (df['timestamp'] <= week_end)]
    
    # Топ треков
    top_tracks = (
        df_week
        .groupby(['artist_name', 'track_name'])
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )
    
    return [{"artist": artist, "track": track} for artist, track in top_tracks]

# ========================
# НОВЫЙ ПОЛЬЗОВАТЕЛЬ
# ========================

def predict_new_user(
    tracks: List[Dict[str, str]],
    user_artist: pd.DataFrame,
    scaler,
    kmeans,
    cluster_recs: Dict,
    top_n: int = 10
) -> Tuple[int, List[Dict[str, str]]]:
    """
    Предсказывает кластер и возвращает рекомендации для нового пользователя.
    
    Args:
        tracks: Список словарей с ключами 'artist' и 'track'
        user_artist: Матрица user-artist
        scaler: StandardScaler
        kmeans: Обученная модель KMeans
        cluster_recs: Словарь с рекомендациями по кластерам
        top_n: Количество рекомендаций
        
    Returns:
        Tuple (cluster, recommendations)
    """
    # Создаём вектор нового пользователя
    new_user_vector = pd.Series(0, index=user_artist.columns[:-1])
    
    for track in tracks:
        artist = track.get('artist')
        if artist in new_user_vector.index:
            new_user_vector[artist] += 1
    
    # Масштабируем
    X_new = scaler.transform(new_user_vector.values.reshape(1, -1))
    
    # Определяем кластер
    cluster = int(kmeans.predict(X_new)[0])
    
    # Рекомендации
    recs = cluster_recs[cluster][:top_n]
    recommendations = [{"artist": artist, "track": track} for artist, track in recs]
    
    return cluster, recommendations

# ========================
# СТАТИСТИКА
# ========================

def get_cluster_stats(user_clusters: pd.DataFrame, cluster_recs: Dict) -> List[Dict]:
    """
    Возвращает статистику по кластерам.
    
    Args:
        user_clusters: DataFrame с кластерами
        cluster_recs: Словарь с рекомендациями по кластерам
        
    Returns:
        Список словарей со статистикой
    """
    stats = []
    for cluster in range(len(cluster_recs)):
        users_in_cluster = len(user_clusters[user_clusters['cluster'] == cluster])
        stats.append({
            "cluster": cluster,
            "users": int(users_in_cluster),
            "tracks_in_recs": len(cluster_recs[cluster])
        })
    return stats


def predict_new_user_with_fallback(
    tracks, 
    user_artist, 
    scaler, 
    kmeans, 
    cluster_recs, 
    df_all,
    week_start,
    min_observations=5,
    top_n=10
):
    """
    Предсказывает кластер или возвращает популярные треки, если данных мало.
    
    Args:
        tracks: список прослушанных треков
        min_observations: минимальное число треков для кластеризации
        week_start: неделя для популярных треков (если нужен fallback)
    
    Returns:
        (cluster, recommendations, is_fallback)
    """
    # Если треков мало → fallback на популярное
    if len(tracks) < min_observations:
        popular = get_popular_tracks(df_all, week_start, top_n)
        return None, popular, True  # fallback
    
    # Иначе — обычная кластеризация
    cluster, recommendations = predict_new_user(
        tracks, user_artist, scaler, kmeans, cluster_recs, top_n
    )
    return cluster, recommendations, False