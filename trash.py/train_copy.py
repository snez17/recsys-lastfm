import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import load_dataset
from src.models.popular import PopularRecommender
from src.models.usertop import UserTopRecommender
from src.models.userknn import UserKNNRecommender


def get_user_strategy(user_id, df_train, df_test, user_artist_matrix, user_ids):
    """
    Определяет стратегию для пользователя.
    
    Returns:
        dict: {
            'strategy': 'userknn' | 'usertop' | 'popular',
            'recommendations': list,
            'hit_rate': float
        }
    """
    # 1. Проверяем UserKNN (есть ли соседи)
    neighbors = find_neighbors(user_id, user_artist_matrix, user_ids, threshold=0.1, max_neighbors=3)
    if neighbors:
        recs = get_neighbor_tracks(neighbors, df_train, n=10)
        hit_rate = calculate_hit_rate(recs, df_test, user_id)
        return {'strategy': 'userknn', 'recommendations': recs, 'hit_rate': hit_rate}
    
    # 2. Проверяем UserTop (консервативен ли пользователь)
    user_tracks = df_train[df_train['user_id'] == user_id]
    total_plays = len(user_tracks)
    repeated_plays = user_tracks.groupby(['artist_name', 'track_name']).size()
    repeated_ratio = len(repeated_plays[repeated_plays >= 3]) / len(repeated_plays) if len(repeated_plays) > 0 else 0
    
    if repeated_ratio > 0.3:  # > 30% треков переслушивает
        usertop = UserTopRecommender()
        usertop.fit(df_train)
        recs = usertop.recommend(user_id, n=10)
        hit_rate = calculate_hit_rate(recs, df_test, user_id)
        return {'strategy': 'usertop', 'recommendations': recs, 'hit_rate': hit_rate}
    
    # 3. Иначе — Popular
    popular = PopularRecommender()
    popular.fit(df_train)
    recs = popular.recommend(user_id, n=10)
    hit_rate = calculate_hit_rate(recs, df_test, user_id)
    return {'strategy': 'popular', 'recommendations': recs, 'hit_rate': hit_rate}


def find_neighbors(user_id, user_artist_matrix, user_ids, threshold=0.9, max_neighbors=3):
    """Находит похожих пользователей."""
    idx = user_ids.index(user_id)
    user_vector = user_artist_matrix[idx].reshape(1, -1)
    similarities = cosine_similarity(user_vector, user_artist_matrix).flatten()
    
    neighbors = []
    for i, sim in enumerate(similarities):
        if i == idx:
            continue
        if sim > threshold:
            neighbors.append((user_ids[i], sim))
            if len(neighbors) >= max_neighbors:
                break
    return neighbors


def get_neighbor_tracks(neighbors, df_train, n=10):
    """Собирает треки соседей."""
    all_tracks = []
    for neighbor_id, _ in neighbors:
        tracks = df_train[df_train['user_id'] == neighbor_id][['artist_name', 'track_name']].values
        all_tracks.extend([(a, t) for a, t in tracks])
    
    # Сортируем по частоте
    from collections import Counter
    counter = Counter(all_tracks)
    return [track for track, _ in counter.most_common(n)]


def calculate_hit_rate(recommendations, df_test, user_id):
    """Считает hit rate для одного пользователя."""
    actual = df_test[df_test['user_id'] == user_id][['artist_name', 'track_name']].itertuples(index=False, name=None)
    actual_set = set(actual)
    hits = sum(1 for track in recommendations if track in actual_set)
    return 1 if hits > 0 else 0


def main():
    print("📂 Загрузка данных...")
    df = load_dataset(year=2007)
    
    # Разбиваем на train/test
    train_end = '2007-12-24'
    test_start = '2007-12-25'
    test_end = '2007-12-31'
    
    df_train = df[(df['timestamp'] <= train_end)]
    df_test = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)]
    
    # Строим матрицу user-artist для поиска соседей
    user_artist = df_train.groupby(['user_id', 'artist_name']).size().unstack(fill_value=0)
    user_ids = user_artist.index.tolist()
    user_artist_matrix = user_artist.values
    
    # Для каждого пользователя в тесте определяем стратегию
    print("\n📊 Оценка стратегий...")
    results = []
    for user_id in df_test['user_id'].unique():
        result = get_user_strategy(user_id, df_train, df_test, user_artist_matrix, user_ids)
        results.append({
            'user_id': user_id,
            'strategy': result['strategy'],
            'hit_rate': result['hit_rate']
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n📊 Результаты по стратегиям:")
    for strategy in ['userknn', 'usertop', 'popular']:
        subset = results_df[results_df['strategy'] == strategy]
        if len(subset) > 0:
            print(f"  {strategy}: {len(subset)} пользователей, hit rate = {subset['hit_rate'].mean():.2%}")
    
    print(f"\n🎯 Общий Hit Rate: {results_df['hit_rate'].mean():.2%}")


if __name__ == "__main__":
    main()