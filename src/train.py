# %%
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ========================
# ЗАГРУЗКА ДАННЫХ
# ========================

print("📂 Загрузка датасета из data/dataset.pkl...")
with open('data/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

df_train = dataset["train"].to_pandas()
df_valid = dataset["valid"].to_pandas()
df_test = dataset["test"].to_pandas()

print(f"✅ train: {len(df_train):,} записей")
print(f"✅ valid: {len(df_valid):,} записей")
print(f"✅ test: {len(df_test):,} записей")

# ========================
# ПАРАМЕТРЫ
# ========================

N_CLUSTERS = 5
TOP_N_RECS = 10
MIN_ARTIST_PLAYS = 5

# ========================
# ОБУЧЕНИЕ МОДЕЛИ
# ========================

print("\n🔄 Кластеризация пользователей...")

# Создаём матрицу user-artist
user_artist = df_train.groupby(['user_id', 'artist_name']).size().unstack(fill_value=0)

# Убираем редких исполнителей
artist_counts = df_train['artist_name'].value_counts()
popular_artists = artist_counts[artist_counts >= MIN_ARTIST_PLAYS].index
user_artist = user_artist[popular_artists]

print(f"✅ Исполнителей: {len(popular_artists):,}")
print(f"✅ Пользователей: {len(user_artist):,}")

# Масштабирование
scaler = StandardScaler()
X = scaler.fit_transform(user_artist)

# Кластеризация
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
user_artist['cluster'] = kmeans.fit_predict(X)
user_clusters = user_artist[['cluster']].reset_index()

print(f"✅ Пользователи распределены по {N_CLUSTERS} кластерам")

# ========================
# РЕКОМЕНДАЦИИ ДЛЯ КЛАСТЕРОВ
# ========================

print("\n🎵 Формируем рекомендации...")
cluster_recs = {}

for cluster in range(N_CLUSTERS):
    users = user_clusters[user_clusters['cluster'] == cluster]['user_id']
    cluster_data = df_train[df_train['user_id'].isin(users)]
    top_tracks = (
        cluster_data
        .groupby(['artist_name', 'track_name'])
        .size()
        .sort_values(ascending=False)
        .head(TOP_N_RECS)
        .index
        .tolist()
    )
    cluster_recs[cluster] = top_tracks
    print(f"  Кластер {cluster}: {len(top_tracks)} треков")

# ========================
# ОЦЕНКА МОДЕЛИ (HIT RATE)
# ========================

print("\n📊 Оценка модели на тестовых данных...")

results = []
for user_id in user_clusters['user_id'].unique()[:100]:  # 100 пользователей для скорости
    cluster = user_clusters[user_clusters['user_id'] == user_id]['cluster'].values[0]
    recs = cluster_recs[cluster]
    
    user_test = df_test[df_test['user_id'] == user_id]
    user_tracks = set(user_test[['artist_name', 'track_name']].dropna().itertuples(index=False, name=None))
    
    hit_count = sum(1 for track in recs if track in user_tracks)
    hit_rate = hit_count / len(recs) if recs else 0
    
    results.append({
        'user_id': user_id,
        'cluster': cluster,
        'hit_rate': hit_rate,
        'listened': hit_count,
        'total_recs': len(recs)
    })

results_df = pd.DataFrame(results)

# ========================
# ВЫВОД РЕЗУЛЬТАТОВ
# ========================

print("\n" + "="*60)
print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
print("="*60)

print(f"\n👥 Проверено пользователей: {len(results_df)}")
print(f"🎯 Средний Hit Rate: {results_df['hit_rate'].mean():.2%}")
print(f"📈 Максимальный Hit Rate: {results_df['hit_rate'].max():.2%}")
print(f"📉 Минимальный Hit Rate: {results_df['hit_rate'].min():.2%}")

print("\n📊 По кластерам:")
for cluster in range(N_CLUSTERS):
    cluster_df = results_df[results_df['cluster'] == cluster]
    if len(cluster_df) > 0:
        print(f"  Кластер {cluster}: {len(cluster_df)} пользователей, hit rate = {cluster_df['hit_rate'].mean():.2%}")

# ========================
# СОХРАНЕНИЕ МОДЕЛИ
# ========================

print("\n💾 Сохранение модели...")
with open('data/lastfm_model.pkl', 'wb') as f:
    pickle.dump({
        'user_clusters': user_clusters,
        'cluster_recs': cluster_recs,
        'kmeans': kmeans,
        'scaler': scaler,
        'user_artist': user_artist,
        'popular_artists': popular_artists
    }, f)

print("✅ Модель сохранена в data/lastfm_model.pkl")

# Сохраняем результаты оценки
results_df.to_csv('data/recommendation_results.csv', index=False)
print("✅ Результаты сохранены в data/recommendation_results.csv")

print("\n🎉 Всё готово! Модель обучена и оценена.")