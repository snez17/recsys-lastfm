#%%

# file_path = r'C:\Users\23167700\Downloads\lastfm-dataset-1K\userid-timestamp-artid-artname-traid-traname.tsv'

# total_rows = 0

# with open(file_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         total_rows += 1
#         if total_rows % 100000 == 0:
#             print(f"Обработано: {total_rows:,} строк", end='\r')

# print(f"\n✅ Всего строк в файле: {total_rows:,}")


#%%
# import pandas as pd
# import os

# file_path = r'C:\Users\23167700\Downloads\lastfm-dataset-1K\userid-timestamp-artid-artname-traid-traname.tsv'
# column_names = ['user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']

# # Проверяем, существует ли уже файл
# parquet_file = 'lastfm_data.parquet'

# # Если файл существует — удаляем (чтобы начать заново)
# if os.path.exists(parquet_file):
#     os.remove(parquet_file)

# print("🔄 Начинаю конвертацию в Parquet...")

# for i, chunk in enumerate(pd.read_csv(
#     file_path,
#     encoding='utf-8',
#     sep='\t',
#     names=column_names,
#     chunksize=500000,
#     on_bad_lines='skip',
#     engine='python',
#     quoting=3
# )):
#     chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')
    
#     # Первый блок — создаём файл
#     if i == 0:
#         chunk.to_parquet(parquet_file, engine='pyarrow', index=False)
#     else:
#         # Следующие блоки — читаем существующий файл и дозаписываем
#         existing_df = pd.read_parquet(parquet_file)
#         combined_df = pd.concat([existing_df, chunk], ignore_index=True)
#         combined_df.to_parquet(parquet_file, engine='pyarrow', index=False)
    
#     print(f"✅ Блок {i+1} сохранён ({len(chunk)} строк)")

# print("🎉 Готово! Файл сохранён как lastfm_data.parquet")

# %%

# import pandas as pd

# df = pd.read_parquet('lastfm_data.parquet', columns=['timestamp'])
# df['timestamp'] = pd.to_datetime(df['timestamp'])

# print(f"Самая ранняя дата: {df['timestamp'].min()}")
# print(f"Самая поздняя дата: {df['timestamp'].max()}")

#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# ========================
# ПАРАМЕТРЫ (2008 год)
# ========================
PERIOD1_START = pd.Timestamp('2008-05-01', tz='UTC')
PERIOD1_END   = pd.Timestamp('2008-06-30', tz='UTC')
PERIOD2_START = pd.Timestamp('2008-07-01', tz='UTC')
PERIOD2_END   = pd.Timestamp('2008-08-31', tz='UTC')

N_CLUSTERS = 5
TOP_N_RECS = 10

print("📂 Загружаем данные...")

# ========================
# ЭТАП 1: Загрузка периодов
# ========================

df_train = pd.read_parquet(
    'lastfm_data.parquet',
    filters=[('timestamp', '>=', PERIOD1_START), ('timestamp', '<=', PERIOD1_END)]
)
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])

df_test = pd.read_parquet(
    'lastfm_data.parquet',
    filters=[('timestamp', '>=', PERIOD2_START), ('timestamp', '<=', PERIOD2_END)]
)
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])

print(f"✅ Обучающая выборка: {len(df_train):,} записей")
print(f"✅ Тестовая выборка:  {len(df_test):,} записей")

if len(df_train) == 0 or len(df_test) == 0:
    print("❌ Нет данных за выбранные даты!")
    exit()

# ========================
# ЭТАП 2: Кластеризация пользователей
# ========================

print("\n🔄 Кластеризация пользователей...")

user_artist = df_train.groupby(['user_id', 'artist_name']).size().unstack(fill_value=0)

# Убираем слишком редких исполнителей
artist_counts = df_train['artist_name'].value_counts()
popular_artists = artist_counts[artist_counts >= 5].index
user_artist = user_artist[popular_artists]

# Масштабирование
scaler = StandardScaler()
X = scaler.fit_transform(user_artist)

# Кластеризация
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
user_artist['cluster'] = kmeans.fit_predict(X)
user_clusters = user_artist[['cluster']].reset_index()

print(f"✅ Пользователи распределены по {N_CLUSTERS} кластерам")

# ========================
# ЭТАП 3: Рекомендации для каждого кластера
# ========================

print("\n🎵 Формируем рекомендации для кластеров...")

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
# ЭТАП 4: Проверка hit rate
# ========================

print("\n📊 Оценка рекомендаций на тестовом периоде...")

results = []

for user_id in user_clusters['user_id'].unique():
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

print(f"\n👥 Пользователей: {len(results_df):,}")
print(f"🎯 Средний Hit Rate: {results_df['hit_rate'].mean():.2%}")
print(f"📈 Максимальный Hit Rate: {results_df['hit_rate'].max():.2%}")
print(f"📉 Минимальный Hit Rate: {results_df['hit_rate'].min():.2%}")

print("\n📊 По кластерам:")
for cluster in range(N_CLUSTERS):
    cluster_df = results_df[results_df['cluster'] == cluster]
    print(f"  Кластер {cluster}: {len(cluster_df)} пользователей, средний hit rate = {cluster_df['hit_rate'].mean():.2%}")

# ========================
# СОХРАНЯЕМ МОДЕЛЬ
# ========================

with open('lastfm_model.pkl', 'wb') as f:
    pickle.dump({
        'user_clusters': user_clusters,
        'cluster_recs': cluster_recs,
        'kmeans': kmeans,
        'scaler': scaler,
        'user_artist': user_artist,
        'popular_artists': popular_artists
    }, f)

print("\n✅ Модель сохранена в lastfm_model.pkl")
print("✅ Детальные результаты сохранены в recommendation_results.csv")
results_df.to_csv('recommendation_results.csv', index=False)