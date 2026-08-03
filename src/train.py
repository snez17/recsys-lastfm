from data_loader import load_dataset
from models.popular import PopularRecommender
from models.als import ALSRecommender
from models.itemknn import ItemKNNRecommender
from models.usertop import UserTopRecommender
from models.hybrid import HybridRecommender  
from evaluate import evaluate_model
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    df = load_dataset(year=2007)
    
    # Фильтруем по датам
    start = '2007-01-01'
    train_end = '2007-12-24'
    test_start = '2007-12-25'
    test_end = '2007-12-31'
    
    # Отсекаем выбросы (99-й перцентиль) 
    research = df[(df['timestamp'] >= start) & (df['timestamp'] <= test_end)]
    user_activity = research.groupby('user_id').size()
    threshold = np.percentile(user_activity, 99)
    df_filtered = research[research.groupby('user_id')['user_id'].transform('size') <= threshold]
    
    # Разбиваем на train и test
    df_train = df_filtered[(df_filtered['timestamp'] >= start) & (df_filtered['timestamp'] <= train_end)]
    df_test = df_filtered[(df_filtered['timestamp'] >= test_start) & (df_filtered['timestamp'] <= test_end)]
    
    # Сжимаем train
    df_train_compressed = df_train.groupby(['user_index', 'track_index']).size().reset_index(name='count')
    
    # ========== ФИЛЬТРУЕМ test ТЕМИ ЖЕ ТРЕКАМИ ==========
    train_tracks = df_train_compressed['track_index'].unique()
    df_test = df_test[df_test['track_index'].isin(train_tracks)]
    
    # Вместо общего количества
    print(f"Уникальных треков в train: {df_train_compressed['track_index'].nunique()}")
    print(f"Уникальных треков в test после фильтрации: {df_test['track_index'].nunique()}")
    print(f"Уникальных пользователей в train: {df_train_compressed['user_index'].nunique()}")
    print(f"Уникальных пользователей в test: {df_test['user_id'].nunique()}")
        
    # ==================== МОДЕЛИ ====================
    models = [
        (UserTopRecommender(top_n=10), "UserTop"), 
        (PopularRecommender(), "Popular"),
        (ItemKNNRecommender(K=50), "ItemKNN"),
        (ALSRecommender(factors=50, iterations=15), "ALS"),
        (HybridRecommender(threshold=0.1, max_neighbors=3), "Hybrid"),
    ]

    results = []

    for model, name in models:
        model_path = Path(f'data/models/{name.lower()}_model.pkl')
        
        if model_path.exists():
            print(f"📂 Загружаем {name}...")
            model = model.__class__.load(model_path)  # ← загружаем
        else:
            print(f"🔄 Обучаем {name}...")
            model.fit(df_train_compressed)            # ← обучаем
            model.save(model_path)                   # ← сохраняем
        
        metrics = evaluate_model(model, df_test, k=50, sample_size=None)
        
        print(f"  Hit Rate: {metrics['hit_rate']:.2%}")  # доля юзеров у окторых хотя бы один трек сошелся
        print(f"  Precision: {metrics['precision']:.2%}")  # сколько релевантного из реально прослушанного
        print(f"  NDCG: {metrics['ndcg']:.2%}") # учитывает порядок рекомендации

        results.append({
        'model': name,
        'hit_rate': metrics['hit_rate'],
        'precision': metrics['precision'],
        'ndcg': metrics['ndcg'],
    })
        
    df = pd.DataFrame(results)

    # Сохраняем в Markdown
    with open('notebooks/model_comparison.md', 'w', encoding='utf-8') as f:
        f.write("# 📊 Сравнение моделей\n\n")
        f.write("| Модель | Hit Rate (%) | Precision (%) | NDCG (%) |\n")
        f.write("|--------|--------------|---------------|----------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['model']} | {row['hit_rate']:.3f} | {row['precision']:.3f} | {row['ndcg']:.3f} |\n")

if __name__ == "__main__":
    main()