from data_loader import load_dataset
from models.popular import PopularRecommender
from models.als import ALSRecommender
from models.itemknn import ItemKNNRecommender
from models.usertop import UserTopRecommender
from evaluate import evaluate_model
import numpy as np
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
    
    print(f"Треков в train: {len(train_tracks)}")
    print(f"Треков в test после фильтрации: {df_test['track_index'].nunique()}")
    
    # ==================== МОДЕЛИ ====================
    models = [
        (UserTopRecommender(top_n=10), "UserTop"),  # ← добавляем
        (PopularRecommender(), "Popular"),
        (ItemKNNRecommender(K=50), "ItemKNN"),
        (ALSRecommender(factors=50, iterations=15), "ALS"),
    ]


    for model, name in models:
        print(f"\n🔄 {name}...")
        model.fit(df_train_compressed)

        # ========== СОХРАНЯЕМ МОДЕЛЬ ==========
        save_path = Path(__file__).parent.parent / "data" / "models" / f"{name.lower()}_model.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        print(f"✅ Модель сохранена: {save_path}")
        
        metrics = evaluate_model(model, df_test, k=50, sample_size=None)
        
        print(f"  Hit Rate: {metrics['hit_rate']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  NDCG: {metrics['ndcg']:.2%}")

if __name__ == "__main__":
    main()