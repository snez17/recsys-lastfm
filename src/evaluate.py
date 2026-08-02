import pandas as pd
import numpy as np
from metrics import hit_rate_at_k, recall_at_k, precision_at_k, ndcg_at_k


def evaluate_model(model, test_data, k=10, sample_size=None):
    results = []
    users = test_data['user_id'].unique()
    if sample_size:
        users = np.random.choice(users, min(sample_size, len(users)), replace=False)
    
    for user_id in users:
        recs = model.recommend(user_id, n=k)  # ← индексы
        actual = test_data[test_data['user_id'] == user_id]['track_index'].values[:k]  # ← индексы
        
        if len(actual) < k:
            actual = test_data[test_data['user_id'] == user_id]['track_index'].values
        
        results.append({
            'user_id': user_id,
            'hit_rate': hit_rate_at_k(recs, actual, k),
            'recall': recall_at_k(recs, actual, k),
            'precision': precision_at_k(recs, actual, k),
            'ndcg': ndcg_at_k(recs, actual, k)
        })
    
    df = pd.DataFrame(results)
    return {
        'hit_rate': df['hit_rate'].mean(),
        'recall': df['recall'].mean(),
        'precision': df['precision'].mean(),
        'ndcg': df['ndcg'].mean(),
        'df_results': df
    }
