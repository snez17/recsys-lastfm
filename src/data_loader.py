import pickle
from pathlib import Path
import pandas as pd

def load_dataset(year=None):
    data_dir = Path(__file__).parent.parent / "data"
    
    # Если год не указан — загружаем всё
    if year is None:
        with open(data_dir / "dataset.pkl", 'rb') as f:
            return pickle.load(f)
    
    # Проверяем кеш
    cache_path = data_dir / f"dataset_{year}.pkl"
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Создаём кеш
    with open(data_dir / "dataset.pkl", 'rb') as f:
        df = pickle.load(f)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'].dt.year == year].copy()
    
    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)
    
    return df