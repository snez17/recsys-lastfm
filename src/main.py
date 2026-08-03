"""FastAPI приложение для рекомендательной системы."""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

from models.popular import PopularRecommender
from models.itemknn import ItemKNNRecommender
from models.als import ALSRecommender
from models.usertop import UserTopRecommender
from models.hybrid import HybridRecommender
# ========================
# ВЫБОР МОДЕЛИ
# ========================

MODEL_NAME = "popular"

MODEL_PATHS = {
    "popular": Path(__file__).parent.parent / "data" / "models" / "popular_model.pkl",
    "itemknn": Path(__file__).parent.parent / "data" / "models" / "itemknn_model.pkl",
    "als": Path(__file__).parent.parent / "data" / "models" / "als_model.pkl",
    "usertop": Path(__file__).parent.parent / "data" / "models" / "usertop_model.pkl",
    "hybrid": Path(__file__).parent.parent / "data" / "models" / "hybrid_model.pkl",
}

MODEL_CLASSES = {
    "popular": PopularRecommender,
    "itemknn": ItemKNNRecommender,
    "als": ALSRecommender,
    "usertop": UserTopRecommender,
    "hybrid": HybridRecommender,
}

print(f"Загрузка модели {MODEL_NAME}...")

try:
    model = MODEL_CLASSES[MODEL_NAME].load(MODEL_PATHS[MODEL_NAME])
    model_type = MODEL_NAME.capitalize()
    print(f"Модель загружена: {model_type}")
except Exception as e:
    print(f"Не удалось загрузить {MODEL_NAME}: {e}")
    sys.exit(1)

# ========================
# ЗАГРУЗКА НАЗВАНИЙ ТРЕКОВ
# ========================

# Загружаем датасет для получения названий
from data_loader import load_dataset
df = load_dataset(year=2007)
track_names = df[['track_index', 'artist_name', 'track_name']].drop_duplicates(subset=['track_index'])

# Создаём словарь для быстрого доступа
track_map = track_names.set_index('track_index')[['artist_name', 'track_name']].to_dict(orient='index')

print(f"Загружено названий треков: {len(track_map)}")

# ========================
# ПРИЛОЖЕНИЕ
# ========================

app = FastAPI(
    title="LastFM RecSys API",
    description="Рекомендации на основе популярности",
    version="1.0"
)

class TrackRequest(BaseModel):
    artist: str
    track: str

# ========================
# ЭНДПОИНТЫ
# ========================

@app.get("/")
def root():
    return {
        "service": "LastFM RecSys",
        "status": "running",
        "model": model_type
    }

@app.get("/stats")
def get_stats():
    return {"message": "Статистика доступна в логах"}

@app.get("/recommend/{user_id}")
def recommend_user(user_id: str, n: int = 10):
    try:
        recs = model.recommend(user_id, n=n)
        if len(recs) == 0:
            raise HTTPException(status_code=404, detail="Пользователь не найден или нет рекомендаций")
        
        # Превращаем track_id в названия
        recommendations = []
        for track_id in recs:
            track_id_int = int(track_id)
            if track_id_int in track_map:
                recommendations.append({
                    "artist": track_map[track_id_int]['artist_name'],
                    "track": track_map[track_id_int]['track_name']
                })
            else:
                recommendations.append({
                    "artist": "Unknown",
                    "track": f"Track {track_id_int}"
                })
        
        return {"user_id": user_id, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/new")
def recommend_new_user(tracks: List[TrackRequest]):
    if not tracks:
        raise HTTPException(status_code=400, detail="Передайте хотя бы один трек")
    
    return {"message": "Функция холодного старта в разработке"}

# ========================
# ЗАПУСК
# ========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)