import logging
import sys
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import uvicorn

from model import load_model, get_user_cluster, get_recommendations, predict_new_user

# ========================
# НАСТРОЙКА ЛОГГИРОВАНИЯ
# ========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ========================
# ЗАГРУЗКА МОДЕЛИ
# ========================

logger.info("Загрузка модели...")
start_time = time.time()

try:
    model = load_model()
    user_clusters = model['user_clusters']
    cluster_recs = model['cluster_recs']
    kmeans = model['kmeans']
    scaler = model['scaler']
    user_artist = model['user_artist']
    
    logger.info(
        "Модель загружена: %d пользователей, %d кластеров, время %.2f сек",
        len(user_clusters), len(cluster_recs), time.time() - start_time
    )
except Exception as e:
    logger.error("Ошибка загрузки модели: %s", str(e))
    sys.exit(1)

# ========================
# ПРИЛОЖЕНИЕ
# ========================

app = FastAPI(
    title="LastFM RecSys API",
    description="Рекомендации на основе кластеризации пользователей",
    version="1.0"
)

class TrackRequest(BaseModel):
    artist: str
    track: str

# ========================
# ОБРАБОТЧИКИ ОШИБОК
# ========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning("HTTP ошибка: %d - %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Необработанная ошибка: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Внутренняя ошибка сервера"}
    )

# ========================
# ЭНДПОИНТЫ
# ========================

@app.get("/")
def root():
    logger.info("Запрос корневого эндпоинта")
    return {
        "service": "LastFM RecSys",
        "status": "running",
        "endpoints": [
            "/stats",
            "/cluster/{user_id}",
            "/recommend/{user_id}",
            "/recommend/new"
        ]
    }

@app.get("/stats")
def get_stats():
    logger.info("Запрос статистики")
    cluster_stats = []
    for c in range(len(cluster_recs)):
        users = len(user_clusters[user_clusters['cluster'] == c])
        cluster_stats.append({
            "cluster": c,
            "users": users,
            "tracks": len(cluster_recs[c])
        })
    return {
        "total_users": len(user_clusters),
        "total_clusters": len(cluster_recs),
        "clusters": cluster_stats
    }

@app.get("/cluster/{user_id}")
def get_user_cluster_endpoint(user_id: str):
    logger.info("Запрос кластера: user_id=%s", user_id)
    cluster = get_user_cluster(user_id, user_clusters)
    if cluster is None:
        logger.warning("Пользователь не найден: %s", user_id)
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return {"user_id": user_id, "cluster": cluster}

@app.get("/recommend/{user_id}")
def recommend_user_endpoint(user_id: str, top_n: int = 10):
    logger.info("Запрос рекомендаций: user_id=%s, top_n=%d", user_id, top_n)
    start = time.time()
    
    cluster, recommendations = get_recommendations(user_id, user_clusters, cluster_recs, top_n)
    
    if cluster is None:
        logger.warning("Пользователь не найден: %s", user_id)
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    logger.info(
        "Рекомендации выданы: user_id=%s, cluster=%d, count=%d, время=%.2f сек",
        user_id, cluster, len(recommendations), time.time() - start
    )
    return {
        "user_id": user_id,
        "cluster": cluster,
        "recommendations": recommendations
    }

@app.post("/recommend/new")
def recommend_new_user_endpoint(tracks: List[TrackRequest]):
    if not tracks:
        logger.warning("Пустой список треков")
        raise HTTPException(status_code=400, detail="Список треков пуст")
    
    tracks_list = [{"artist": t.artist, "track": t.track} for t in tracks]
    logger.info("Новый пользователь: %d треков", len(tracks_list))
    start = time.time()
    
    cluster, recommendations = predict_new_user(
        tracks_list, user_artist, scaler, kmeans, cluster_recs, top_n=10
    )
    
    logger.info(
        "Новый пользователь: cluster=%d, recommendations=%d, время=%.2f сек",
        cluster, len(recommendations), time.time() - start
    )
    return {
        "predicted_cluster": cluster,
        "recommendations": recommendations,
        "message": f"Пользователь отнесён к кластеру {cluster}"
    }

@app.get("/cluster/{cluster_id}/tracks")
def get_cluster_tracks_endpoint(cluster_id: int, top_n: int = 20):
    logger.info("Запрос треков кластера: cluster=%d, top_n=%d", cluster_id, top_n)
    
    if cluster_id not in cluster_recs:
        logger.warning("Кластер не найден: %d", cluster_id)
        raise HTTPException(status_code=404, detail="Кластер не найден")
    
    recs = cluster_recs[cluster_id][:top_n]
    return {
        "cluster": cluster_id,
        "tracks": [{"artist": artist, "track": track} for artist, track in recs]
    }

# ========================
# ЗАПУСК
# ========================

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )