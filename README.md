# 🎵 LastFM Recommender System

Рекомендательная система на основе кластеризации пользователей LastFM.

## 📖 О проекте

Система кластеризует пользователей на основе их музыкальных предпочтений (исполнители, которые они слушают) и выдаёт персонализированные рекомендации. Подходит для холодного старта и быстрой кластеризации.


## 🛠 Технологии

- **FastAPI** — API
- **scikit-learn** — кластеризация
- **Pandas** — данные
- **Pytest** — тесты
- **Logging** — логирование



## 📁 Структура

ecsys/
├── src/
│ ├── main.py # FastAPI
│ ├── model.py # Логика модели
│ └── train.py # Обучение
├── tests/ # Тесты
├── data/ # Данные
├── notebooks/ # EDA
└── requirements.txt

## 📡 API

### GET /recommend/{user_id}
**Пример:** `GET /recommend/user_000001`
**Ответ:**
json
{
  "user_id": "user_000001",
  "cluster": 2,
  "recommendations": [
    {"artist": "Radiohead", "track": "Creep"},
    {"artist": "Nirvana", "track": "Smells Like Teen Spirit"}
  ]
} 

## 🧪 Тесты

bash
pytest tests/


## 📊 Результаты

- **Пользователей:** 794
- **Кластеров:** 5
- **Hit Rate:** 25%


## 📄 Лицензия

MIT
