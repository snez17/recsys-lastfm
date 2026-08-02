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

```
recsys/
├── src/
│   ├── main.py          # FastAPI приложение
│   ├── model.py         # Логика модели
│   └── train.py         # Обучение модели
├── tests/               # Тесты
├── data/                # Данные
├── notebooks/           # Jupyter ноутбуки (EDA)
├── archive/             # Архивные скрипты
├── .gitignore
├── README.md
└── requirements.txt     # Зависимости
```

## 🚀 API Endpoints

### Получить рекомендации для пользователя

**Endpoint:** `GET /recommend/{user_id}`

**Параметры:**
| Параметр | Тип | Описание |
|----------|-----|----------|
| `user_id` | `string` | ID пользователя (например, `user_000001`) |

---

### 📥 Пример запроса

```http
GET /recommend/user_000001
```
### 📤 Пример ответа

```json
{
  "user_id": "user_000001",
  "cluster": 2,
  "recommendations": [
    {
      "artist": "Radiohead",
      "track": "Creep"
    },
    {
      "artist": "Nirvana",
      "track": "Smells Like Teen Spirit"
    }
  ]
}
```

## 🧪 Тесты

```bash
pytest tests/
```

## 📊 Результаты

- **Пользователей:** 794
- **Кластеров:** 5
- **Hit Rate:** 25%


## 📄 Лицензия

MIT
