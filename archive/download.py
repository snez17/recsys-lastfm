# Установка (если ещё не установлена)
# pip install datasets
#%%

# import pickle
# from datasets import load_dataset

# print("📥 Загрузка датасета...")
# dataset = load_dataset("matthewfranglen/lastfm-1k")

# print("💾 Сохранение в dataset.pkl...")
# with open('dataset.pkl', 'wb') as f:
#     pickle.dump(dataset, f)

# print("✅ Датасет сохранён в dataset.pkl")

# # Проверка
# print("\n🔍 Проверка загрузки из файла:")
# with open('dataset.pkl', 'rb') as f:
#     loaded = pickle.load(f)
#     print(f"✅ train: {len(loaded['train']):,} записей")
#     print(f"✅ valid: {len(loaded['valid']):,} записей")
#     print(f"✅ test: {len(loaded['test']):,} записей")
