def hit_rate_at_k(recommendations, actual, k=10):
    actual_set = set(actual)
    hits = sum(1 for item in recommendations[:k] if item in actual_set)
    return 1 if hits > 0 else 0

def recall_at_k(recommendations, actual, k=10):
    actual_set = set(actual)
    if not actual_set:
        return 0.0
    hits = sum(1 for item in recommendations[:k] if item in actual_set)
    return hits / len(actual_set)

def precision_at_k(recommendations, actual, k=10):
    if k == 0:
        return 0.0
    actual_set = set(actual)
    hits = sum(1 for item in recommendations[:k] if item in actual_set)
    return hits / k

def ndcg_at_k(recommendations, actual, k=10):
    actual_set = set(actual)
    if not actual_set:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(recommendations[:k]):
        if item in actual_set:
            dcg += 1 / (i + 2)  # упрощённая версия без log2
    idcg = sum(1 / (i + 2) for i in range(min(len(actual_set), k)))
    return dcg / idcg if idcg > 0 else 0.0