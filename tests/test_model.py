import pytest
import pandas as pd
from pathlib import Path
import sys
import pickle

# Добавляем src в путь (работает на всех ОС)
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Импорт должен работать
from model import get_user_cluster, get_recommendations


@pytest.fixture
def sample_user_clusters():
    return pd.DataFrame({
        'user_id': ['user_001', 'user_002', 'user_003'],
        'cluster': [0, 1, 0]
    })


@pytest.fixture
def sample_cluster_recs():
    return {
        0: [('Artist A', 'Track 1'), ('Artist A', 'Track 2')],
        1: [('Artist B', 'Track 3')]
    }


def test_existing_user(sample_user_clusters):
    cluster = get_user_cluster('user_001', sample_user_clusters)
    assert cluster == 0


def test_missing_user(sample_user_clusters):
    cluster = get_user_cluster('user_999', sample_user_clusters)
    assert cluster is None


def test_get_recommendations(sample_user_clusters, sample_cluster_recs):
    cluster, recs = get_recommendations('user_001', sample_user_clusters, sample_cluster_recs, top_n=2)
    assert cluster == 0
    assert len(recs) == 2