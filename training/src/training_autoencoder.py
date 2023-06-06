import pandas as pd
import torch
import numpy as np
from .data_containers import AutoencoderBatch

def train_test_split(data: pd.Series, test_percentage: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_test = int(len(data) * test_percentage)
    data = data.sample(frac=1)
    return data.iloc[n_test:], data.iloc[:n_test]

def batch_from_user_ratings(ratings: pd.DataFrame, n_movies: int, device: str) -> torch.Tensor:
    vector = np.zeros((len(ratings), n_movies))
    relevancy_vector = np.zeros((len(ratings), n_movies))
    for index, (_, row) in enumerate(ratings.iterrows()):
        vector[index, row['film']] = row['relative_rating']
        relevancy_vector[index, row['film']] = 1
    vector[vector < 0] = -1
    vector[vector > 0] = 1
    return AutoencoderBatch(
        rating_vectors=torch.from_numpy(vector).to(device).float(),
        relevancy_vectors=torch.from_numpy(relevancy_vector).to(device).float()
    )