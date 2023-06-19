import torch
import pandas as pd
import numpy as np
from .data_containers import DataBatch

def df_subset_to_batch(df_subset: pd.DataFrame, n_movies: int, device: torch.device, is_test: bool) -> torch.Tensor:
    vector = torch.zeros((len(df_subset), n_movies))
    relevancy_mask = torch.zeros((len(df_subset), n_movies))
    for index, (_, row) in enumerate(df_subset.iterrows()):
        vector[index, row['film']] = torch.from_numpy(np.array(row['rating'])).float()
        if is_test:
            relevant_films = [film_id for film_id, is_masked in zip(row['film'], row['is_masked']) if is_masked]
            relevancy_mask[index, relevant_films] = 1
        else:
            relevancy_mask[index, row['film']] = 1
    return DataBatch(
        input_vector=vector.to(device).float(),
        relevancy_mask=relevancy_mask.to(device).float()
    )

def train_step(df_subset, model, optim, n_movies, device, loss_fn, noising) -> float:
    model.train()
    optim.zero_grad()
    batch = df_subset_to_batch(df_subset, n_movies, device, is_test=False)
    pred = model(batch.input_vector)
    loss = loss_fn((pred*batch.relevancy_mask), batch.input_vector)
    loss.backward(retain_graph=True)
    optim.step()
    first_loss = loss.item()

    optim.zero_grad()
    second_pred = model(pred)
    loss = loss_fn(second_pred, pred)
    loss.backward()
    optim.step()

    return first_loss+loss.item()

def test_step(df_subset, model, n_movies, device, loss_fn) -> float:
    model.eval()
    batch = df_subset_to_batch(df_subset, n_movies, device, is_test=False)
    pred = model(batch.input_vector)
    loss = loss_fn((pred*batch.relevancy_mask), batch.input_vector)
    return loss.item()
