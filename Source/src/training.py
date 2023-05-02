import polars as pl
import torch
import numpy as np

def df_sample_to_batch(sample: pl.DataFrame, n_movies: int, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    """
    Converts a sample from the dataframe to a batch for the model.

    Args:
        sample (pl.DataFrame): A sample from the dataframe.
        n_movies (int): The number of movies in the dataset.

    Returns:
        torch.Tensor: A batch for the model.
        int: The number of users in the batch.
    """
    n_ratings = []
    batch = np.zeros((len(sample), n_movies))
    for index, (user_movies, user_ratings) in enumerate(sample.to_numpy()[:, 1:]):
        batch[index, user_movies] = user_ratings
        n_ratings.append(len(user_movies))
    assert not (batch.sum(axis=1).any() == 0)
    return torch.Tensor(batch).to(device), torch.Tensor(n_ratings).to(device)

def calc_loss(ground_truth: torch.Tensor, predictions: torch.Tensor, n_ratings: torch.Tensor) -> float:
    """
    Calculates the loss between the ground truth and the predictions.

    Args:
        ground_truth (torch.Tensor): The ground truth.
        predictions (torch.Tensor): The predictions.

    Returns:
        torch.Tensor: The loss.
    """
    predictions = predictions * (ground_truth != 0).float()
    instance_losses = torch.sum((ground_truth - predictions) ** 2, dim=1) / n_ratings
    assert not torch.isnan(instance_losses).any()
    return instance_losses.sum()

def train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: torch.Tensor, n_ratings: torch.Tensor) -> torch.Tensor:
    """
    Performs a training step.

    Args:
        model (torch.nn.Module): The model.
        optimizer (torch.optim.Optimizer): The optimizer.
        batch (torch.Tensor): The batch.

    Returns:
        torch.Tensor: The loss.
    """
    optimizer.zero_grad()
    predictions = model(batch)
    loss = calc_loss(batch, predictions, n_ratings)
    loss.backward()
    optimizer.step()
    return loss

def test_step(model: torch.nn.Module, test_set: pl.DataFrame) -> torch.Tensor:
    """
    Performs a test step.

    Args:
        model (torch.nn.Module): The model.
        batch (torch.Tensor): The batch.

    Returns:
        torch.Tensor: The loss.
    """
    predictions = model(batch)
    loss = calc_loss(batch, predictions, n_ratings)
    return loss