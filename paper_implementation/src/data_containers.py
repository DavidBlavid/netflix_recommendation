from typing import NamedTuple
import torch

class DataBatch(NamedTuple):
    input_vector: torch.Tensor
    relevancy_mask: torch.Tensor

class TestBatch(NamedTuple):
    input_vector: torch.Tensor
    masked_films: list[int]
    masked_ratings: list[float]