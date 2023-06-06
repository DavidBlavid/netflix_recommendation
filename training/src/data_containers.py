from __future__ import annotations
from dataclasses import dataclass
import torch
import random

@dataclass
class UnmaskedUserData:
    user: int
    film_ids: list[int]
    ratings: list[float]

    def mask_values(self) -> MaskedUserData:
        """Mask values in the user data and return new object

        Returns:
            MaskedUserData: User values with random ratings masked
        """
        ids_and_ratings = list(zip(self.film_ids, self.ratings))
        random.shuffle(ids_and_ratings)

        shuffled_film_ids, shuffled_ratings = zip(*ids_and_ratings)
        max_masked_labels = min(len(shuffled_film_ids) // 3, 10)
        n_masked_ratings = random.randint(1, max_masked_labels)
        n_unmasked_ratings = random.randint(1, len(shuffled_film_ids)-n_masked_ratings)

        masked_film_ids = shuffled_film_ids[:n_masked_ratings]
        masked_ratings = shuffled_ratings[:n_masked_ratings]

        unmasked_film_ids = shuffled_film_ids[n_masked_ratings:(n_masked_ratings+n_unmasked_ratings)]
        unmasked_ratings = shuffled_ratings[n_masked_ratings:(n_masked_ratings+n_unmasked_ratings)]
        return MaskedUserData(
            user=self.user,
            masked_film_ids=masked_film_ids,
            masked_ratings=masked_ratings,
            unmasked_film_ids=unmasked_film_ids,
            unmasked_ratings=unmasked_ratings
        )
    
@dataclass
class MaskedUserData:
    user              : int
    masked_film_ids   : list[int]
    masked_ratings    : list[float]
    unmasked_film_ids : list[int]
    unmasked_ratings  : list[float]

@dataclass
class UserDataBatch:
    input_user_rating_vector : torch.Tensor
    target_user_rating_vector: torch.Tensor
    relevancy_vector         : torch.Tensor # 1 if user rated movie (both unmasked and masked), 0 otherwise
    masked_film_ids          : list[list[int]]
    masked_ratings           : list[list[float]]
    unmasked_film_ids        : list[int]
    unmasked_ratings         : list[float]
    n_masked_ratings         : torch.Tensor


@dataclass
class AutoencoderBatch:
    rating_vectors: torch.Tensor
    relevancy_vectors: torch.Tensor