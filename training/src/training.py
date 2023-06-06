import polars as pl
import pandas as pd
import src.data_containers as containers
import torch

def format_user_data(data: pl.DataFrame) -> pd.Series:
    """Format user data into a series of UserData objects.

    Args:
        data (pl.DataFrame): pl dataframe containing user data.

    Returns:
        pd.Series: Series of UserData objects.
    """

    formatted = []
    for user_id, film_ids, ratings in data[["user", "film", "relative_rating"]].to_numpy():
        formatted.append(containers.UnmaskedUserData(user=user_id, film_ids=film_ids, ratings=ratings))
    return pd.Series(formatted)


def train_test_split(data: pd.Series, test_percentage: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    n_test = int(len(data) * test_percentage)
    data = data.sample(frac=1)
    return data.iloc[n_test:], data.iloc[:n_test]

def masked_user_data_to_batch(masked_user_data: list[containers.MaskedUserData] | pd.Series, n_movies: int) -> containers.UserDataBatch:
    inp_vector        = torch.zeros(len(masked_user_data), n_movies)
    target_vector     = torch.zeros(len(masked_user_data), n_movies)
    relevancy_vector  = torch.zeros(len(masked_user_data), n_movies)
    masked_film_ids   = []
    masked_ratings    = []
    unmasked_film_ids = []
    unmasked_ratings  = []
    n_masked_ratings  = torch.zeros(len(masked_user_data))


    for i, user_data in enumerate(masked_user_data):

        inp_vector[i, user_data.unmasked_film_ids]       = torch.tensor(user_data.unmasked_ratings, dtype=torch.float32)
        target_vector[i, user_data.unmasked_film_ids]    = torch.tensor(user_data.unmasked_ratings, dtype=torch.float32)
        target_vector[i, user_data.masked_film_ids]      = torch.tensor(user_data.masked_ratings, dtype=torch.float32)
        relevancy_vector[i, user_data.unmasked_film_ids] = 1
        relevancy_vector[i, user_data.masked_film_ids]   = 1
        n_masked_ratings[i]                              = len(user_data.masked_ratings)

        masked_film_ids.append(user_data.masked_film_ids)
        masked_ratings.append(user_data.masked_ratings)
        unmasked_film_ids.append(user_data.unmasked_film_ids)
        unmasked_ratings.append(user_data.unmasked_ratings)

    return containers.UserDataBatch(
        input_user_rating_vector=inp_vector,
        target_user_rating_vector=target_vector,
        relevancy_vector=relevancy_vector,
        masked_film_ids=masked_film_ids,
        masked_ratings=masked_ratings,
        unmasked_film_ids=unmasked_film_ids,
        unmasked_ratings=unmasked_ratings,
        n_masked_ratings=n_masked_ratings
    )