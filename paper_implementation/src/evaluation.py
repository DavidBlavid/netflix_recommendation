import torch
import pandas as pd
from .data_containers import TestBatch


def get_embeddings(n_movies: int, model: torch.nn.Sequential, device: torch.device) -> torch.Tensor:
    model.eval()
    model.to(device)
    rating_matrix = torch.eye(n_movies)*5
    with torch.no_grad():
        return model.encode(rating_matrix.to(device).float()).to("cpu")
    
def df_subset_to_test_batch(subset: pd.DataFrame, n_movies: int) -> TestBatch:
    """Gererate a TestBatch from a subset of the dataframe.

    Args:
        subset (pd.DataFrame): dataframe subset with columns 'film', 'rating' and 'is_masked'
        n_movies (int): _description_

    Returns:
        TestBatch: _description_
    """
    input_vector = torch.zeros((len(subset), n_movies))
    masked_films_batch = []
    masked_ratings_batch = []

    for index, (_, row) in enumerate(subset.iterrows()):
        unmasked_films = [film_id for film_id, is_masked in zip(row['film'], row['is_masked']) if not is_masked]
        unmasked_ratings = [rating for rating, is_masked in zip(row['rating'], row['is_masked']) if not is_masked]
        masked_films = [film_id for film_id, is_masked in zip(row['film'], row['is_masked']) if is_masked]
        masked_ratings = [rating for rating, is_masked in zip(row['rating'], row['is_masked']) if is_masked]
        input_vector[index, unmasked_films] = torch.tensor(unmasked_ratings).float()
        masked_films_batch.append(masked_films)
        masked_ratings_batch.append(masked_ratings)
    
    return TestBatch(input_vector, masked_films_batch, masked_ratings_batch)

def recommend_n_movies(model: torch.nn.Sequential, test_batch: TestBatch, embeddings: torch.Tensor, n_recommendations: int, device: torch.device, n_movies: int):
    user_embeddings = model.encode(test_batch.input_vector.to(device).float()).to("cpu")
    n_users, latent_dim = user_embeddings.shape
    user_embeddings = user_embeddings.unsqueeze(0) \
        .reshape(n_users, 1, latent_dim) \
        .repeat(1, n_movies, 1)
    distances = (user_embeddings - embeddings.unsqueeze(0).repeat(n_users, 1, 1)) \
        .pow(2) \
        .sum(dim=2)
    return torch.argsort(distances, dim=1, descending=True)[:, :n_recommendations]

def rate_user_recommendations(user_recommendations: list[int], user_masked_films: list[int], user_masked_ratings: list[int]):
    """Rate the recommendations for a single user.

    Args:
        user_recommendations (list[int]): n recommendations for a single user
        user_masked_films (list[int]): masked films for a single user 
        user_masked_ratings (list[int]): masked ratings for a single user (same order as user_masked_films)

    Returns:
        _type_: _description_
    """

    good_ratings = {film for film, rating in zip(user_masked_films, user_masked_ratings) if rating > 3}
    matches = [film for film in user_recommendations if film in good_ratings]
    return len(matches) / len(good_ratings)


def evaluate_batch(subset: pd.DataFrame, model: torch.nn.Sequential, n_recommendations: int, device: torch.device, n_movies: int, embeddings: torch.Tensor) -> float:
    """Evaluate a batch of user recommendations.

    Args:
        subset (pd.DataFrame): dataframe subset with columns 'film', 'rating' and 'is_masked'
        model (torch.nn.Sequential): model to use for recommendations
        n_recommendations (int): number of recommendations to make for each user
        device (torch.device): device to use for model
        n_movies (int): number of movies in the dataset
        embeddings (torch.Tensor): embeddings of the movies

    Returns:
        percentage of positive recommendations that are in the top n_recommendations of the model
    """
    batch = df_subset_to_test_batch(subset, n_movies)
    recommendations = recommend_n_movies(model, batch, embeddings, n_recommendations, device, n_movies)
    recommender_score = 0
    for user_recommendations, user_masked_films, user_masked_ratings in zip(recommendations, batch.masked_films, batch.masked_ratings):
        recommender_score += rate_user_recommendations([rec.item() for rec in user_recommendations], user_masked_films, user_masked_ratings)
    return recommender_score
