import torch
from .data_containers import TestBatch


def get_embeddings(n_movies, model, device) -> torch.Tensor:
    model.eval()
    model.to(device)
    rating_matrix = torch.eye(n_movies)*5
    with torch.no_grad():
        return model.encode(rating_matrix.to(device).float()).to("cpu")
    
def df_subset_to_test_batch(subset, n_movies) -> TestBatch:
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

def recommend_n_movies(model, test_batch, embeddings, n_recommendations, device, n_movies):
    user_embeddings = model.encode(test_batch.input_vector.to(device).float()).to("cpu")
    n_users, latent_dim = user_embeddings.shape
    user_embeddings = user_embeddings.unsqueeze(0) \
        .reshape(n_users, 1, latent_dim) \
        .repeat(1, n_movies, 1)
    distances = (user_embeddings - embeddings.unsqueeze(0).repeat(n_users, 1, 1)) \
        .pow(2) \
        .sum(dim=2)
    return torch.argsort(distances, dim=1, descending=True)[:, :n_recommendations]

def rate_user_recommendations(user_recommendations, user_masked_films, user_masked_ratings):
    good_ratings = {film for film, rating in zip(user_masked_films, user_masked_ratings) if rating > 3}
    matches = [film.item() for film in user_recommendations if film.item() in good_ratings]
    return len(matches) / len(good_ratings)


def evaluate_batch(subset, model, n_recommendations, device, n_movies, embeddings):
    batch = df_subset_to_test_batch(subset, n_movies)
    recommendations = recommend_n_movies(model, batch, embeddings, n_recommendations, device, n_movies)
    recommender_score = 0
    for user_recommendations, user_masked_films, user_masked_ratings in zip(recommendations, batch.masked_films, batch.masked_ratings):
        recommender_score += rate_user_recommendations(user_recommendations, user_masked_films, user_masked_ratings)
    return recommender_score
