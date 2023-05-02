import gc
import polars as pl


def clean_data():
    print("Cleaning data...")
    connection_string = 'sqlite://netflix.db'

    data = pl.read_database("SELECT * from netflix_data", connection_string)
    movies = pl.read_csv("data/movie_titles.csv", ignore_errors=True, new_columns=["film", "year", "title"], has_header=False)

    data = data.with_columns([
        pl.col("date") \
        .str.strip() \
        .str.strptime(pl.Date, "%Y-%m-%d") \
        .alias("date")
    ])

    popular_movies = data["film"] \
        .value_counts() \
        .filter(pl.col("counts") > 10_000) \
        .select(["film"])

    data = data.filter(pl.col("film").is_in(popular_movies["film"]))
    movies = movies \
        .filter(pl.col("film") \
        .is_in(popular_movies["film"]))

    movies = movies.with_columns([
        pl.Series(name="new_id", values=range(len(movies)))
    ])

    data = data.join(movies.select(["film", "new_id"]), on="film") \
        .with_columns([
            pl.col("new_id").alias("film")
        ]) \
        .drop(columns=["new_id"])


    users_with_enough_ratings = data["user"] \
        .value_counts() \
        .filter(pl.col("counts") > 5) \
        .select(["user"])

    data = data.filter(pl.col("user").is_in(users_with_enough_ratings["user"]))

    user_ratings = data.groupby("user") \
        .agg(pl.col("rating").mean() \
        .alias("mean_rating"))

    data = data.join(user_ratings, on="user") \
        .with_columns([
            (pl.col("rating") - pl.col("mean_rating")) \
                .alias("relative_rating")
        ]) \
        .drop(columns=["rating", "mean_rating"])

    user_matrix = data.groupby("user") \
        .agg(pl.col("film"), pl.col("relative_rating"))

    del data
    del popular_movies
    del users_with_enough_ratings

    gc.collect()

    print("Data cleaned!")

    return user_matrix, movies