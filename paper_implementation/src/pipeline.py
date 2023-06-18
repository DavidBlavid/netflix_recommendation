import gc
import polars as pl
import datetime as dt

def clean_data():
    print("Cleaning data...")
    connection_string = 'sqlite://data/netflix.db'

    data = pl.read_database("SELECT * from netflix_data", connection_string)
    movies = pl.read_csv("data/movie_titles.csv", ignore_errors=True, new_columns=["film", "year", "title"]) \
        .with_columns([
            pl.col("film").cast(pl.Int64, strict=False)
        ])
    
    data = data.with_columns([
        (pl.col("film")-1).alias("film")
    ])

    movies = movies.with_columns([
       ( pl.col("film") - 1).alias("film")
    ])

    data = data.with_columns([
        pl.col("date") \
        .str.strip() \
        .str.strptime(pl.Date, "%Y-%m-%d") \
        .alias("date")
    ])

    users_with_enough_ratings = data["user"] \
        .value_counts() \
        .filter(pl.col("counts") > 5) \
        .select(["user"])

    data = data.filter(pl.col("user").is_in(users_with_enough_ratings["user"]))
    return data, movies


def split_data(data: pl.DataFrame, date: dt.datetime):
    train = data.filter(pl.col("date") < date) \
        .groupby("user") \
        .agg(pl.col("film"), pl.col("rating"))
    
    test = data.filter((pl.col("date") >= date) & (pl.col("rating") >= 4))
    test_data_users = test.select(pl.col("user").unique())
    test = data.filter(pl.col("user").is_in(test_data_users["user"]))
    test = test.with_columns([
        (pl.col("date") >= date).alias("is_masked")]) \
        .groupby("user") \
        .agg(pl.col("film"), pl.col("rating"), pl.col("is_masked"))
    return train.to_pandas(), test.to_pandas()
