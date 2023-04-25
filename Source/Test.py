import polars as pl

data = pl.read_csv("/Dataset/movie_titles.csv")

print(data)
