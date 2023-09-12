# Importieren der notwendigen Bibliotheken
import numpy as np
import pandas as pd
import polars as pl 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class Cosine_Similarity:
    
    def __init__(self):
        # Daten einlesen
        self.df = pd.read_csv(r"C:\\Users\\julia\\OneDrive\\Desktop\\netflix_recommendation\\Source\\data\\movie_data.csv", sep='|')
        # Spalten umbenennen
        columns = [...]
        self.df.columns = columns

        # Textvorverarbeitung und Erstellung einer TF-IDF-Matrix
        tfidf = TfidfVectorizer(stop_words='english')
        self.df['Overview'] = self.df['Overview'].fillna("")
        tfidf_matrix = tfidf.fit_transform(self.df['Overview'])

        # Kosinus-Ähnlichkeitsmatrix basierend auf den Übersichten erstellen
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # NaNs in Genre durch leere Strings ersetzen und Cluster erstellen
        self.df['Genre'] = self.df['Genre'].fillna('')
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
        genre_matrix = vectorizer.fit_transform(self.df['Genre'])
        kmeans = KMeans(n_clusters=3)
        self.genre_clusters = kmeans.fit_predict(genre_matrix)

        # Schauspieler und Regisseure in Sets umwandeln
        self.df['Actors'] = self.df['Actors'].apply(lambda x: set(x.split(',')) if isinstance(x, str) else set())
        self.df['Director'] = self.df['Director'].apply(lambda x: set(x.split(',')) if isinstance(x, str) else set())
        
 


    def get_movie_cos_scores(self, movie_id, bonus_actor=0.1, bonus_same_director=0.2):
        # Anpassung der Indizes
        idx = movie_id - 1
        # Ähnlichkeitswerte ermitteln
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Ermittlung des Genres und der Schauspieler des Ausgangsfilms
        genre_cluster = self.genre_clusters[idx]
        director1 = self.df['Director'].iloc[idx]
        actors1 = self.df['Actors'].iloc[idx]
        
        # Anpassung der Ähnlichkeitswerte basierend auf den Bonusregeln
        for i, _ in sim_scores:
            if self.genre_clusters[i] != genre_cluster:
                sim_scores[i] = (i, 0)
                continue

            common_actors = actors1.intersection(self.df['Actors'].iloc[i])
            common_directors = director1.intersection(self.df['Director'].iloc[i])
            
            if len(common_actors) > 0:
                sim_scores[i] = (i, sim_scores[i][1] + bonus_actor)
            if len(common_directors) > 0:
                sim_scores[i] = (i, sim_scores[i][1] + bonus_same_director)

        # Sortierung und Extraktion der Top 50 Filme
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_50_scores = sorted_scores[1:51]
        top_50_movie_ids = [x[0] + 1 for x in top_50_scores]
        
        return top_50_movie_ids

    def get_title(self, item_id, movie_titles):
        # Ermitteln des Filmtitels anhand der ID
        return movie_titles.filter(pl.col("film") == item_id)["title"].to_list()[0]

