import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class Cosine_Similarity:

    def __init__(self):
        # Einlesen der Daten
        self.df = pd.read_csv(r"C:\\Users\\julia\\OneDrive\\Desktop\\netflix_recommendation\\Source\\data\\movie_data.csv", sep='|')
        columns = ['index', 'Title', 'Year', 'Response', 'Rated', 'Released', 'Runtime', 'Genre', 'Director', 'Writer', 'Actors', 'Overview', 'Language', 'Country', 'Awards', 'Poster', 'Ratings', 'Metascore', 'imdbRating', 'imdbVotes', 'imdbID', 'Type', 'DVD', 'BoxOffice', 'Production', 'Website']
        self.df.columns = columns

        # Textvorverarbeitung und TF-IDF-Matrix erstellen
        tfidf = TfidfVectorizer(stop_words='english')
        self.df['Overview'] = self.df['Overview'].fillna("")
        tfidf_matrix = tfidf.fit_transform(self.df['Overview'])

        # Overview Cosine Similarity Matrix berechnen
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Ersetzen Sie NaN durch einen leeren String
        self.df['Genre'] = self.df['Genre'].fillna('')

        # Create a count vectorizer object
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))

        # Fit and transform the genres to a 2-D matrix
        genre_matrix = vectorizer.fit_transform(self.df['Genre'])

        # Clustering nach Genres
        kmeans = KMeans(n_clusters=3)  # Anzahl der Cluster anpassen
        self.genre_clusters = kmeans.fit_predict(genre_matrix)

        # Calculate the genre cosine similarity
        #self.genre_cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

        # Preprocessing actors and directors into sets
        self.df['Actors'] = self.df['Actors'].apply(lambda x: set(x.split(',')) if isinstance(x, str) else set())
        self.df['Director'] = self.df['Director'].apply(lambda x: set(x.split(',')) if isinstance(x, str) else set())

        # Pre-calculate genre counts
        self.genre_counts = self.df['Genre'].value_counts()

    def get_movie_cos_scores(self, movie_id, bonus_actor=0.1, 
                            bonus_same_director=0.2, bonus_same_genre=0.2, min_same_genre_count=4):
        
        #Netflix Data und movie_titles fängt an bei 1 zu indizieren, movie_data bei 0, deswegen Notwendigkeit es anzupassen 
        idx = movie_id - 1 
        
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        genre_cluster = self.genre_clusters[idx]  # Cluster des Ausgangsfilms

        genre = self.df['Genre'].iloc[idx]
        director1 = self.df['Director'].iloc[idx]
        actors1 = self.df['Actors'].iloc[idx]

        for i, _ in sim_scores:
            if self.genre_clusters[i] != genre_cluster:  # Nur Filme aus demselben Genre-Cluster berücksichtigen
                sim_scores[i] = (i, 0)  # Setze den Score auf -1, um ihn unten zu sortieren
                continue

            common_actors = actors1.intersection(self.df['Actors'].iloc[i])
            common_directors = director1.intersection(self.df['Director'].iloc[i])
            genre2 = self.df['Genre'].iloc[i]

            #if self.genre_cosine_sim[idx][i] > 0.2:
            if len(common_actors) > 0:
                sim_scores[i] = (i, sim_scores[i][1] + bonus_actor)
            if len(common_directors) > 0:
                sim_scores[i] = (i, sim_scores[i][1] + bonus_same_director)
            if genre == genre2 and self.genre_counts[genre2]>= min_same_genre_count:
                sim_scores[i] = (i, sim_scores[i][1] + bonus_same_genre)
                
        return sim_scores
    
'''c = Cosine_Similarity()

print (c.get_movie_cos_scores(17500))
'''
