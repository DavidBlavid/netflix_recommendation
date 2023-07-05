import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class Cosine_Similarity_Cluster_Matrix:

    def __init__(self):
         # Einlesen der Daten
        self.df = pd.read_csv(r"C:\\Users\\julia\\OneDrive\\Desktop\\netflix_recommendation\\Source\\data\\movie_data.csv", sep='|')
        columns = ['index', 'Title', 'Year', 'Response', 'Rated', 'Released', 'Runtime', 'Genre', 'Director', 'Writer', 'Actors', 'Overview', 'Language', 'Country', 'Awards', 'Poster', 'Ratings', 'Metascore', 'imdbRating', 'imdbVotes', 'imdbID', 'Type', 'DVD', 'BoxOffice', 'Production', 'Website']
        self.df.columns = columns

        # Clusterinformationen laden
        df_cluster = pd.read_csv(r"C:\\Users\\julia\\OneDrive\\Desktop\\netflix_recommendation\\Source\\data\\movie_data_with_clusters2.csv", sep='|')

        # Füge die Clusterinformationen zur vollständigen Tabelle hinzu
        self.df['GenreCluster'] = df_cluster['GenreCluster']

        self.df_relevant = self.df[self.df['Response'] == True]
        self.df_irrelevant = self.df[self.df['Response'] == False]

        # Füllen Sie fehlende Werte mit -1 aus, bevor Sie der Tabelle hinzufügen
        self.df['GenreCluster'] = self.df['GenreCluster'].fillna(-1)

        # Konvertieren Sie die Clusterlabels zu Ganzzahlen
        self.df['GenreCluster'] = self.df['GenreCluster'].astype(int)

        # Textvorverarbeitung und TF-IDF-Matrix erstellen
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.df_relevant['Overview'] = self.df_relevant['Overview'].fillna("")
        
        self.create_cluster_matrix(0)
        self.create_cluster_matrix(1)
        self.create_cluster_matrix(2)

    def create_cluster_matrix(self, cluster_id):
        df_cluster = self.df_relevant[self.df_relevant['GenreCluster'] == cluster_id]
        tfidf_matrix_cluster = self.tfidf.fit_transform(df_cluster['Overview'])
        cosine_sim_cluster = cosine_similarity(tfidf_matrix_cluster, tfidf_matrix_cluster)
        
        setattr(self, f"cosine_sim_cluster_{cluster_id}", cosine_sim_cluster)
        setattr(self, f"df_cluster_{cluster_id}", df_cluster)


    def get_movie_cos_scores(self, movie_id):
        idx = movie_id - 1 

        if self.df.loc[idx, 'Response'] == False:
            return [(i, 0) for i in range(len(self.df))]

        cluster = self.df.loc[idx, 'GenreCluster']

        cosine_sim_matrix = getattr(self, f"cosine_sim_cluster_{cluster}")
        df_cluster = getattr(self, f"df_cluster_{cluster}")

        cluster_indices = df_cluster.index.values

        idx_cluster = np.where(cluster_indices == idx)[0][0]
        sim_scores = list(enumerate(cosine_sim_matrix[idx_cluster]))
        sim_scores = [(cluster_indices[i], score) for i, score in sim_scores]


        all_scores = [(i, 0) if i not in cluster_indices else (i, sim_scores[np.where(cluster_indices == i)[0][0]][1]) for i in range(len(self.df))]

        '''# Erstellen Sie ein DataFrame aus den all_scores-Daten
        df_scores = pd.DataFrame(all_scores, columns=['film', 'cosine_similarity'])


        # Konvertieren Sie das DataFrame in eine Sparse-Matrix
        sparse_matrix = sparse.csr_matrix(df_scores.values)

        # Erstellen Sie ein neues Sparse-DataFrame aus der Sparse-Matrix
        sparse_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=df_scores.columns)
        # Konvertieren Sie das DataFrame in ein sparse DataFrame
        sparse_df = df_scores.to_sparse(fill_value=0)

        # Konvertieren Sie das sparse DataFrame in eine CSR-Matrix
        cos_scores_sparse = sparse_df.to_coo().tocsr()'''
    

        return all_scores	
    
    def save_to_csv(self, cluster_id):
        cosine_sim_matrix = getattr(self, f"cosine_sim_cluster_{cluster_id}")
        np.savetxt(rf"C:\Users\julia\OneDrive\Desktop\netflix_recommendation\cosine_sim_cluster_{cluster_id}.csv", cosine_sim_matrix, delimiter=",")

        df_cluster = getattr(self, f"df_cluster_{cluster_id}")
        df_cluster['index'].to_csv(rf"C:\Users\julia\OneDrive\Desktop\netflix_recommendation\movie_indices_cluster_{cluster_id}.csv", index=False)

    @staticmethod
    def read_from_csv(cluster_id):
        cosine_sim_matrix = np.loadtxt(rf"C:\Users\julia\OneDrive\Desktop\netflix_recommendation\cosine_sim_cluster_{cluster_id}.csv", delimiter=",")
        movie_indices = pd.read_csv(rf"C:\Users\julia\OneDrive\Desktop\netflix_recommendation\movie_indices_cluster_{cluster_id}.csv")
        movie_indices = movie_indices.squeeze()  # Umwandlung des DataFrame in eine Series
        return cosine_sim_matrix, movie_indices



    
    def get_cosine_sim_scores(self, movie_id):
        idx = movie_id - 1 

        if self.df.loc[idx, 'Response'] == False:
            return pd.Series(np.zeros(len(self.df)), index=self.df.index)

        cluster = self.df.loc[idx, 'GenreCluster']

        cosine_sim_matrix, movie_indices = self.read_from_csv(cluster)

        idx_cluster = np.where(movie_indices == idx)[0][0]
        sim_scores = pd.Series(cosine_sim_matrix[idx_cluster], index=movie_indices)

        all_scores = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        all_scores.update(sim_scores)

        return all_scores


# Erstellen Sie das Objekt Ihrer Klasse
cos_sim = Cosine_Similarity_Cluster_Matrix()

# Speichern Sie die Matrizen und Indizes für die benötigten Cluster in CSV-Dateien
# Sie müssen dies nur einmal für jeden Cluster tun
'''cos_sim.save_to_csv(0)
cos_sim.save_to_csv(1)
cos_sim.save_to_csv(2)'''

# Rufen Sie die Methode get_cosine_sim_scores() auf, um die Kosinusähnlichkeitswerte für den Film mit der Movie_ID 465 zu erhalten
cos_sim_scores = cos_sim.get_cosine_sim_scores(268)

# Drucken Sie die Kosinusähnlichkeitswerte aus
print(cos_sim_scores)


