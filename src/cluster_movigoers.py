import pandas as pd 
import numpy as np 
import sklearn as sk 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import itertools

genres = ["Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western"]
movies =  pd.read_csv('movies_sample.csv')
ratings =  pd.read_csv('ratings_sample.csv')

def ratings_per_genre(movies, ratings):
        a = movies[["movieId", "genres"]]
        ratings_and_genres = ratings.set_index('movieId').join(a.set_index('movieId'))
        genre_ratings = ratings_and_genres[ratings_and_genres["genres"].str.contains(genres[0])]
        genre_avg_per_user = genre_ratings[["userId","rating"]].groupby("userId").mean()
        for g in genres[1:]:
                genre_ratings2 = ratings_and_genres[ratings_and_genres["genres"].str.contains(g)]
                genre_avg_per_user2 = genre_ratings2[["userId","rating"]].groupby("userId").mean()
                genre_avg_per_user = genre_avg_per_user.join(genre_avg_per_user2, rsuffix=g)
      
        genre_avg_per_user = genre_avg_per_user.rename(index=str, columns={"rating": "rating"+genres[0]}).fillna(0)

        #plot 2
        pairs = itertools.combinations(genres, 2)

        for pair in pairs:
                print(pair)
                cluster_users(genre_avg_per_user,pair[0], pair[1])
        # X =genre_avg_per_user.values
        # Y = KMeans(n_clusters=2).fit_predict(X)
        # plt.scatter(genre_avg_per_user["ratingHorror"],genre_avg_per_user["ratingComedy"],c=Y,alpha=0.03)
        # plt.show()


def cluster_users(ratings_per_genre,genre1, genre2):
        # ratings_per_genre =  pd.read_csv("ratings_per_genre_sample.csv")
        X =ratings_per_genre.values
        Y = KMeans(n_clusters=2).fit_predict(X)
        x_random_error = np.random.normal(ratings_per_genre["rating"+str(genre1)], 0.2)
        y_random_error = np.random.normal(ratings_per_genre["rating"+str(genre2)], 0.2)
        x = ratings_per_genre["rating"+str(genre1)]+x_random_error
        y = ratings_per_genre["rating"+str(genre2)]+y_random_error
        plt.scatter(x,y,c=Y)
        plt.xlabel("Avg "+genre1+" rating")
        plt.ylabel("Avg "+genre2+" rating")
        plt.title(genre1+" vs. "+genre2)
        plt.show()
        
# cluster_users("Horror", "Comedy")
ratings_per_genre(movies,ratings)