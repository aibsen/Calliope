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
movies =  pd.read_csv('../ml-20m/movies.csv')
ratings =  pd.read_csv('../ml-20m/ratings.csv')


# movies =  pd.read_csv('movies_sample.csv')
# ratings =  pd.read_csv('ratings_sample.csv')

def ratings_per_genre(movies, ratings):
        a = movies[["movieId", "genres"]]
        ratings_and_genres = ratings.set_index('movieId').join(a.set_index('movieId'))
        genre_ratings = ratings_and_genres[ratings_and_genres["genres"].str.contains(genres[0])]
        genre_avg_per_user = genre_ratings[["userId","rating"]].groupby("userId").mean()
        for g in genres[1:]:
                genre_ratings2 = ratings_and_genres[ratings_and_genres["genres"].str.contains(g)]
                genre_avg_per_user2 = genre_ratings2[["userId","rating"]].groupby("userId").mean()
                genre_avg_per_user = genre_avg_per_user.join(genre_avg_per_user2, rsuffix=g)
      
        genre_avg_per_user = genre_avg_per_user.rename(index=str, columns={"rating": "rating"+genres[0]})

        plot_clusters(genre_avg_per_user)

def plot_clusters(genre_avg_per_user):
        pairs = itertools.combinations(genres, 2)
        for pair in pairs:
                for n_clusters in range(2,7):
                        print(pair)
                        print(n_clusters)
                        index0 = "rating"+pair[0]
                        index1 = "rating"+pair[1]
                        ratings_per_genre = genre_avg_per_user[[index0,index1]].dropna()
                        cluster_users(n_clusters,ratings_per_genre, index0, index1)


def cluster_users(n,ratings_per_genre,index0, index1):
        X =ratings_per_genre.values
        try:
                Y = KMeans(n_clusters=n).fit_predict(X)
                x_random_error = np.random.normal(0, 0.3)
                y_random_error = np.random.normal(0, 0.3)
                x = ratings_per_genre[index0]+x_random_error
                y = ratings_per_genre[index1]+y_random_error
                plt.scatter(x,y,c=Y)
                plt.xlabel("Avg "+index0.split("rating")[1]+" rating")
                plt.ylabel("Avg "+index1.split("rating")[1]+" rating")
                plt.show()
        except:
                print("error!")

def time_vs_rating(ratings_and_genres,genre, frec_or_avg):
        genred_movies = ratings_and_genres[ratings_and_genres["genres"].str.contains(genre)]
        
        plt.title(genre + " movies  during the year")
        #one movie 
        #if flag is true, plot frequency of ratings
        if frec_or_avg:       
                ratings_per_month = genred_movies.groupby(["movieId","timestamp"])["rating"].count()
                plt.ylabel("Frequency of ratings")

        #if flag is false, plot avg of ratings
        else :
                ratings_per_month = genred_movies.groupby(["movieId","timestamp"])["rating"].mean()
                plt.ylabel("Avg rating")


        movie_ids = list(set( map( lambda x : x[0] ,ratings_per_month.keys().values)))
        plt.xlabel("Month")
        for mid in movie_ids:
                print(mid)
                movie_ratings_per_month = ratings_per_month.loc[mid]
                movie_ratings_per_month[movie_ratings_per_month>0]
                # print(movie_ratings_per_month)
                x = movie_ratings_per_month.keys().values
                y = movie_ratings_per_month.values
                plt.scatter(x,y)
        plt.show()

def plot_ratings_per_month():
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'],unit='s')
        ratings['timestamp'] = pd.DatetimeIndex(ratings['timestamp']).month
        a = movies[["movieId", "genres"]]
        ratings_and_genres = ratings.set_index('movieId').join(a.set_index('movieId'))
        for g in genres:
                #frequency
                time_vs_rating(ratings_and_genres,g,True)
                #average
                time_vs_rating(ratings_and_genres,g,False)


def frequency_over_time():
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'],unit='s')
        ratings['timestamp'] = pd.DatetimeIndex(ratings['timestamp']).month
        a = movies[["movieId", "genres"]]
        ratings_and_genres = ratings.set_index('movieId').join(a.set_index('movieId'))
        # plt.title("movies during the year")
        ratings_per_month = ratings_and_genres.groupby(["movieId","timestamp"])["rating"].count()
        plt.ylabel("Frequency of ratings")
        movie_ids = list(set( map( lambda x : x[0] ,ratings_per_month.keys().values)))
        plt.xlabel("Month")
        for mid in movie_ids[100]:
                print(mid)
                movie_ratings_per_month = ratings_per_month.loc[mid]
                movie_ratings_per_month[movie_ratings_per_month>0]
                # print(movie_ratings_per_month)
                x = movie_ratings_per_month.keys().values
                y = movie_ratings_per_month.values
                plt.scatter(x,y)
        plt.show()

frequency_over_time()