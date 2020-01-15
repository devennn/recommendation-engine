import pickle as pkl
from pathlib import Path
import os
import pandas as pd


def load_data():
    path  = Path('.').parent.absolute()
    path1 = os.path.join(path, 'result', 'user_features.dat')
    path2 = os.path.join(path, 'result', 'products_features.dat')
    path3 = os.path.join(path, 'result', 'predicted_ratings.dat')
    path4 = os.path.join(path, 'dataset', 'movies.csv')

    with open(path1, 'rb') as f:
        U = pkl.load(f)

    with open(path2, 'rb') as f:
        M = pkl.load(f)

    with open(path3, 'rb') as f:
        predicted_ratings = pkl.load(f)

    return U, M, predicted_ratings, pd.read_csv(path4, index_col='movie_id')

def recommend_existing():
    # Load prediction rules from data files
    U, M, predicted_ratings, movies_df = load_data()

    print("Enter a user_id to get recommendations (Between 1 and 100):")
    user_id_to_search = int(input())

    print("Movies we will recommend:")

    user_ratings = predicted_ratings[user_id_to_search - 1]
    movies_df['rating'] = user_ratings
    movies_df = movies_df.sort_values(by=['rating'], ascending=False)

    print(movies_df[['title', 'genre', 'rating']].head(5))
