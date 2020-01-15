import pickle as pkl
import pandas as pd
from pathlib import Path
import os

def load_data():
    path = Path('.').parent.absolute()
    path1 = os.path.join(path, 'result', 'means.dat')
    print(path1)

    with open(path1, 'rb') as f:
        means = pkl.load(f)

    path2 = os.path.join(path, 'dataset', 'movies.csv')
    movies_df = pd.read_csv(path2, index_col='movie_id')

    return means, movies_df

def recommend_new():
    # Load prediction rules from data files and movies title
    means, movies_df = load_data()

    # Just use the average movie ratings directly as the user's predicted ratings
    user_ratings = means

    print("Movies we will recommend:")

    movies_df['rating'] = user_ratings
    movies_df = movies_df.sort_values(by=['rating'], ascending=False)

    print(movies_df[['title', 'genre', 'rating']].head(5))
