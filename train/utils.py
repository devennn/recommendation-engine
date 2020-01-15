import pandas as pd
import pickle as pkl
from pathlib import Path
import os

def load_data():
    # Load user ratings
    path = Path('.').parent.absolute()
    path = os.path.join(path, 'dataset', 'movie_ratings_data_set.csv')
    return pd.read_csv(path)

def save_result(U, M, predicted_ratings):

    # Load user ratings
    path = Path('.').parent.absolute()
    path1 = os.path.join(path, 'result', 'user_features.dat')
    path2 = os.path.join(path, 'result', 'products_features.dat')
    path3 = os.path.join(path, 'result', 'predicted_ratings.dat')

    pkl.dump(U, open(path1, "wb"))
    pkl.dump(M, open(path2, "wb"))
    pkl.dump(predicted_ratings, open(path3, "wb"))
