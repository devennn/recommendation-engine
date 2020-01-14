import numpy as np
import pandas as pd
import pickle
import matrix
from .utils import *

def train_existing_user():

    raw_dataset_df = load_data()

    # Convert the running list of user ratings into a matrix
    ratings_df = pd.pivot_table(raw_dataset_df,
                                    index='user_id',
                                    columns='movie_id',
                                    aggfunc=np.max
                                )

    # Apply matrix factorization to find the latent features
    U, M = matrix.lrank_mtrx_fctrize(ratings_df.values,
                                        num_features=15,
                                        regularization_amount=0.1
                                    )

    # Find all predicted ratings by multiplying U and M
    predicted_ratings = np.matmul(U, M)
    save_result(U, M, predicted_ratings)
