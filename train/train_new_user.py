import numpy as np
import pandas as pd
import matrix
from .utils import *

def train_new_user():
    raw_dataset_df = load_data()

    # Convert the running list of user ratings into a matrix
    ratings_df = pd.pivot_table(raw_dataset_df,
                                    index='user_id',
                                    columns='movie_id',
                                    aggfunc=np.max
                                )

    # Normalize the ratings (center them around their mean)
    print(ratings_df)
    normalized_ratings, means = matrix.norm_rtngs(ratings_df.as_matrix())

    # Apply matrix factorization to find the latent features
    U, M = matrix.lrank_mtrx_fctrize(normalized_ratings,
                                        num_features=11,
                                        regularization_amount=1.1
                                    )

    # Find all predicted ratings by multiplying U and M
    predicted_ratings = np.matmul(U, M)

    # Add back in the mean ratings for each product to de-normalize the predicted results
    predicted_ratings = predicted_ratings + means

    save_result(U, M, predicted_ratings)

    path = os.path.join(Path('.').parent.absolute(), 'result', 'means.dat')
    with open(path, 'wb') as f:
        pkl.dump(means, f)
    path = ''
