import pandas as pd
import numpy as np
import math 
import sklearn

#/////////////////////////////ADDITIONAL DATA CLEANING AND PREPARATION FOR COLLABORATIVE FILTERING/////////////////////////////////////////////

def prepare_for_collab_filtering():
    #Read in the cleaned dataset
    complete = pd.read_csv("cleaned_dataset/cleanedOntario10.csv") 

    # 2. Collaborative Filtering
    #Drop all columns except for business id, user id, and user rating 
    df = complete.filter(['business_id','user_id', 'review_stars'])

    #https://www.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/

    n_users = df.user_id.unique().shape[0]
    n_items = df.business_id.unique().shape[0]
    print("Total users: " + str(n_users) + " Total items: " + str(n_items))

    #Create integer user id and business id keys, which will be used to index the Numpy matrix
    df['user_id_key'] = pd.factorize(df.user_id)[0]
    df['business_id_key'] = pd.factorize(df.business_id)[0]

    #Rearrange the columns
    df = df[["user_id_key", "business_id_key", "review_stars", "user_id", "business_id"]]

    #Export data to CSV
    #csv_name = "collabRatingsKeyed.csv"
    #df.to_csv(csv_name, index=False)
    return df

#//////////////////////////////////////////////////////////////////////////

#Construct user-item matrix
def construct_ratings_matrix(df):
    n_users = df.user_id.unique().shape[0]
    n_items = df.business_id.unique().shape[0]
    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3]
    return ratings

#Assess the sparsity of constructed matrix - used during data wrangling and performance analysis. Even when restricting datset to users with 10+ reviews, observed 0.35% sparsity.

#sparsity = float(len(ratings.nonzero()[0]))
#sparsity /= (ratings.shape[0] * ratings.shape[1])
#sparsity *= 100
#print('Sparsity: {:4.2f}%'.format(sparsity))

#Generate training and test sets for evaluating recommender performance
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=1, 
                                        replace=False)
        train[user, test_ratings] = 0
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

from sklearn.metrics import mean_squared_error
#Compute root mean squared error
def get_rmse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return math.sqrt(mean_squared_error(pred, actual))

from sklearn.metrics import pairwise_distances
#Memory-based approach following algorithm presented by #https://www.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/
def memory_based_collab(ratings):

    #Intialize training and test sets for performance analysis
    train, test = train_test_split(ratings)

    print("Computing cosine similarities...")
    #Construct matrix of similarity values - cosine metric is used here
    cosine_sim = 1-pairwise_distances(ratings, metric="cosine")

    print("Computing prediction matrix...")
    def prediction_matrix(ratings, similarity_matrix):
        #Compute prediction matrix - due to extreme sparsity of the dataset and to reduce computation time, computed over all neighbors
        return similarity_matrix.dot(ratings) / np.array([np.abs(similarity_matrix).sum(axis=1)]).T

    user_item_predictions = prediction_matrix(ratings, cosine_sim)

    #Assess recommender performance 
    user_item_predictions = prediction_matrix(train, cosine_sim)
    print('Memory-based Collaborative Filtering RMSE (User-Item): ' + str(get_rmse(user_item_predictions, test)))

    return user_item_predictions

def memory_based_prediction(user_id, prediction_matrix):  
    print("Prediction for user " + str(user_id))
    
    #Sort items by prediction score
    sorted_items = np.sort(prediction_matrix[user_id])[::-1]

    #Extract the 5 restaurants with top prediction score
    top5 = sorted_items[:5]
    print("Top 5 prediction scores: ", top5)

    business_keys = []
    for restaurant in top5:
        business = np.where(prediction_matrix[user_id] == restaurant)
        #Break ties if multiple restaurants have same prediction score value
        business_keys.append(business[0][0])

    print("Recommended businesses (keyed)", business_keys)
    return business_keys

from scipy.sparse.linalg import svds
#SVD - model-based collaborative filtering using matrix factorization: following algorithm presented by https://beckernick.github.io/matrix-factorization-recommender/
def SVD_matrix_manip(ratings, k):
    #Normalize the data by each user's mean rating
    user_ratings_mean = np.mean(ratings, axis = 1)
    R_demeaned = ratings - user_ratings_mean.reshape(-1, 1)

    #Intialize model matrix with k = number of latent factors

    print("Computing prediction matrix by SVD...")
    U, sigma, Vt = svds(R_demeaned, k)
    #Convert output into numpy matrix, with diagonalization
    sigma = np.diag(sigma)
    #Output prediction matrix - taking into account normalization of user ratings performed at the beginning
    prediction_matrix = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    return prediction_matrix

def model_based_collab(ratings):

    #Intialize training and test sets for performance analysis - disabled for live recommender
    #train, test = train_test_split(ratings)

    #Output prediction matrix - taking into account normalization of user ratings performed at the beginning
    prediction_matrix = SVD_matrix_manip(ratings, 50)

    #Assess recommender performance 
    #Perform hyperparmeter tuning on k to reduce RMSE, while taking compute time into consideration - k=50 was optimal
    #k_values = [10,20,50,100,200,400,600]
    #for k in k_values:
        #test_predictions = SVD_matrix_manip(train, k)
        #print('Model-based Collaborative Filtering RMSE (SVD): ' + str(get_rmse(test_predictions, test)) + "for k = " + str(k))

    return prediction_matrix

def model_based_prediction(user_id, prediction_matrix):
    #print("Prediction for user " + str(user_id))

    #Sort items by prediction score
    sorted_items = np.sort(prediction_matrix[user_id])[::-1]

    #Extract the restaurants with top prediction score
   
    top = sorted_items[:15]

    #print("Top prediction scores: ", top)

    business_keys = []
    for restaurant in top:
        business = np.where(prediction_matrix[user_id] == restaurant)
        #Break ties if multiple restaurants have same prediction score value
        if restaurant < 5:
            business_keys.append((business[0][0], restaurant))
        if len(business_keys) == 9:
            break

    #print("Recommended businesses (keyed)", business_keys)
    return business_keys

#Retrive Yelp data set user id to dataframe index mapping, and vice versa
def get_userid_from_index(index, df):
    return df[df["user_id_key"] == index]["user_id"].values[0]

def get_index_from_userid(title, df):
    return df[df["user_id"] == title]["user_id_key"].values[0]
