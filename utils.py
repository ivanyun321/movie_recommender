import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_similarities_on_the_fly(df, movie_name, tfidf_matrix):
    """
    Compute cosine similarities between the vector for a specific movie
    and all other movies, without precomputing the full similarity matrix.
    
    Parameters:
        movie_index (int): The index of the queried movie in the tfidf_matrix.
        tfidf_matrix (sparse matrix): TF-IDF feature matrix for all movies.
    
    Returns:
        similarities (np.array): 1D array of cosine similarities.
    """
    # Get the query movie's TF-IDF vector
    indices = df.index[df['primaryTitle'] == movie_name].tolist()

    if not indices:
        print("Movie not found in df.")
        return
    else:
        movie_index = indices[0]

    query_vector = tfidf_matrix[movie_index]
    
    # Compute cosine similarities using sparse dot product
    # Since the vectors are normalized, dot product equals cosine similarity.
    cosine_similarities = (tfidf_matrix * query_vector.T).toarray().ravel()
    
    return cosine_similarities

def normalize_ratings(ratings, max_rating=10):
    """Normalize ratings to a 0-1 scale."""
    return ratings / max_rating

def get_recommendations(movie_title, df, tfidf_matrix, alpha=0.5, top_n=5):
    """
    Compute combined scores for content similarity and normalized average ratings,
    then return the top_n recommendations.
    
    Parameters:
      movie_title (str): The title of the movie to base recommendations on.
      df (DataFrame): Merged DataFrame containing movie details and ratings.
      tfidf_matrix (sparse matrix): TF-IDF feature matrix computed from movie features.
      alpha (float): Weight for content similarity (between 0 and 1).
      top_n (int): Number of recommendations to return.
      
    Returns:
      DataFrame: A subset of df with recommended movies.
    """
    # Look up the movie index by title
    indices = df.index[df['primaryTitle'] == movie_title].tolist()
    if not indices:
        print("Movie not found.")
        return None
    movie_index = indices[0]
    
    # Get the query movie's TF-IDF vector
    query_vector = tfidf_matrix[movie_index]
    
    # Compute cosine similarity using sparse dot product (assuming L2-normalized vectors)
    content_similarities = (tfidf_matrix @ query_vector.T).toarray().ravel()
    
    # Convert and normalize the average ratings (assuming scale is 0-10)
    try:
        ratings_float = df['averageRating'].astype(float)
    except Exception as e:
        print("Error converting averageRating to float:", e)
        return None
    normalized_ratings = normalize_ratings(ratings_float)
    
    # Combine scores: weight content similarity and normalized ratings
    combined_scores = alpha * content_similarities + (1 - alpha) * normalized_ratings
    
    # Exclude the queried movie itself from recommendations
    combined_scores[movie_index] = -np.inf
    
    # Get indices of the top_n movies based on the combined score
    recommended_indices = np.argsort(combined_scores)[::-1][:top_n]
    
    return df.iloc[recommended_indices][['primaryTitle', 'startYear', 'averageRating']]

def get_recommendations_on_the_fly(movie_index, tfidf_matrix, df, top_n=5):
    # Compute cosine similarities for the queried movie
    similarities = get_similarities_on_the_fly(df, movie_index, tfidf_matrix)
    
    # Get indices of the top_n most similar movies (ignoring the movie itself)
    similar_indices = similarities.argsort()[::-1][1:top_n+1]
    
    # Retrieve the movie titles or any other details from the DataFrame
    return df.iloc[similar_indices][['primaryTitle', 'startYear']]

def compute_weighted_rating(row, m, C):
    """
    Compute the weighted rating for a movie using the Bayesian formula.
    
    Parameters:
        row (Series): A row from the DataFrame containing 'averageRating' and 'numVotes'.
        m (float): The vote count threshold.
        C (float): The overall average rating (prior).
    
    Returns:
        float: The weighted rating.
    """
    R = row['averageRating']
    v = row['numVotes']
    return (v / (v + m)) * R + (m / (v + m)) * C

def get_recommendations_with_prior(movie_title, df, tfidf_matrix, alpha=0.5, top_n=5):
    """
    Compute combined scores for content similarity and weighted ratings,
    then return the top_n recommendations.
    
    Parameters:
      movie_title (str): The title of the movie to base recommendations on.
      df (DataFrame): Merged DataFrame containing movies with columns 'primaryTitle',
                      'startYear', 'weightedRating', etc.
      tfidf_matrix (sparse matrix): TF-IDF feature matrix computed from movie features.
      alpha (float): Weight for content similarity (between 0 and 1).
      top_n (int): Number of recommendations to return.
      
    Returns:
      DataFrame: A subset of df with recommended movies.
    """
    # Look up the movie index by title.
    indices = df.index[df['primaryTitle'] == movie_title].tolist()
    if not indices:
        print("Movie not found.")
        return None
    movie_index = indices[0]
    
    # Get the query movie's TF-IDF vector.
    query_vector = tfidf_matrix[movie_index]
    
    # Compute cosine similarity on the fly.
    content_similarities = (tfidf_matrix @ query_vector.T).toarray().ravel()
    
    # Normalize the weighted ratings (assuming scale is 0-10)
    try:
        weighted_ratings = df['weightedRating'].astype(float)
    except Exception as e:
        print("Error converting weightedRating to float:", e)
        return None
    normalized_weighted_ratings = normalize_ratings(weighted_ratings)
    
    # Combine the scores: alpha for content similarity and (1 - alpha) for normalized weighted rating.
    combined_scores = alpha * content_similarities + (1 - alpha) * normalized_weighted_ratings
    
    # Exclude the queried movie itself.
    combined_scores[movie_index] = -np.inf
    
    # Get indices of the top_n movies.
    recommended_indices = np.argsort(combined_scores)[::-1][:top_n]
    
    return df.iloc[recommended_indices][['primaryTitle', 'startYear', 'weightedRating']]

def compute_weighted_rating(row, m, C):
    """
    Compute the weighted rating for a movie.
    
    Parameters:
      row (Series): A row of the DataFrame, containing 'averageRating' and 'numVotes'.
      m (float): The vote threshold.
      C (float): The overall mean rating (prior).
    
    Returns:
      float: The weighted rating.
    """
    R = row['averageRating']
    v = row['numVotes']
    return (v / (v + m)) * R + (m / (v + m)) * C
