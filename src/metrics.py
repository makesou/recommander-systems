import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
import math

def precision_at_k(df: pd.DataFrame, k: int = 5, user_id: int = None):
    """
    Compute Precision@K metric

    Parameters
    ---
    df: pd.DataFrame
        Dataframe containing the predictions of any model. 
        It must contains columns `user_id` and `like`
    k: int
        Number of items to retrieve for each user
        default: 5
    user_id: int
        Compute Precision@K for one specific user
        default: None

    Returns  
    ---
    float
        metric computed on the dataframe df
    """
    def compute_one(user_id: int):
        user_top_videos = df[df['user_id'] == user_id].head(k)
        precision = len(user_top_videos[user_top_videos['like'] == 1]) / k
        return precision
    
    if user_id is not None:
        return compute_one(user_id)

    precisions = []
    for user_id in df['user_id'].unique():
        precisions.append(compute_one(user_id))

    return np.mean(precisions)


def ndcg_at_k(df: pd.DataFrame, k: int = 5, user_id: int = None):
    """
    Compute NDCG@K metric

    Parameters
    ---
    df: pd.DataFrame
        Dataframe containing the predictions of any model. 
        It must contains columns `user_id` and `like`
    k: int
        Number of items to retrieve for each user
        default: 5
    user_id: int
        Compute for one specific user
        default: None

    Returns  
    ---
    float
        metric computed on the dataframe df
    """

    def compute_one(user_id: int):
        user_top_videos = df[df['user_id'] == user_id].head(k)
        ratings = user_top_videos['like']
        ideal_ratings = ratings.sort_values(ascending=False)
        dcg = np.sum(ratings / np.log2(np.arange(2, ratings.size + 2)))
        idcg = np.sum(np.array(ideal_ratings) / np.log2(np.arange(2, len(ideal_ratings) + 2)))
        ndcg = dcg / idcg if idcg > 0 else 0
        return ndcg
        
    if user_id is not None:
        return compute_one(user_id)

    ndcgs = []
    for user_id in df['user_id'].unique():
        ndcgs.append(compute_one(user_id))

    return np.mean(ndcgs)

def inter_list_diversity_at_k(df: pd.DataFrame, content_columns: list[str] = [], k: int = 5, user_id: int = None):
    """
    Compute ILD@K metric. Measure the diversity of the prediction of any model.

    Parameters
    ---
    df: pd.DataFrame
        Dataframe containing the predictions of any model. 
        It must contains columns `user_id` and `like`
    content_columns: list[str]
        Columns present in `df` to use to compute similarities between predictions
        default: []
    k: int
        Number of items to retrieve for each user
        default: 5
    user_id: int
        Compute for one specific user
        default: None

    Returns  
    ---
    float
        metric computed on the dataframe df
    """

    scaler = StandardScaler().fit(df[content_columns])

    def compute_one(user_id: int):
        user_top_videos = df[df['user_id'] == user_id].head(k)
        top_videos_content = scaler.transform(user_top_videos[content_columns])
        distances = [
            1 - cosine(top_videos_content[i], top_videos_content[j])
            for i in range(len(top_videos_content)) for j in range(i + 1, len(top_videos_content))
        ]
        return 1 - np.mean(distances)
        
    
    if user_id is not None:
        return compute_one(user_id)

    diversities = []
    for user_id in df['user_id'].unique():
        diversities.append(compute_one(user_id))

    return np.mean(diversities)