import pandas as pd

ROOT_PATH='../kuairec/data/'
FILE_NAMES = {
    'big_interactions': 'big_matrix',
    'small_interactions': 'small_matrix',
    'social': 'social_network',
    'item': 'item_categories',
    'item_daily': 'item_daily_features',
    'user': 'user_features',
    'caption': 'caption_category_readable'
}

def preprocess_social(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing applied to the social matrix
    
    Arguments:
        df (DataFrame): Social matrix
    
    Returns:
        df (DataFrame): the social matrix after applying the preprocessing step
    """
    df["friend_list"] = df["friend_list"].map(eval)
    return df


def preprocess_item(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing applied to the items matrix
    
    Arguments:
        df (DataFrame): Items matrix
    
    Returns:
        df (DataFrame): the items matrix after applying the preprocessing step
    """
    df["feat"] = df["feat"].map(eval)
    return df


def preprocess_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing applied to the interactions matrices (small and big ones)
    
    Arguments:
        df (DataFrame): Interactions matrix
    
    Returns:
        df (DataFrame): the given matrix after applying the preprocessing step
    """
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def preprocess_captions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing applied to the captions matrix
    
    Arguments:
        df (DataFrame): Captions matrix
    
    Returns:
        df (DataFrame): the given matrix after applying the preprocessing step
    """
    df['video_id'] = df.index
    df['first_level_category_id'] = df['first_level_category_id'].fillna(-1).astype(int)
    df['second_level_category_id'] = df['second_level_category_id'].fillna(-1).astype(int)
    df['third_level_category_id'] = df['third_level_category_id'].fillna(-1).astype(int)
    df['manual_cover_text'] = df['manual_cover_text'].fillna('')
    df['caption'] = df['caption'].fillna('')
    df['manual_cover_text'] = df['manual_cover_text'].str.replace(r'[\x00-\x7F]', '', regex=True)
    df['caption'] = df['caption'].str.replace(r'[\x00-\x7F]', '', regex=True)
    df['topic_tag'] = df['topic_tag'].str.replace(r'[\x00-\x7F]', '', regex=True)
    df['first_level_category_name'] = df['first_level_category_name'].str.replace(r'[\x00-\x7F]', '', regex=True)
    df['second_level_category_name'] = df['second_level_category_name'].str.replace(r'[\x00-\x7F]', '', regex=True)
    df['third_level_category_name'] = df['third_level_category_name'].str.replace(r'[\x00-\x7F]', '', regex=True)
    return df

# Functional programming -> maps file with its processing function
PREPROCESSING = {
    'social': preprocess_social,
    'item': preprocess_item,
    'big_interactions': preprocess_interactions,
    'small_interactions': preprocess_interactions,
    'caption': preprocess_captions
}


def load_matrix(matrix: str) -> pd.DataFrame:
    """
    Load a specific matrix according to a given name that matches
    any valid matrix name.
    
    Valid file names are:
    big_interactions, small_interactions, social, item, item_daily, user, caption
    
    Arguments:
        matrix (str): Matrix name to load
    
    Returns:
        df (DataFrame): the preprocessed matrix
    """
    if matrix not in FILE_NAMES:
        raise KeyError('Given matrix name does not exist.')
    print(f'Loading {matrix}...')

    df = pd.read_csv(f'{ROOT_PATH}{FILE_NAMES[matrix]}.csv')
    if matrix in PREPROCESSING:
        df = PREPROCESSING[matrix](df)
    return df


def load_matrices(*matrices):
    """
    Load a specific matrices according to given names that matches
    any valid matrix name.
    
    Valid file names are:
    big_interactions, small_interactions, social, item, item_daily, user, caption
    
    Arguments:
        matrices (*str): Matrice names to load
    
    Returns:
        multiple_df (tuple of DataFrames): the preprocessed matrices
    """
    return tuple(load_matrix(mat) for mat in matrices)


def load_dataset():
    """
    Load all the matrices available in the dataset
    
    Returns:
        multiple_df (tuple of DataFrames): matricies containing the data
    """
    return load_matrices(*list(FILE_NAMES.keys()))

def load_train_set():
    """
    Load all the specific matrices for building embeddings that is all matrices
    except for the small one.

    Returns:
        multiple_df (tuple of DataFrames): matricies containing the training data
    """
    interactions = load_matrix('big_interactions')
    social = load_matrix('social')
    item = load_matrix('item')
    item_daily = load_matrix('item_daily')
    user = load_matrix('user')
    caption = load_matrix('caption')
    return interactions, social, item, item_daily, user, caption
    