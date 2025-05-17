# Project Assignment: Short Video Recommender System (KuaiRec)

## Introduction

Welcome to my proposal solution for recommanding videos using the KuaiRec dataset. I am RUFF Maxime, EPITA SCIA 2026, and I am happy to guide you here in my journey of designing my first recommender system!

---

## Architecture

The intention of this `README` file is to provide you directions to discover how I implemented my system.

First, here is a high level architecture of the repository:

- `docs/`: contains documents and papers downloaded from the dataset
- `src/`: contains all the code files
- `src/collaborative-models/`: contains all the notebooks about implementing 2 collaborative models
- `src/content-based-models/`: contains all the notebooks about implementing 2 users/items content based models
- `src/embeddings/`: directory to store the pickle files that store embeddings of content, items. Everything is explained into the notebooks 

---

## Roadmap: how to explore this repository

Here are the notebooks in directory `src/` to open (read from start to begin, especially headers and footers because I try to make some great introductions and a summaries to help making links between the notebooks!) in the correct order to discover the whole of my work:

1. `exploratory_data_analysis`: exploration of the dataset
2. `captions_preprocessing`: fix encoding issues with the captions CSV file
3. `content_embedding`: embed features (items, users, interactions...)
4. `content-based-models/logistic_regression`: implement a logistic regression model, explaination of metrics/dataset splitting
5. `content-based-models/nnencoding`: implementation of a neural network
6. `collaborative-models/collaborative_filtering`: implementing the collaborative filtering algorithm 
7. `collaborative-models/knn`: using k-nn algorithm for collaborative recommendations
8. `serendipity`: use of serendipity for the final model
8. `video_recommender_system`: final recommender system for recommending videos

Have a great reading time!

---

### Instructions for loading the dataset

In order to run all the code that requires to use the raw CSV files from the KuaiRec dataset, you need first to download the dataset and put the files under a directory called `kuairec` at the root of the repository. Nothing should change in the directory structure when dowloading the dataset, except renaming the folder `kuairec`.