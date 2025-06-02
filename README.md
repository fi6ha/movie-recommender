# Movie Recommender App

This is a simple movie recommendation system built using Streamlit. It uses content-based filtering to suggest similar movies based on their overview, tagline, keywords, and genres.

## How it works

The recommender system works by:

- Combining text-based metadata (overview, tagline, keywords, genres) for each movie
- Converting the combined text into numerical vectors using TF-IDF (Term Frequency–Inverse Document Frequency)
- Calculating cosine similarity between all movies
- Recommending the top N most similar movies to the one selected by the user

## Requirements

Install the required Python packages using pip:

```bash
pip install streamlit pandas scikit-learn

```

## Dataset Source

The dataset used in this project (`tmdb_5000_movies.csv`) is **not included** in this repository due to licensing and redistribution restrictions.

You can download the dataset yourself from Kaggle:  
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Please place the dataset CSV file in the project directory before running the app.
