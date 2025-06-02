import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import ast

    
# load data
df = pd.read_csv("tmdb_5000_movies.csv")

# keep only useful columns
df = df[['title', 'overview', 'genres', 'tagline', 'keywords']]

# fill missing values with empty strings
df.fillna('', inplace=True)

# combine text features
def combine_features(row):
    return row['overview'] + ' ' + row['tagline'] + ' ' + row['keywords'] + ' ' + row['genres']

df['content'] = df.apply(combine_features, axis=1)


# tf-idf vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['content'])

# cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# map movie titles to indices
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def parse_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        genre_names = [g['name'] for g in genres]
        return ", ".join(genre_names)
    except:
        return ""
    
df['genres'] = df['genres'].apply(parse_genres)

# recommendation function
def recommend(title, num_recommendations=5):
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]

    recs = []
    for i in movie_indices:
        recs.append({
            'title': df.iloc[i]['title'],
            'genres': df.iloc[i]['genres'],
            'overview': df.iloc[i]['overview'][:300] + '...' if len(df.iloc[i]['overview']) > 300 else df.iloc[i]['overview']
        })
    return recs

# streamlit app
st.title(" 🎬  Simple Movie Recommender")

movie_list = df['title'].tolist()
selected_movie = st.selectbox("Choose a movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    if not recommendations:
        st.write("Movie not found or no recommendations.")
    else:
        st.write(f"because you watched **{selected_movie}**, you might like:")
        for movie in recommendations:
            st.markdown(f"### ⭐ {movie['title']}")
            st.markdown(f"**Genres:** {movie['genres']}")
            st.write(movie['overview'])
            st.markdown("---")