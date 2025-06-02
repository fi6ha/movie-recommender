import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load the dataset
df = pd.read_csv("tmdb_5000_movies.csv")

# keep only the columns we need
df = df[['title', 'overview', 'genres', 'tagline', 'keywords']].fillna('')

# combine all text into one column
df['content'] = df['overview'] + ' ' + df['tagline'] + ' ' + df['keywords'] + ' ' + df['genres']

# turn text into numbers using tf-idf
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# calculate similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# get movie titles with their index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# function to recommend similar movies
def recommend(title, num_recs=5):
    if title not in indices:
        return ["movie not found."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recs+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# streamlit app starts here
st.title("🎬 movie recommender")

# dropdown to select a movie
selected_movie = st.selectbox("choose a movie", df['title'].sort_values())

# button to show recommendations
if st.button("recommend"):
    recs = recommend(selected_movie)
    st.write(f"because you watched **{selected_movie}**, you might like:")
    for movie in recs:
        st.write("•", movie)