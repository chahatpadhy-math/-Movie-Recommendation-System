
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)

st.title("🎬 Movie Recommendation System")
st.write("Discover movies similar to your favourite movie!")

@st.cache_data
def load_data():

    df = pd.read_csv("tmdb_5000_movies.csv")

    df = df.drop_duplicates(
        subset=["original_title"]
    ).reset_index(drop=True)

    for col in ["overview", "genres", "keywords"]:
        df[col] = df[col].fillna("")

    df["text"] = (
        df["overview"] + " " +
        df["genres"] + " " +
        df["keywords"]
    )

    def clean_text(text):

        text = text.lower()

        text = re.sub(
            r"[^a-zA-Z0-9\s]",
            " ",
            text
        )

        words = text.split()

        words = [
            word for word in words
            if word not in ENGLISH_STOP_WORDS
        ]

        return " ".join(words)

    df["clean_text"] = df["text"].apply(clean_text)

    return df

@st.cache_resource
def create_vectors(text_data):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(text_data)

    return matrix


df = load_data()
tfidf_matrix = create_vectors(df["clean_text"])

def recommend(item_name, top_n=5):

    indices = df[
        df["original_title"].str.lower()
        == item_name.lower()
    ].index

    if len(indices) == 0:
        return pd.DataFrame()

    index = indices[0]
  
    scores = cosine_similarity(
        tfidf_matrix[index],
        tfidf_matrix
    ).flatten()

    similar_indices = scores.argsort()[::-1]
  
    similar_indices = [
        i for i in similar_indices
        if i != index
    ]

    top_indices = similar_indices[:top_n]

    return df.iloc[top_indices][
        ["original_title", "genres", "overview"]
    ]

st.subheader("Select Your Favourite Movie")

movie_list = sorted(
    df["original_title"].dropna().unique()
)

selected_movie = st.selectbox(
    "Choose a movie:",
    movie_list
)

if st.button("Recommend Movies"):

    result = recommend(selected_movie, 5)

    if result.empty:

        st.warning("No recommendations found.")

    else:

        st.success(
            f"Recommended movies similar to {selected_movie}"
        )

        for _, movie in result.iterrows():

            st.markdown("---")

            st.subheader(movie["original_title"])

            st.write("Genres:", movie["genres"])

            st.write("Overview:", movie["overview"])


st.markdown("---")

st.caption(
    "Created using Python, Pandas, Scikit-learn and Streamlit"
)
