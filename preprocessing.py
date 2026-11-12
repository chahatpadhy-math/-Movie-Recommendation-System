#Task1

import pandas as pd
df = pd.read_csv("tmdb_5000_movies.csv")
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nFirst Five Rows:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Records:")
print(df.duplicated().sum())
print("\nStatistical Summary:")
print(df.describe())

#Task2

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
df["overview"] = df["overview"].fillna("")
df["genres"] = df["genres"].fillna("")
df["keywords"] = df["keywords"].fillna("")
df["text"] = (
    df["overview"] + " " +
    df["genres"] + " " +
    df["keywords"]
)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    words = text.split()
    words = [
        word for word in words
        if word not in ENGLISH_STOP_WORDS
    ]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)
print(df[["original_title", "overview", "clean_text"]].head())
print("\nMissing values in clean_text:")
print(df["clean_text"].isnull().sum())

#Task3 

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
tfidf_matrix = tfidf.fit_transform(df["clean_text"])
print("\nTF-IDF Matrix Shape:")
print(tfidf_matrix.shape)
print("\nFirst 20 Features:")
print(tfidf.get_feature_names_out()[:20])

#Task4

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(tfidf_matrix)
print("\nSimilarity Matrix Shape:")
print(similarity.shape)
print("\nSimilarity Scores:")
print(similarity[:5, :5])

#Task5

def recommend(item_name, top_n=5):
    movie_indices = df[
        df["original_title"].str.lower() == item_name.lower()
    ].index
    if len(movie_indices) == 0:
        print("Movie not found!")
        return
    index = movie_indices[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )
    scores = scores[1:top_n + 1]
    movie_indices = [i[0] for i in scores]
    recommendations = df.iloc[movie_indices][
        ["original_title", "genres", "overview"]
    ]
    return recommendations
print("\nRecommendations for Avatar:")
print(recommend("Avatar", 5))
print("\nRecommendations for Batman Begins:")
print(recommend("Batman Begins", 5))
print("\nRecommendations for The Avengers:")
print(recommend("The Avengers", 5))
