from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import difflib
import ast
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = FastAPI(title="K-Drama Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "kdramas.csv")

kdrama_data = pd.read_csv(CSV_PATH)
kdrama_data = kdrama_data.fillna("")
kdrama_data = kdrama_data[
    ~kdrama_data["genres"].str.contains("Animation", case=False, na=False)
]


import re
import unicodedata

def normalize_title(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)     # normalize spaces
    return text.strip()

def clean_text(value):
    """
    Converts ['Name'] -> Name
    Keeps normal strings unchanged
    """
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return " ".join(parsed)
        except:
            pass
    return str(value)

for col in ["writer", "director"]:
    if col in kdrama_data.columns:
        kdrama_data[col] = kdrama_data[col].apply(clean_text)


selected_features = [
    "rating",
    "genres",
    "synopsis",
    "mainLead1",
    "mainLead2",
    "writer",
    "director",
]

for feature in selected_features:
    kdrama_data[feature] = kdrama_data[feature].fillna("").astype(str)

combined_features = (
    (kdrama_data["genres"] + " ") * 3 +
    kdrama_data["synopsis"] + " " +
    kdrama_data["mainLead1"] + " " +
    kdrama_data["mainLead2"] + " " +
    kdrama_data["writer"] + " " +
    kdrama_data["director"]+" "+
    kdrama_data["rating"]
)



vectorizer = TfidfVectorizer(stop_words="english")
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

kdrama_data["title_clean"] = (
    kdrama_data["title"].str.lower().str.strip()
)

list_of_all_titles = kdrama_data["title_clean"].tolist()

class KdramaRequest(BaseModel):
    kdrama: str

@app.get("/")
def root():
    return {"message": "K-Drama Recommendation API is running 🚀"}

@app.post("/recommendation")
def recommend(request: KdramaRequest):
    user_input = normalize_title(request.kdrama)

    exact_match = kdrama_data[
        kdrama_data["title_clean"] == user_input
    ]

    if not exact_match.empty:
        index_of_drama = exact_match.index[0]

    else:
        
        close_matches = difflib.get_close_matches(
            user_input,
            list_of_all_titles,
            n=1,
            cutoff=0.6   
        )

        if not close_matches:
            return {
                "error": "No matching K-drama found",
                "recommendations": []
            }

        close_match = close_matches[0]

        matched_rows = kdrama_data[
            kdrama_data["title_clean"] == close_match
        ]

        if matched_rows.empty:
            return {
                "error": "Matched drama not found in dataset",
                "recommendations": []
            }

        index_of_drama = matched_rows.index[0]

    similarity_scores = list(enumerate(similarity[index_of_drama]))
    sorted_similar = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []

    for idx, score in sorted_similar[1:11]:
        try:
            row = kdrama_data.iloc[idx]

            recommendations.append({
                "title": str(row.get("title", "")),
                "image_url": str(row.get("image_url", "")),
                "rating": str(row.get("rating", "")),
                "genres": str(row.get("genres", "")),
                "synopsis":str(row.get("synopsis", ""))
            })

        except:
            continue  

    return {
        "matched_kdrama": str(kdrama_data.iloc[index_of_drama]["title"]),
        "recommendations": recommendations
    }
