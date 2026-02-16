# 🎬 K-Drama Recommendation API

A content-based K-Drama recommendation system built using **FastAPI**, **Pandas**, and **Scikit-Learn**.
This API recommends similar dramas based on genres, synopsis, cast, writer, director, and rating using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 🚀 Features

* 🔍 Content-based recommendation engine
* ✨ Fuzzy title matching (handles spelling mistakes)
* 🎭 Genre-weighted similarity scoring
* 🧠 TF-IDF + Cosine Similarity
* 🎯 Returns top 10 similar dramas
* 🖼 Includes title, image, rating, genres & synopsis
* 🌐 CORS enabled (ready for frontend integration)
* 📦 Deployable on Render

---

## 🛠 Tech Stack

* **FastAPI**
* **Pandas**
* **Scikit-Learn**
* **TF-IDF Vectorizer**
* **Cosine Similarity**
* **Difflib (Fuzzy Matching)**

---

## 📂 Project Structure

```
API/
│
├── main.py
├── kdramas.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

1. Dataset is loaded and cleaned.
2. Animation genre is excluded.
3. Writer and director fields are processed.
4. Text features combined:

   * Genres (weighted 3x)
   * Synopsis
   * Main Leads
   * Writer
   * Director
   * Rating
5. TF-IDF vectorization applied.
6. Cosine similarity matrix generated.
7. User input normalized.
8. Exact match → else fuzzy match.
9. Top 10 similar dramas returned in JSON format.

---

## 🧪 Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the API

```bash
uvicorn main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🌐 API Endpoints

### GET `/`

Returns API status message.

Example response:

```json
{
  "message": "K-Drama Recommendation API is running 🚀"
}
```

---

### POST `/recommendation`

Request Body:

```json
{
  "kdrama": "Crash Landing on You"
}
```

Response:

```json
{
  "matched_kdrama": "Crash Landing on You",
  "recommendations": [
    {
      "title": "Descendants of the Sun",
      "image_url": "...",
      "rating": "8.5",
      "genres": "Action, Romance",
      "synopsis": "..."
    }
  ]
}
```

---

## 🚀 Deployment (Render)

**Build Command**

```
pip install -r requirements.txt
```

**Start Command**

```
uvicorn main:app --host 0.0.0.0 --port 10000
```

---

## 📊 Dataset Source

This project uses a dataset sourced from Kaggle.
Licensed under **Attribution–NonCommercial 4.0 (CC BY-NC 4.0)**.

All rights belong to the original dataset creator.
This project is developed for educational and demonstration purposes only.

---

## 👩‍💻 Author

Shruti Singh
B.Tech Student | AI/ML Enthusiast
Building full-stack ML-powered applications 🚀

---

