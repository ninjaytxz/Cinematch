import json
from openai import OpenAI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# --- CONFIGURATION ---
DEEPSEEK_API_KEY = "REPLACE WITH YOUR ACTUAL API"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def analyze_prompt_deepseek(prompt: str):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a movie assistant. The user will give you a free-form prompt "
                        "describing the type of movie they want to watch. Extract and return only a valid JSON object "
                        "with three keys: 'moods', 'genres', and 'keywords'. Each should be a list of lowercase strings."
                    )
                },
                {
                    "role": "user",
                    "content": f"Prompt: {prompt}\n\nReturn JSON like: "
                               "{\"moods\": [...], \"genres\": [...], \"keywords\": [...]}"
                }
            ],
            response_format={"type": "json_object"},
            stream=False,
            temperature=0.7
        )

        content = response.choices[0].message.content
        print("✅ DeepSeek Response:", content)
        return json.loads(content)

    except Exception as e:
        print("❌ DeepSeek API Error:", e)
        return None


def get_movies_by_prompt(prompt: str, recommender, top_n: int = 5):
    df = recommender.df
    parsed = analyze_prompt_deepseek(prompt)
    if not parsed:
        return pd.DataFrame()

    # --- Extracted from DeepSeek ---
    all_terms = parsed.get("moods", []) + parsed.get("genres", []) + parsed.get("keywords", [])
    search_query = " ".join(all_terms).strip()
    if not search_query:
        return pd.DataFrame()

    # --- Prepare TF-IDF text field ---
    combined_text = (
            df["overview"].fillna("").astype(str) + " " +
            df["tag"].fillna("").astype(str) + " " +
            df["genres"].fillna("").apply(lambda g: " ".join(g) if isinstance(g, list) else str(g))
    )

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    query_vector = vectorizer.transform([search_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # --- Metadata Quality Scoring ---
    metadata_scores = []
    for i in range(len(df)):
        score = 1.0  # Base score

        # Penalize missing or very short overviews
        overview = df.iloc[i]["overview"]
        if not overview or len(str(overview).strip()) < 50:
            score *= 0.7  # 30% penalty

        # Penalize missing posters
        if not df.iloc[i]["poster_url"] or pd.isna(df.iloc[i]["poster_url"]):
            score *= 0.9  # 10% penalty

        # Penalize empty tags
        if not df.iloc[i]["tag"] or pd.isna(df.iloc[i]["tag"]):
            score *= 0.95  # 5% penalty

        # Boost complete metadata
        if (overview and len(str(overview).strip()) > 100 and
                df.iloc[i]["poster_url"] and
                df.iloc[i]["tag"] and
                len(df.iloc[i]["genres"]) > 0):
            score *= 1.2  # 20% boost

        metadata_scores.append(score)

    # Apply metadata quality adjustments
    cosine_similarities = cosine_similarities * metadata_scores

    # --- Boost for known titles/franchises ---
    known_titles = recommender.all_titles()
    title_matches = [
        (i, fuzz.partial_ratio(prompt.lower(), title.lower()))
        for i, title in enumerate(df["title"])
    ]
    for i, score in title_matches:
        if score > 85:
            cosine_similarities[i] += 0.5  # Strong boost
        elif score > 70:
            cosine_similarities[i] += 0.25  # Mild boost

    # --- Optional: keyword boost ---
    boost_keywords = parsed.get("keywords", [])
    for i in range(len(df)):
        title = df.iloc[i]["title"].lower()
        overview = df.iloc[i]["overview"].lower()
        tag = df.iloc[i]["tag"].lower() if isinstance(df.iloc[i]["tag"], str) else ""
        for kw in boost_keywords:
            if kw in title:
                cosine_similarities[i] += 0.08
            elif kw in overview or kw in tag:
                cosine_similarities[i] += 0.04

    # --- Get Top Matches ---
    top_indices = cosine_similarities.argsort()[::-1][:top_n * 3]  # Get more candidates for filtering

    # Filter out movies with very poor metadata
    filtered_indices = [
                           i for i in top_indices
                           if (len(str(df.iloc[i]["overview"]).strip()) > 30 and  # At least some overview
                               len(df.iloc[i]["genres"]) > 0)  # Has at least one genre
                       ][:top_n]  # Take top_n after filtering

    if not filtered_indices:
        # Fallback if all results were filtered out
        filtered_indices = top_indices[:top_n]

    results = df.iloc[filtered_indices].copy()
    results["match_score"] = cosine_similarities[filtered_indices]

    # --- Final Output ---
    columns_to_return = [
        "title", "overview", "poster_url", "tag", "genres", "year", "content_warning", "match_score"
    ]
    return results[[col for col in columns_to_return if col in results.columns]]