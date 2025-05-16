import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from collections import defaultdict
import re
import ast

def flag_adult_content(title, tags, overview):
    if not overview:
        return ""
    warning_keywords = {
        "nudity": 2, "sexual": 2, "sex": 1.5, "erotic": 2, "explicit": 1.5,
        "violence": 1.5, "graphic": 1.5, "gore": 2, "rape": 2.5,
        "abuse": 1.5, "intimacy": 1, "torture": 2, "assault": 1.5, "incest": 2.5,
        "disturbing": 1, "mature": 0.5, "adult": 0.5
    }
    adult_tags = {
        "nsfw", "18+", "r-rated", "nc-17", "adult", "mature audience", "explicit content",
        "violent", "nudity", "sexual content"
    }
    content_score = 0
    overview_lower = overview.lower()
    for keyword, weight in warning_keywords.items():
        if keyword in overview_lower:
            content_score += weight
    title_lower = title.lower() if title else ""
    for keyword, weight in warning_keywords.items():
        if keyword in title_lower:
            content_score += weight * 0.5
    if tags:
        tags_lower = tags.lower()
        for adult_tag in adult_tags:
            if adult_tag in tags_lower:
                content_score += 1.5
        for keyword, weight in warning_keywords.items():
            if keyword in tags_lower:
                content_score += weight * 0.4
        genre_indicators = {"horror", "thriller", "crime", "war", "action"}
        for genre in genre_indicators:
            if genre in tags_lower:
                content_score -= 0.2
    threshold = 2.5 if tags else 3.0
    return "⚠️ Might Contain Mature Content" if content_score >= threshold else ""

class Recommender:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path,encoding='ISO-8859-1')
        self.df.fillna("", inplace=True)

        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].str.encode('ISO-8859-1').str.decode('utf-8', errors='replace')

        # Normalize genres
        def normalize_genres(g):
            if isinstance(g, list):
                return g
            elif isinstance(g, str):
                try:
                    if g.startswith("[") and ("'" in g or '"' in g):
                        return ast.literal_eval(g)
                    elif "," in g:
                        return [x.strip() for x in g.split(",")]
                except Exception:
                    pass
            return []
        self.df["genres"] = self.df["genres"].apply(normalize_genres)

        # Detect franchises
        self._detect_franchises()

        # Normalize ratings and rating counts
        self.df["normalized_rating"] = (
            (self.df["average_rating"] - self.df["average_rating"].min()) /
            (self.df["average_rating"].max() - self.df["average_rating"].min())
        ).fillna(0.5)

        # Log-scale for rating_count to reduce skew
        self.df["normalized_rating_count"] = (
            np.log1p(self.df["rating_count"]) /
            np.log1p(self.df["rating_count"].max())
        ).fillna(0.1)

        # Combined popularity score (70% rating count, 30% rating)
        self.df["normalized_popularity"] = (
            0.7 * self.df["normalized_rating_count"] +
            0.3 * self.df["normalized_rating"]
        )

        # Initialize vectorizers
        self.tag_vectorizer = TfidfVectorizer()
        self.genre_vectorizer = TfidfVectorizer()
        self.overview_vectorizer = TfidfVectorizer()
        self.cast_crew_vectorizer = TfidfVectorizer()
        self.search_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")

        # Create matrices
        self.tag_matrix = self.tag_vectorizer.fit_transform(self.df["tag"].fillna(""))
        self.genre_matrix = self.genre_vectorizer.fit_transform(
            self.df["genres"].apply(lambda x: ' '.join(x))
        )
        self.overview_matrix = self.overview_vectorizer.fit_transform(self.df["overview"].fillna(""))
        self.cast_crew_matrix = self.cast_crew_vectorizer.fit_transform(
            (self.df["cast"] + " " + self.df["crew"]).fillna("")
        )
        combined_search_text = (
            self.df["title"] + " " +
            self.df["overview"] + " " +
            self.df["tag"] + " " +
            self.df["cast"] + " " +
            self.df["crew"]
        ).fillna("")
        self.search_matrix = self.search_vectorizer.fit_transform(combined_search_text)

        # Precompute similarity matrices
        self.tag_sim = cosine_similarity(self.tag_matrix)
        self.genre_sim = cosine_similarity(self.genre_matrix)
        self.overview_sim = cosine_similarity(self.overview_matrix)
        self.cast_crew_sim = cosine_similarity(self.cast_crew_matrix)

    def _detect_franchises(self):
        base_titles = self.df["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)
        base_titles = base_titles.str.replace(r"\s*[IVXLCDM]+$", "", regex=True)
        title_parts = base_titles.str.split().explode()
        common_words = title_parts.value_counts()
        franchise_words = common_words[common_words > 3].index.tolist()
        self.df["franchise"] = base_titles.apply(
            lambda x: next((word for word in x.split() if word in franchise_words), "")
        )

    def get_index_by_title(self, title):
        match = self.df[self.df["title"].str.lower() == title.lower()]
        return match.index[0] if not match.empty else None

    def smart_search(
            self,
            query,
            top_n=5,
            weights=None,  # Change from mutable dict to None
            min_score_threshold=0.4
    ):
        """Enhanced smart search with multiple matching strategies."""

        # Initialize weights if not provided
        if weights is None:
            weights = {
                'title_fuzzy': 0.25,
                'title_exact': 0.3,
                'context': 0.2,
                'genre': 0.1,
                'year': 0.05,
                'popularity': 0.1
            }

        # 1. Preprocess query
        query = query.lower().strip()
        query_terms = set(re.findall(r'\w+', query))

        # 2. Initialize score dictionaries
        scores = {
            'title_fuzzy': defaultdict(float),
            'title_exact': defaultdict(float),
            'context': defaultdict(float),
            'genre': defaultdict(float),
            'year': defaultdict(float)
        }

        # 3. Exact and fuzzy title matching (multi-strategy)
        for idx, title in self.df["title"].fillna("").items():
            title_lower = title.lower()

            # Exact matches (including partial matches)
            if query in title_lower:
                scores['title_exact'][idx] = weights['title_exact']

            # Fuzzy matching with multiple algorithms
            fuzzy_score = max(
                fuzz.token_set_ratio(query, title_lower),
                fuzz.partial_ratio(query, title_lower),
                fuzz.WRatio(query, title_lower)
            ) / 100.0

            if fuzzy_score > 0.5:  # Only consider decent matches
                scores['title_fuzzy'][idx] = fuzzy_score * weights['title_fuzzy']

        # 4. Contextual matching with expanded features
        query_vector = self.search_vectorizer.transform([query])
        context_sim = cosine_similarity(query_vector, self.search_matrix).flatten()
        for idx, sim in enumerate(context_sim):
            if sim > 0.1:  # Higher threshold to reduce noise
                scores['context'][idx] = sim * weights['context']

        # 5. Genre matching
        if len(query_terms) > 1:  # Only apply for multi-word queries
            for idx, genres in self.df["genres"].items():
                genre_match = query_terms & set(' '.join(genres).lower().split())
                if genre_match:
                    scores['genre'][idx] = (len(genre_match) / len(query_terms)) * weights['genre']

        # 6. Year matching (if year is detected in query)
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        if year_match:
            target_year = int(year_match.group())
            for idx, year in self.df["year"].items():
                if not pd.isna(year):
                    year_diff = abs(year - target_year)
                    scores['year'][idx] = (1 / (1 + year_diff)) * weights['year']

        # 7. Combine all scores with normalization
        combined_scores = defaultdict(float)
        all_indices = set()

        # Collect all indices that have any score
        for score_type in scores:
            all_indices.update(scores[score_type].keys())

        # Combine scores with normalization per type
        for idx in all_indices:
            combined = 0
            for score_type in scores:
                combined += scores[score_type].get(idx, 0)

            # Add popularity boost (scaled by combined score to prevent unpopular items from dominating)
            popularity_boost = self.df.at[idx, "normalized_popularity"] * weights['popularity']
            combined += popularity_boost * combined  # Scale boost with existing score

            combined_scores[idx] = combined

        # 8. Normalize final scores to [0, 1]
        if combined_scores:
            max_score = max(combined_scores.values())
            if max_score > 0:
                combined_scores = {k: v / max_score for k, v in combined_scores.items()}

        # 9. Filter by threshold and get top candidates
        candidates = [(idx, score) for idx, score in combined_scores.items() if score >= min_score_threshold]

        # 10. Multi-stage ranking:
        # - First by combined score
        # - Then by popularity among high-scoring candidates
        # - Finally by year proximity if year was specified
        if year_match:
            candidates.sort(
                key=lambda x: (
                    -x[1],  # Primary: combined score
                    -self.df.at[x[0], "normalized_popularity"],  # Secondary: popularity
                    -scores['year'].get(x[0], 0)  # Tertiary: year proximity
                )
            )
        else:
            candidates.sort(
                key=lambda x: (
                    -x[1],  # Primary: combined score
                    -self.df.at[x[0], "normalized_popularity"]  # Secondary: popularity
                )
            )

        # 11. Prepare results
        top_indices = [idx for idx, _ in candidates[:top_n * 2]]  # Get extra for deduplication

        # Deduplicate similar titles
        unique_titles = set()
        final_indices = []
        for idx in top_indices:
            title = self.df.at[idx, "title"]
            norm_title = re.sub(r'[^a-z0-9]', '', title.lower())
            if norm_title not in unique_titles:
                unique_titles.add(norm_title)
                final_indices.append(idx)
                if len(final_indices) >= top_n:
                    break

        results = self.df.loc[final_indices, [
            "title", "overview", "poster_url", "tag", "genres", "year"
        ]].copy()

        results["content_warning"] = results.apply(
            lambda row: flag_adult_content(row["title"], row.get("tag", ""), row.get("overview", "")),
            axis=1
        )
        results["score"] = [combined_scores[idx] for idx in final_indices]

        return results.reset_index(drop=True)

    def recommend(self, movie_title, top_n=10, alpha=0.7, year_weight=0.1, franchise_boost=0.3, pop_boost=0.2):
        index = self.get_index_by_title(movie_title)
        if index is None:
            return pd.DataFrame()

        tag_scores = list(enumerate(self.tag_sim[index]))
        genre_scores = list(enumerate(self.genre_sim[index]))
        overview_scores = list(enumerate(self.overview_sim[index]))
        cast_crew_scores = list(enumerate(self.cast_crew_sim[index]))

        content_scores = [
            (i, 0.35 * tag_scores[i][1] + 0.2 * genre_scores[i][1] + 0.25 * overview_scores[i][1] + 0.2 * cast_crew_scores[i][1])
            for i in range(len(self.df))
        ]

        content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i for i, score in content_scores[1:150]]

        candidates = self.df.iloc[top_indices].copy()
        candidates["content_score"] = [score for i, score in content_scores[1:150]]

        current_franchise = self.df.at[index, "franchise"]
        candidates["franchise_boost"] = candidates["franchise"].apply(
            lambda x: franchise_boost if x == current_franchise else 0
        ) if current_franchise else 0

        current_year = self.df.at[index, "year"]
        candidates["year_similarity"] = 1 / (1 + abs(candidates["year"] - current_year))

        candidates["hybrid_score"] = (
            alpha * candidates["content_score"] +
            (1 - alpha) * candidates["normalized_rating"] +
            year_weight * candidates["year_similarity"] +
            pop_boost * candidates["normalized_popularity"] +
            candidates["franchise_boost"]
        )

        candidates["content_warning"] = candidates.apply(
            lambda row: flag_adult_content(row["title"], row.get("tag", ""), row.get("overview", "")),
            axis=1
        )

        return candidates.sort_values(by="hybrid_score", ascending=False).head(top_n)[[
            "title", "overview", "poster_url", "tag", "genres", "year", "content_warning",
            "cast", "crew"  # Add these columns
        ]]

    def all_titles(self):
        return self.df["title"].dropna().unique().tolist()