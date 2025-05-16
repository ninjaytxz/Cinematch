import pandas as pd
import numpy as np
import requests
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import diskcache
from gensim.models import KeyedVectors

# Configuration
TMDB_API_KEY = "REPLACE WITH YOUR ACTUAL API KEY"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
MAX_WORKERS = 20
REQUEST_DELAY = 0.02
ENABLE_CACHING = True
FASTTEXT_MODEL_PATH = "wiki-news-300d-1M.vec"

# File paths
DATA_DIR = "../data/MovieLens_Dataset"
INPUT_FILES = {
    "movies": os.path.join(DATA_DIR, "movies.csv"),
    "ratings": os.path.join(DATA_DIR, "ratings.csv"),
    "tags": os.path.join(DATA_DIR, "tags.csv"),
    "links": os.path.join(DATA_DIR, "links.csv")
}
OUTPUT_FILE = os.path.join(DATA_DIR, "../data/movie_dataset_full.csv")

# Setup persistent session and caching
session = requests.Session()
session.params = {"api_key": TMDB_API_KEY}
cache = diskcache.Cache('./tmdb_cache')


class MovieDataProcessor:
    def __init__(self):
        self.movies = None
        self.ratings = None
        self.tags = None
        self.links = None
        self.final_data = None
        self.w2v_model = None

    def load_fasttext_model(self):
        """Load FastText word vectors for semantic filtering"""
        try:
            print("🔄 Loading FastText model (wiki-news-300d-1M.vec)...")
            self.w2v_model = KeyedVectors.load_word2vec_format(FASTTEXT_MODEL_PATH, binary=False)
            print("✅ FastText model loaded.")
        except Exception as e:
            print(f"⚠️ Could not load FastText model: {e}")
            self.w2v_model = None

    def load_data(self):
        """Load all raw data files"""
        print("🚀 Loading datasets...")
        self.movies = pd.read_csv(INPUT_FILES["movies"])
        self.ratings = pd.read_csv(INPUT_FILES["ratings"])
        self.tags = pd.read_csv(INPUT_FILES["tags"])
        self.links = pd.read_csv(INPUT_FILES["links"])

    def clean_tags(self, tags_df, level="full", min_tag_freq=5):
        """Clean and filter tags data with semantic filtering"""
        print(f"🧽 Cleaning tags (mode: {level})...")

        # Whitelist and banned keywords
        WHITELIST = {
            "comedy", "drama", "thriller", "horror", "romance", "documentary", "action",
            "sci-fi", "fantasy", "crime", "adventure", "animation", "mystery",
            "biography", "history", "war", "music", "western", "sport",
            "mind-bending", "emotional", "dark", "slow-burn", "feel-good", "tragic",
            "based on true story", "classic", "noir"
        }

        BANNED_KEYWORDS = {
            "watched", "seen", "movie", "trailer", "blu-ray", "flop", "blockbuster",
            "oscar", "subtitles", "3d", "imax", "better than expected"
        }

        def basic_clean(tag):
            tag = str(tag).lower().strip()
            tag = re.sub(r"\(.*?\)", "", tag)
            tag = re.sub(r"[^\w\s\-]", "", tag)
            tag = re.sub(r"\b\w{1,2}\b", "", tag)
            tag = re.sub(r"\s+", " ", tag).strip()
            return tag

        def semantic_filter(tag, threshold=0.35):
            """Use FastText model to check semantic similarity with whitelist"""
            if not self.w2v_model:
                return False  # No model available

            try:
                similarities = [
                    self.w2v_model.similarity(tag, ref_tag)
                    for ref_tag in WHITELIST
                    if tag in self.w2v_model and ref_tag in self.w2v_model
                ]
                return max(similarities, default=0) >= threshold
            except Exception as e:
                print(f"⚠️ Semantic filter error for '{tag}': {e}")
                return False

        def is_tag_valid(tag, level="full"):
            if tag in WHITELIST:
                return True
            if any(banned in tag for banned in BANNED_KEYWORDS):
                return False
            if len(tag.split()) > 6:
                return False  # too long, likely noisy
            if level == "light":
                return True
            # Full mode: use semantic similarity with fallback
            return semantic_filter(tag) or len(tag.split()) <= 2

        tags_df = tags_df.copy()
        tags_df['tag'] = tags_df['tag'].astype(str).str.lower()
        tags_df['tag'] = tags_df['tag'].apply(basic_clean)
        tags_df = tags_df[tags_df['tag'].notnull()]

        # Apply validation
        tags_df['tag'] = tags_df['tag'].apply(
            lambda tag: tag if is_tag_valid(tag, level=level) else "")
        tags_df['tag'] = tags_df['tag'].apply(lambda x: x if x.strip() else np.nan)

        # Frequency filter (preserve NaNs)
        tag_counts = tags_df['tag'].value_counts()
        valid_tags = tag_counts[tag_counts >= min_tag_freq].index
        tags_df = tags_df[tags_df['tag'].isin(valid_tags) | tags_df['tag'].isna()]
        tags_df = tags_df.drop_duplicates().reset_index(drop=True)

        print(f"✅ Tags cleaned: {len(tags_df)} rows, {tags_df['tag'].nunique()} unique tags")
        return tags_df

    def process_movies(self):
        """Process movies and ratings data"""
        print("🧹 Processing movies and ratings...")

        # Process movies
        self.movies['genres'] = self.movies['genres'].apply(
            lambda x: x.split('|') if isinstance(x, str) else [])

        # Add year column from title
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)')
        self.movies['year'] = pd.to_numeric(self.movies['year'], errors='coerce')

        # Filter movies by year (1970-2023)
        self.movies = self.movies[(self.movies['year'] >= 1970) & (self.movies['year'] <= 2023)]

        # Process ratings
        self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'], unit='s')

        # Filter users by rating count (20-200 ratings)
        user_rating_counts = self.ratings['userId'].value_counts()
        valid_users = user_rating_counts[(user_rating_counts >= 20) & (user_rating_counts <= 200)].index
        self.ratings = self.ratings[self.ratings['userId'].isin(valid_users)]

    def merge_data(self):
        """Merge and filter all datasets"""
        print("🔗 Merging datasets...")

        # Clean tags
        self.tags = self.clean_tags(self.tags, level="full", min_tag_freq=5)

        # Create tag summary
        tag_summary = self.tags.groupby('movieId')['tag'].apply(
            lambda x: ', '.join(set(x.dropna()))).reset_index()

        # Calculate rating stats
        rating_stats = self.ratings.groupby('movieId').agg(
            average_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        ).reset_index()

        # Filter movies by rating stats (≥10 ratings, avg ≥3.0)
        rating_stats = rating_stats[(rating_stats['rating_count'] >= 10) &
                                    (rating_stats['average_rating'] >= 3.0)]

        # Merge all data
        merged_data = pd.merge(self.movies, rating_stats, on='movieId', how='inner')
        merged_data = pd.merge(merged_data, tag_summary, on='movieId', how='left')

        # Merge with links to get TMDB IDs
        merged_data = pd.merge(merged_data, self.links, on='movieId', how='left')

        print("✅ Data merged and filtered")
        return merged_data

    def clean_title(self, title):
        """Extract clean title and year from formatted title"""
        match = re.search(r"\((\d{4})\)$", title)
        year = int(match.group(1)) if match else None
        clean = re.sub(r"\s*\(\d{4}\)$", "", title).strip()
        return clean, year

    @cache.memoize()
    def search_movie(self, title, year=None):
        """Search for movie with caching"""
        clean, extracted_year = self.clean_title(title)
        params = {
            "query": clean,
            "include_adult": False,
            "language": "en-US",
            "year": year if year is not None else extracted_year
        }

        try:
            response = session.get(f"{TMDB_BASE_URL}/search/movie", params=params)
            if response.status_code == 429:
                time.sleep(1)
                return self.search_movie(title, year)

            response.raise_for_status()
            results = response.json().get("results", [])

            if results:
                return results[0]

            # Fallback without year
            if params["year"]:
                del params["year"]
                response = session.get(f"{TMDB_BASE_URL}/search/movie", params=params)
                if response.ok:
                    results = response.json().get("results", [])
                    if results:
                        return results[0]

        except Exception as e:
            print(f"⚠️ Search error for {clean}: {e}")
        return None

    @cache.memoize()
    def get_movie_details(self, movie_id):
        """Get movie details with caching"""
        try:
            response = session.get(f"{TMDB_BASE_URL}/movie/{movie_id}")
            if response.status_code == 429:
                time.sleep(1)
                return self.get_movie_details(movie_id)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️ Details error for {movie_id}: {e}")
            return {}

    @cache.memoize()
    def get_movie_credits(self, movie_id):
        """Get credits with caching"""
        try:
            response = session.get(f"{TMDB_BASE_URL}/movie/{movie_id}/credits")
            if response.status_code == 429:
                time.sleep(1)
                return self.get_movie_credits(movie_id)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️ Credits error for {movie_id}: {e}")
            return {}

    @cache.memoize()
    def get_movie_release_rating(self, movie_id):
        """Get certification with caching"""
        try:
            response = session.get(f"{TMDB_BASE_URL}/movie/{movie_id}/release_dates")
            if response.status_code == 429:
                time.sleep(1)
                return self.get_movie_release_rating(movie_id)

            if response.ok:
                for entry in response.json().get("results", []):
                    if entry.get("iso_3166_1") == "US":
                        for release in entry.get("release_dates", []):
                            return release.get("certification", "")
        except Exception:
            pass
        return ""

    def process_movie_row(self, row, idx, df):
        """Process a single movie row"""
        if ENABLE_CACHING and pd.notna(row.get("popularity")):
            return

        # First try using the TMDB ID if available
        if pd.notna(row.get("tmdbId")):
            try:
                movie_id = int(row["tmdbId"])
                details = self.get_movie_details(movie_id)
                if details:  # If found with TMDB ID
                    credits = self.get_movie_credits(movie_id)
                    certification = self.get_movie_release_rating(movie_id)

                    # Extract top 5 cast members
                    cast = ", ".join([c["name"] for c in credits.get("cast", [])[:5]])

                    # Extract key crew members
                    director = next((m["name"] for m in credits.get("crew", [])
                                     if m["job"] == "Director"), None)
                    writer = next((m["name"] for m in credits.get("crew", [])
                                   if m["job"] in ["Writer", "Screenplay"]), None)
                    crew = ", ".join(filter(None, [director, writer]))

                    # Update dataframe
                    df.at[idx, "popularity"] = details.get("popularity")
                    df.at[idx, "release_date"] = details.get("release_date")
                    df.at[idx, "cast"] = cast
                    df.at[idx, "crew"] = crew
                    df.at[idx, "genres"] = ", ".join([g["name"] for g in details.get("genres", [])])
                    df.at[idx, "rating"] = details.get("vote_average")
                    df.at[idx, "certification"] = certification
                    time.sleep(REQUEST_DELAY)
                    return
            except:
                pass

        # Fallback to title search if no TMDB ID or lookup failed
        movie_data = self.search_movie(row["title"], row.get("year"))
        if not movie_data:
            return

        movie_id = movie_data["id"]
        details = self.get_movie_details(movie_id)
        credits = self.get_movie_credits(movie_id)
        certification = self.get_movie_release_rating(movie_id)

        # Extract top 5 cast members
        cast = ", ".join([c["name"] for c in credits.get("cast", [])[:5]])

        # Extract key crew members
        director = next((m["name"] for m in credits.get("crew", [])
                         if m["job"] == "Director"), None)
        writer = next((m["name"] for m in credits.get("crew", [])
                       if m["job"] in ["Writer", "Screenplay"]), None)
        crew = ", ".join(filter(None, [director, writer]))

        # Update dataframe
        df.at[idx, "popularity"] = details.get("popularity")
        df.at[idx, "release_date"] = details.get("release_date")
        df.at[idx, "cast"] = cast
        df.at[idx, "crew"] = crew
        df.at[idx, "genres"] = ", ".join([g["name"] for g in details.get("genres", [])])
        df.at[idx, "rating"] = details.get("vote_average")
        df.at[idx, "certification"] = certification

        # Small delay between requests
        time.sleep(REQUEST_DELAY)

    def enrich_movies(self, df):
        """Enrich movie data with TMDB metadata"""
        print("🎬 Enriching movie data with TMDB metadata...")

        # Initialize columns if they don't exist
        for col in ["popularity", "release_date", "cast", "crew", "rating", "certification"]:
            if col not in df.columns:
                df[col] = None

        # Process in parallel with progress bar
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.process_movie_row, row, idx, df): idx
                for idx, row in df.iterrows()
            }

            for future in tqdm(as_completed(futures), total=len(df), desc="Processing"):
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠️ Processing error: {e}")

        return df

    def fix_alternative_titles(self, df):
        """Handle movies with alternative titles (a.k.a.)"""
        print("🔎 Fixing missing movies with alternative titles...")

        aka_df = df[df['title'].str.contains(r'\(a\.k\.a\.', regex=True, na=False)].copy()

        def extract_title_variants(title):
            """Extract original and all a.k.a. variants from a title."""
            year_match = re.search(r'\((\d{4})\)$', title)
            year = year_match.group(1) if year_match else None
            title_wo_year = re.sub(r'\s*\(\d{4}\)$', '', title)

            aka_parts = re.findall(r'\(a\.k\.a\.\s*(.*?)\)', title_wo_year)
            main_title = re.split(r'\(a\.k\.a\.', title_wo_year)[0].strip()

            def fix_the_case(t):
                match = re.match(r'^(.*),\s*The$', t)
                return f"The {match.group(1)}" if match else t

            variants = []
            variants.append(fix_the_case(main_title) + f" ({year})" if year else fix_the_case(main_title))

            for aka in aka_parts:
                aka_fixed = fix_the_case(aka.strip())
                variants.append(f"{aka_fixed} ({year})" if year else aka_fixed)

            return variants, year

        for idx, row in aka_df.iterrows():
            title = row['title']
            variants, year = extract_title_variants(title)
            found = False

            for variant in variants:
                for try_year in [True, False]:
                    variant_query = variant if try_year else re.sub(r'\s*\(\d{4}\)$', '', variant).strip()
                    result = self.search_movie(variant_query, year if try_year else None)
                    if result:
                        details = self.get_movie_details(result["id"])
                        credits = self.get_movie_credits(result["id"])
                        certification = self.get_movie_release_rating(result["id"])

                        # Update metadata
                        df.at[idx, "popularity"] = details.get("popularity")
                        df.at[idx, "release_date"] = details.get("release_date")
                        df.at[idx, "cast"] = ", ".join([c["name"] for c in credits.get("cast", [])[:5]])
                        director = next((m["name"] for m in credits.get("crew", []) if m["job"] == "Director"), None)
                        writer = next(
                            (m["name"] for m in credits.get("crew", []) if m["job"] in ["Writer", "Screenplay"]), None)
                        df.at[idx, "crew"] = ", ".join(filter(None, [director, writer]))
                        df.at[idx, "rating"] = details.get("vote_average")
                        df.at[idx, "certification"] = certification
                        found = True
                        break
                    time.sleep(0.25)
                if found:
                    break

        return df

    def save_final_data(self):
        """Save final dataset"""
        self.final_data.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 Saved final dataset to {OUTPUT_FILE}")

    def run_full_pipeline(self):
        """Run the complete data processing pipeline"""
        self.load_fasttext_model()  # Load FastText model first
        self.load_data()
        self.process_movies()
        self.final_data = self.merge_data()
        self.final_data = self.enrich_movies(self.final_data)
        self.final_data = self.fix_alternative_titles(self.final_data)
        self.save_final_data()
        print("✅ All processing complete!")


if __name__ == "__main__":
    processor = MovieDataProcessor()
    processor.run_full_pipeline()