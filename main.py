import asyncio
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from src.recommender import Recommender, flag_adult_content
from src.genre_classifier import GenrePredictor
from src.mood_matcher import get_movies_by_prompt, analyze_prompt_deepseek
import ast
import pandas as pd
import re
import time
import numpy as np

GENRE_STYLES = {
    "Action": {"color": "#171c26", "emoji": "🔥"},
    "Adventure": {"color": "#171c26", "emoji": "🗺️"},
    "Comedy": {"color": "#171c26", "emoji": "😂"},
    "Drama": {"color": "#171c26", "emoji": "🎭"},
    "Horror": {"color": "#171c26", "emoji": "👻"},
    "Romance": {"color": "#171c26", "emoji": "❤️"},
    "Sci-Fi": {"color": "#171c26", "emoji": "🚀"},
    "Fantasy": {"color": "#171c26", "emoji": "🧙"},
    "Thriller": {"color": "#171c26", "emoji": "🔪"},
    "Mystery": {"color": "#171c26", "emoji": "🕵️"},
    "Animation": {"color": "#171c26", "emoji": "🎞️"},
    "Crime": {"color": "#171c26", "emoji": "🪓"},
    "Children": {"color": "#171c26", "emoji": "👶"},
    "IMAX": {"color": "#171c26", "emoji": "📽️"},
    "Documentary": {"color": "#171c26", "emoji": "📃"},
    "War": {"color": "#171c26", "emoji": "🪖"},
    "Music": {"color": "#171c26", "emoji": "🎵"},
    "History": {"color": "#171c26", "emoji": "📚"},
    "Family": {"color": "#171c26", "emoji": "👨‍👩‍👧‍👦"},
    "Science Fiction": {"color": "#171c26", "emoji": "🧬"},
    "TV Movie": {"color": "#171c26", "emoji": "📺"},
    "Western": {"color": "#171c26", "emoji": "🤠"},
}

# --- UI Setup ---
st.set_page_config(
    page_title="🎬 CineMatch | Movie Recommendation System",
    layout="wide"
)

# New caching function
def cache_data_with_pickle(*args, **kwargs):
    kwargs['hash_funcs'] = {pd.DataFrame: lambda _: None}
    return st.cache_data(*args, **kwargs)


# --- Custom CSS ---
@cache_data_with_pickle
def load_css(file_path):
    with open(file_path, "r") as f:
        css = f.read()
    return css

def is_shawshank(title):
    """Check if the movie is The Shawshank Redemption"""
    return "Shawshank Redemption" in title

st.markdown(f"<style>{load_css('styles.css')}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_genre_predictor():
    try:
        predictor = GenrePredictor(
            model_path='src/classifier_model/movie_genre_classifier.pth',
            mlb_path='src/classifier_model/label_binarizer.pkl'
        )
        return predictor
    except Exception as e:
        st.error(f"Failed to load genre classifier: {str(e)}")
        st.stop()

genre_predictor = load_genre_predictor()

@cache_data_with_pickle(ttl=3600, show_spinner="Calculating weighted ratings...")
def calculate_weighted_rating(df, min_votes_alltime=10000, min_votes_yearly=2000):
    """
    Calculate a statistically sound weighted score that:
    1. Requires minimum votes to qualify
    2. Balances movie's rating with global average
    3. Gives proper weight to vote counts

    Formula:
    Weighted Rating (WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C

    Where:
    R = movie's average rating
    v = number of votes for the movie
    m = minimum votes required to be listed
    C = mean vote across entire dataset
    """
    # Calculate mean vote across all movies
    C = df['average_rating'].mean()

    # Calculate minimum votes required (percentile-based)
    m_alltime = max(min_votes_alltime, df['rating_count'].quantile(0.90))
    m_yearly = max(min_votes_yearly, df['rating_count'].quantile(0.75))

    # For all-time rankings
    qualified = df[df['rating_count'] >= m_alltime].copy()
    qualified['weighted_score'] = (
            (qualified['rating_count'] / (qualified['rating_count'] + m_alltime) * qualified['average_rating']) +
            (m_alltime / (qualified['rating_count'] + m_alltime) * C)
    )

    # Additional refinements
    qualified['final_score'] = qualified['weighted_score'] * (
            1 + np.log10(qualified['rating_count'] / m_alltime)  # Bonus for exceeding min votes
    )

    return qualified.sort_values('final_score', ascending=False)

# --- Filtering function ---
@cache_data_with_pickle(ttl=600, show_spinner="Applying filters...")
def apply_filters(df, genres=None, year_range=None, hide_adult=True):
    filtered = df.copy()

    if genres:
        def genre_match(row):
            try:
                row_genres = ast.literal_eval(row["genres"]) if isinstance(row.get("genres", ""), str) else []
                return any(genre in row_genres for genre in genres)
            except:
                return False

        filtered = filtered[filtered.apply(genre_match, axis=1)]

    if year_range:
        filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

    if hide_adult:
        filtered = filtered[filtered["content_warning"].fillna("") == ""]

    return filtered

# --- Display card ---
def show_movie_card(row):
    with st.container():
        st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.9, 3.1])

        with col1:
            st.markdown("<div class='poster-container'>", unsafe_allow_html=True)
            if row.get("poster_url"):
                st.image(row["poster_url"], width=300)
            else:
                st.image("no-poster-available.png", width=180)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            title = row["title"]
            # Fix formatting
            title = re.sub(r"^(.*), The \((\d{4})\)", r"The \1 (\2)", title)
            title = re.sub(r"^(.*), A \((\d{4})\)", r"A \1 (\2)", title)
            title = re.sub(r"^(.*), An \((\d{4})\)", r"An \1 (\2)", title)
            title = re.sub(r"\(\d{4}\)", "", title).strip()

            try:
                parsed_date = pd.to_datetime(row.get("release_date", ""), errors='coerce')
                if pd.notna(parsed_date):
                    release_display = parsed_date.strftime("%d/%m/%Y")
                elif pd.notna(row.get("year", "")):
                    release_display = f"{int(row['year'])}"
            except:
                if pd.notna(row.get("year", "")):
                    release_display = f"{int(row['year'])}"

            # Title with year
            year_value = row.get("year")
            year_text = f" ({int(year_value)})" if pd.notna(year_value) else ""
            title_html = f"""
            <div style="display: flex; align-items: center; gap: 8px;">
                <h3 class='movie-title'>{title}{year_text}</h3>
                {'<span class="shawshank-badge" title="Highest rated movie of all time!">🏆</span>' if is_shawshank(row["title"]) else ''}
            </div>
            """
            st.markdown(title_html, unsafe_allow_html=True)

            if row.get("content_warning"):
                warning = re.sub(r"(⚠️\s*)+", "⚠️ ", row["content_warning"])
                st.markdown(f"<p class='content-warning'>{warning}</p>", unsafe_allow_html=True)

            # In the show_movie_card function
            # Get cast information
            cast_display = ""
            if pd.notna(row.get("cast")) and row.get("cast"):
                try:
                    cast_raw = row.get("cast", "")
                    cast_list = []

                    if isinstance(cast_raw, str):
                        if cast_raw.startswith("[") and "'" in cast_raw:
                            cast_list = ast.literal_eval(cast_raw)
                        elif "," in cast_raw:
                            cast_list = [c.strip() for c in cast_raw.split(",")]
                        else:
                            cast_list = cast_raw.split()
                    elif isinstance(cast_raw, list):
                        cast_list = cast_raw

                    # Get top 3-4 cast members
                    if cast_list:
                        top_cast = [c for c in cast_list[:4] if c]  # Filter out empty strings
                        if top_cast:
                            cast_display = f"Starring: {', '.join(top_cast)}"
                except:
                    pass

            # Get director information
            director_display = ""
            if pd.notna(row.get("crew")) and row.get("crew"):
                try:
                    crew_raw = row.get("crew", "")
                    directors = []

                    if isinstance(crew_raw, str):
                        if crew_raw.startswith("[") and "'" in crew_raw:
                            crew_list = ast.literal_eval(crew_raw)
                        elif "," in crew_raw:
                            crew_list = [c.strip() for c in crew_raw.split(",")]
                        else:
                            crew_list = crew_raw.split()
                    elif isinstance(crew_raw, list):
                        crew_list = crew_raw

                    # Remove duplicates while preserving order
                    seen = set()
                    directors = [x for x in crew_list if x and not (x in seen or seen.add(x))]

                    if directors:
                        director_display = f"Directed by: {', '.join(directors[:2])}"
                except:
                    pass

            # Only show "Unknown" if we actually have a crew field but couldn't parse it
            if not director_display and pd.notna(row.get("crew")):
                director_display = "Directed by: Unknown"

            # Display cast and director information
            if cast_display or director_display:
                st.markdown(f"""
                <div style='color: #BDBDBD; font-size: 15px; font-style: italic; margin: 5px 0;'>
                    {cast_display if cast_display else ''}
                    <br>
                    {director_display if director_display else ''}
                </div>
                """.replace("\n", "").strip(), unsafe_allow_html=True)

            overview = row.get("overview", "")
            if overview:
                st.markdown(f"<div style='font-size: 18px; line-height: 1.7; margin: 10px 0;'>{overview}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 18px; line-height: 1.7;'>No overview available.</div>",
                            unsafe_allow_html=True)

            # Release date display
            if release_display != "Unknown":
                st.markdown(f"<div style='color: #888; font-size: 15px;'>Released on: {release_display}</div>",
                            unsafe_allow_html=True)

            # Genre badges
            try:
                genres_raw = row.get("genres", "")
                genres = []

                if isinstance(genres_raw, str):
                    if genres_raw.startswith("[") and "'" in genres_raw:
                        genres = ast.literal_eval(genres_raw)
                    elif "," in genres_raw:
                        genres = [g.strip() for g in genres_raw.split(",")]
                    else:
                        genres = genres_raw.split()
                elif isinstance(genres_raw, list):
                    genres = genres_raw

                if genres:
                    genre_html = ""
                    for genre in genres:
                        style = GENRE_STYLES.get(genre, {"color": "#171c26", "emoji": ""})
                        genre_html += f"<span class='genre-badge' style='background-color: {style['color']};'>{style['emoji']} {genre}</span>"

                    st.markdown(f"<div class='genre-tags'>{genre_html}</div>", unsafe_allow_html=True)
            except:
                pass

        st.markdown("</div>", unsafe_allow_html=True)

# --- Display Mood Results ---
def show_mood_results(mood_analysis, recommendations):
    # Display the analyzed moods in bold plain text
    moods = mood_analysis.get("moods", [])

    if moods:
        mood_text = ", ".join(moods)
        st.markdown(f"### **Your mood:** {mood_text}")
        st.markdown("---")

    # Display recommendations using the existing card style
    if not recommendations.empty:
        st.markdown("## 🎯 Recommended Movies", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        for i, (_, row) in enumerate(recommendations.iterrows()):
            with col1 if i % 2 == 0 else col2:
                show_movie_card(row)
    else:
        st.warning("No movies match your mood description. Try being more specific.")

# --- Display Leaderboard Movies ---
def display_leaderboard_movies(movies_df):
    if movies_df.empty:
        st.warning("No movies found with enough ratings.")
        return

    # Create a scrollable container
    st.markdown("<div class='leaderboard-container'>", unsafe_allow_html=True)

    # Display each movie in the leaderboard
    for idx, (_, row) in enumerate(movies_df.iterrows()):
        rank = idx + 1
        title = row["title"]

        # Fix title formatting
        title = re.sub(r"^(.*), The \((\d{4})\)", r"The \1 (\2)", title)
        title = re.sub(r"^(.*), A \((\d{4})\)", r"A \1 (\2)", title)
        title = re.sub(r"^(.*), An \((\d{4})\)", r"An \1 (\2)", title)
        title = re.sub(r"\(\d{4}\)", "", title).strip()

        # Year
        year = int(row["year"]) if pd.notna(row.get("year")) else "Unknown"

        # Rating information
        avg_rating = round(row["average_rating"], 1) if pd.notna(row.get("average_rating")) else "N/A"
        rating_count = int(row["rating_count"]) if pd.notna(row.get("rating_count")) else 0

        # Special rank badges and colors
        if rank == 1:
            rank_badge = """
            <div class="rank-badge rank-gold">
                <div class="rank-medal"></div>
                <div class="rank-number">1</div>
            </div>
            """.replace("\n", "").strip()
        elif rank == 2:
            rank_badge = """
            <div class="rank-badge rank-silver">
                <div class="rank-medal"></div>
                <div class="rank-number">2</div>
            </div>
            """.replace("\n", "").strip()
        elif rank == 3:
            rank_badge = """
            <div class="rank-badge rank-bronze">
                <div class="rank-medal"></div>
                <div class="rank-number">3</div>
            </div>
            """.replace("\n", "").strip()
        else:
            rank_badge = f"""
            <div class="rank-badge rank-normal">
                <div class="rank-number">#{rank}</div>
            </div>
            """.replace("\n", "").strip()

        # Genre badges
        genre_html = ""
        try:
            genres_raw = row.get("genres", "")
            genres = []

            if isinstance(genres_raw, str):
                if genres_raw.startswith("[") and "'" in genres_raw:
                    genres = ast.literal_eval(genres_raw)
                elif "," in genres_raw:
                    genres = [g.strip() for g in genres_raw.split(",")]
                else:
                    genres = genres_raw.split()
            elif isinstance(genres_raw, list):
                genres = genres_raw

            for genre in genres[:3]:  # Limit to 3 genres
                style = GENRE_STYLES.get(genre, {"color": "#171c26", "emoji": ""})
                genre_html += f"<span class='genre-badge' style='background-color: {style['color']};'>{style['emoji']} {genre}</span>"
        except:
            pass

        # Get cast information
        cast_display = ""
        if pd.notna(row.get("cast")):
            try:
                cast_raw = row.get("cast", "")
                cast_list = []

                if isinstance(cast_raw, str):
                    if cast_raw.startswith("[") and "'" in cast_raw:
                        cast_list = ast.literal_eval(cast_raw)
                    elif "," in cast_raw:
                        cast_list = [c.strip() for c in cast_raw.split(",")]
                    else:
                        cast_list = cast_raw.split()
                elif isinstance(cast_raw, list):
                    cast_list = cast_raw

                # Get top 3-4 cast members
                if cast_list:
                    top_cast = cast_list[:4]
                    cast_display = f"Starring: {', '.join(top_cast)}"
            except:
                pass

        # Get director information
        director_display = ""
        if pd.notna(row.get("crew")):
            try:
                crew_raw = row.get("crew", "")
                directors = []

                if isinstance(crew_raw, str):
                    if crew_raw.startswith("[") and "'" in crew_raw:
                        crew_list = ast.literal_eval(crew_raw)
                    elif "," in crew_raw:
                        crew_list = [c.strip() for c in crew_raw.split(",")]
                    else:
                        crew_list = crew_raw.split()
                elif isinstance(crew_raw, list):
                    crew_list = crew_raw

                # Remove duplicates while preserving order
                seen = set()
                directors = [x for x in crew_list if not (x in seen or seen.add(x))]

                if directors:
                    director_display = f"Directed by: {', '.join(directors[:2])}"  # Limit to 2 directors
            except:
                pass

        if not director_display:
            director_display = "Directed by: Unknown"

        overview = row.get("overview", "")
        truncated_overview = (overview[:150] + "...") if len(overview) > 150 else overview
        if overview:
            # Truncate overview to first 150 characters
            if len(overview) > 150:
                overview = overview[:150] + "..."
        else:
            overview = "No overview available."

        # Movie container with improved layout
        st.markdown(f"""
        <div class='leaderboard-movie' data-rank='{rank}'>
            {rank_badge}
            <div class='leaderboard-content'>
                <div class='leaderboard-poster-container'>
                    <img src="{row.get('poster_url', 'no-poster-available.png')}" class='leaderboard-poster' alt="{title} poster">
                    <div class="poster-rating-badge">{avg_rating}</div>
                </div>
                <div class='leaderboard-info'>
                    <div class='leaderboard-title-row'>
                        <h3 class='leaderboard-title'>{title}</h3>
                        <span class='leaderboard-year-large'>{year}</span>
                        {'<span class="shawshank-badge" title="Best movie of all time!">🏆</span>' if is_shawshank(row["title"]) else ''}
                    </div>

                    <!-- Cast and director display -->
                    <div class='cast-display'>{cast_display}</div>
                    <div></div>
                    <div></div>
                    <div class='cast-display'>{director_display}</div>

                    <div class='rating-row'>
                        <span class='single-red-star'>★</span>
                        <span class='rating-count'>{rating_count:,} ratings</span>
                    </div>

                    <div class='overview-heading'>Overview</div>
                    <div class='leaderboard-overview' data-full-overview="{overview}">{truncated_overview}</div>

                    <div class='genre-tags'>{genre_html}</div>
                </div>
            </div>
        </div>
        """.replace("\n", "").strip(), unsafe_allow_html=True)

    # Close the scrollable container
    st.markdown("</div>", unsafe_allow_html=True)

# --- Load recommender ---
@st.cache_resource
def load_recommender():
    try:
        recommender = Recommender("data/movies_dataset_full.csv")
        return recommender
    except Exception as e:
        st.error(f"Failed to load movie data: {str(e)}")
        st.stop()

# Cache mood analysis results for 1 hour since they're expensive to compute
# --- Mood Matcher Caching Functions ---
@cache_data_with_pickle(ttl=3600, show_spinner=False)
def cached_analyze_prompt(prompt):
    return analyze_prompt_deepseek(prompt)

# Modified to handle the recommender object properly
@cache_data_with_pickle(ttl=3600, show_spinner=False)
def cached_get_movies_by_prompt(_recommender, prompt, top_n=5):
    """Note the leading underscore in _recommender to tell Streamlit not to hash it"""
    return get_movies_by_prompt(prompt, recommender, top_n=top_n)

recommender = load_recommender()
df = recommender.df
movie_list = recommender.all_titles()

# Add missing column if needed
if "content_warning" not in df.columns:
    df["content_warning"] = ""

# --- Header ---
st.markdown("<h1 style='text-align: center;'>🎬 CineMatch</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #757575; margin-top: -10px;'>Discover your next favorite movie</h3>",
            unsafe_allow_html=True)

# --- Session state management ---
if "active_feature" not in st.session_state:
    st.session_state.active_feature = "recommendation"  # Default view

# --- Feature selection tabs ---
st.markdown("<div class='feature-tabs'>", unsafe_allow_html=True)
tab_col1, tab_col2, tab_col3, tab_col4 = st.columns(4)
with tab_col1:
    if st.button("🔍 Find & Recommend", use_container_width=True):
        st.session_state.active_feature = "recommendation"
with tab_col2:
    if st.button("🪄 Mood Matcher", use_container_width=True):
        st.session_state.active_feature = "mood"
with tab_col3:
    if st.button("🏆 Top Movies Leaderboard", use_container_width=True):
        st.session_state.active_feature = "leaderboard"
st.markdown("</div>", unsafe_allow_html=True)
with tab_col4:
    if st.button("🔮 Genre from Description", use_container_width=True):
        st.session_state.active_feature = "genre_from_desc"
st.markdown("</div>", unsafe_allow_html=True)

# --- Default filters ---
hide_explicit = False
year_range = (1970, 2023)
selected_genres = []

# --- Recommendation Feature ---
if st.session_state.active_feature == "recommendation":
    # --- Search input ---
    st.markdown("# 🔍 Find a movie to get started")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query = st.text_input("Enter a movie title:",
                                     placeholder="E.g., The Shawshank Redemption, James Bond, etc.",
                                     key="movie_search")
    with search_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        random_movie = st.button("🎲 Surprise Me!")


    # --- Random Movie Selection ---
    if random_movie:
        try:
            with st.spinner("Finding a random movie for you..."):
                # Get a random movie from the entire dataset (not filtered)
                random_movie_row = df.sample(1).iloc[0].copy()
                random_movie_row["content_warning"] = flag_adult_content(
                    random_movie_row["title"], random_movie_row.get("tag", ""), random_movie_row.get("overview", "")
                )
                search_query = random_movie_row["title"]
                st.success(f"We've selected '{random_movie_row['title']}' for you!")
        except Exception as e:
            st.error(f"Error selecting random movie: {str(e)}")

    # --- Recommendation Flow ---
    if search_query:
        search_query = search_query.strip()

        try:
            with st.spinner("Finding your movie..."):
                # Don't cache search results as they need to be fresh
                best_matches_df = recommender.smart_search(search_query, top_n=1)

            if not best_matches_df.empty:
                # Get the best match
                best_match_title = best_matches_df.iloc[0]["title"]

                # Get the full information for the selected movie
                matched_df = df[df["title"] == best_match_title]

                if not matched_df.empty:
                    matched_row = matched_df.iloc[0].copy()
                    matched_row["content_warning"] = flag_adult_content(
                        matched_row["title"], matched_row.get("tag", ""), matched_row.get("overview", "")
                    )

                    st.markdown("## 🎬 Selected Movie", unsafe_allow_html=True)
                    show_movie_card(matched_row)

                    with st.spinner("Finding similar movies..."):
                        # Cache recommendations for this movie for 1 hour
                        st.cache_data.clear()
                        @cache_data_with_pickle(ttl=3600, show_spinner=False)
                        def get_cached_recommendations(title, genres, year_range, hide_explicit):
                            recs = recommender.recommend(title, top_n=10)
                            if "content_warning" not in recs.columns:
                                recs["content_warning"] = ""
                            return apply_filters(recs, genres, year_range, hide_explicit)

                        recs = get_cached_recommendations(
                            matched_row["title"],
                            selected_genres,
                            year_range,
                            hide_explicit
                        )

                    if not recs.empty:
                        st.markdown(f"## 🎯 Movies Similar to {matched_row['title']}", unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        for i, (_, row) in enumerate(recs.iterrows()):
                            with col1 if i % 2 == 0 else col2:
                                show_movie_card(row)
                    else:
                        st.warning("😕 No matching recommendations after applying filters.")
                else:
                    st.error("There was an error retrieving movie information. Please try again.")
            else:
                st.error("No movies match your search. Try a different name or spelling.")
        except Exception as e:
            st.error(f"Error processing your search: {str(e)}")
    else:
        st.info("👆 Type a movie title or click 'Surprise Me!' to get started.")

# --- Mood Matcher Feature ---
elif st.session_state.active_feature == "mood":
    st.markdown("## 🧙‍♂️ Describe what you're in the mood for")
    mood_col1, mood_col2 = st.columns([3, 1])
    with mood_col1:
        mood_query = st.text_input(
            "Enter your mood:",
            placeholder="E.g., 'funny sci-fi with aliens', 'dark crime thriller', 'heartwarming romance'",
            key="mood_input"
        )
    with mood_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        mood_search = st.button("✨ Find Movies")

    if mood_query and mood_search:
        mood_query = mood_query.strip()

        try:
            with st.spinner("Analyzing your mood and finding the best movies for it..."):
                # Show progress while waiting
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)  # Simulate processing time
                    progress_bar.progress(percent_complete + 1)

                # Use cached functions with the recommender object properly handled
                recommendations = cached_get_movies_by_prompt(recommender, mood_query, top_n=5)
                mood_analysis = cached_analyze_prompt(mood_query)

                if mood_analysis is None:
                    st.error("Couldn't analyze your mood description. Please try again.")
                elif not recommendations.empty:
                    show_mood_results(mood_analysis, recommendations)
                else:
                    st.error("Couldn't find matching movies. Try being more specific or using different words.")
        except Exception as e:
            st.error(f"Error processing your mood request: {str(e)}")
        finally:
            progress_bar.empty()
    else:
        st.info("👆 Describe what kind of movie you're in the mood for and click 'Find Movies'")

# --- Leaderboard Feature ---
elif st.session_state.active_feature == "leaderboard":
    st.markdown("# 🏆 Top Movies Leaderboard")

    # Selection for All-time or by Year
    leaderboard_type = st.radio(
        "Select leaderboard view:",
        ["All-time Top 100", "Top Movies by Year"],
        horizontal=True
    )

    if leaderboard_type == "All-time Top 100":
        with st.spinner("Loading top 100 movies of all time..."):
            # Cache the all-time leaderboard for 24 hours
            @cache_data_with_pickle(ttl=86400, show_spinner=False)
            def get_alltime_leaderboard():
                # Calculate mean rating across all movies
                C = df['average_rating'].mean()

                # Set minimum votes to 85th percentile or 10,000 votes, whichever is higher
                m = max(10000, df['rating_count'].quantile(0.85))

                # Filter movies that meet the minimum vote threshold
                qualified = df.dropna(subset=['average_rating', 'rating_count']).copy()
                qualified = qualified[qualified['rating_count'] >= m]

                # Calculate weighted score
                qualified['weighted_score'] = (
                        (qualified['rating_count'] / (qualified['rating_count'] + m) * qualified['average_rating']) +
                        (m / (qualified['rating_count'] + m) * C)
                )

                # Add logarithmic bonus for vote count
                qualified['final_score'] = qualified['weighted_score'] * (
                        1 + np.log10(qualified['rating_count'] / m)
                )

                # Get top 100
                return qualified.sort_values('final_score', ascending=False).head(100)

            top_movies = get_alltime_leaderboard()

            st.markdown("## 🌟 Top 100 Movies of All Time")
            display_leaderboard_movies(top_movies)

    else:  # Top Movies by Year
        selected_year = st.slider("Select Year:", min_value=1970, max_value=2023, value=2000)

        with st.spinner(f"Loading top movies from {selected_year}..."):
            # Cache yearly leaderboards for 24 hours
            @cache_data_with_pickle(ttl=86400, show_spinner=False)
            def get_yearly_leaderboard(year):
                year_df = df[df['year'] == year].copy()
                if year_df.empty:
                    return pd.DataFrame()

                # For yearly rankings, use lower threshold (60th percentile or 100 votes)
                C_year = year_df['average_rating'].mean()
                m_year = max(100, year_df['rating_count'].quantile(0.60))

                qualified_year = year_df.dropna(subset=['average_rating', 'rating_count']).copy()
                qualified_year = qualified_year[qualified_year['rating_count'] >= m_year]

                # Calculate weighted score (no logarithmic bonus for yearly rankings)
                qualified_year['final_score'] = (
                        (qualified_year['rating_count'] / (qualified_year['rating_count'] + m_year) * qualified_year['average_rating']) +
                        (m_year / (qualified_year['rating_count'] + m_year) * C_year)
                )

                # Get top 20
                return qualified_year.sort_values('final_score', ascending=False).head(40)

            top_year_movies = get_yearly_leaderboard(selected_year)

            st.markdown(f"## 🌟 Top Movies of {selected_year}")
            display_leaderboard_movies(top_year_movies)

elif st.session_state.active_feature == "genre_from_desc":
    st.markdown("## 🔮 Predict Genres from Movie Description")
    st.markdown(
        "*Here's a fun little experimentation tool to predict the genre(s) of any movie based on its description!*")

    with st.form("genre_prediction_form"):
        overview_input = st.text_area(
            "Enter a movie description:",
            placeholder="E.g., 'In a dystopian future, a group of rebels fight against a tyrannical regime...'",
            height=150
        )
        submitted = st.form_submit_button("Predict Genres & Find Similar Movies")

    if submitted and overview_input.strip():
        with st.spinner("Analyzing description and finding recommendations..."):
            try:
                # Predict genres with threshold=0.4
                predicted_genres, confidences = genre_predictor.predict_genres(overview_input, threshold=0.3)

                # Display the predicted genres with confidence scores
                st.markdown("### Predicted Genres:")
                if not predicted_genres:
                    st.warning("No genres predicted above threshold. Try a more descriptive text.")
                else:
                    cols = st.columns(4)
                    for i, (genre, conf) in enumerate(confidences.items()):
                        with cols[i % 4]:
                            style = GENRE_STYLES.get(genre, {"color": "#171c26", "emoji": ""})
                            st.markdown(
                                f"<div style='background-color: {style['color']}; color: white; padding: 8px; "
                                f"border-radius: 12px; margin: 5px 0; text-align: center;'>"
                                f"{style['emoji']} {genre}</div>",
                                unsafe_allow_html=True
                            )
                            st.progress(conf, text=f"{conf:.0%}")


                    # Function to parse genres consistently
                    def parse_genres(genre_data):
                        if isinstance(genre_data, list):
                            return [g.strip().lower() for g in genre_data]
                        try:
                            if isinstance(genre_data, str):
                                if genre_data.startswith("[") and ("'" in genre_data or '"' in genre_data):
                                    return [g.strip().lower().strip("'\"") for g in ast.literal_eval(genre_data)]
                                elif "," in genre_data:
                                    return [g.strip().lower() for g in genre_data.split(",")]
                                else:
                                    return [genre_data.strip().lower()]
                        except:
                            pass
                        return []


                    # Create a working copy of the dataframe
                    working_df = df.copy()
                    working_df["parsed_genres"] = working_df["genres"].apply(parse_genres)

                    # Calculate content warnings for all movies
                    working_df["content_warning"] = working_df.apply(
                        lambda row: flag_adult_content(
                            row["title"],
                            row.get("tag", ""),
                            row.get("overview", "")
                        ),
                        axis=1
                    )


                    # Create genre matching score
                    def calculate_genre_match_score(row_genres):
                        if not row_genres:
                            return 0

                        score = 0
                        total_weight = sum(confidences.values())

                        for pred_genre, confidence in confidences.items():
                            norm_pred = pred_genre.lower().strip()
                            if norm_pred in row_genres:
                                score += confidence / total_weight
                        return score


                    working_df["genre_match_score"] = working_df["parsed_genres"].apply(calculate_genre_match_score)

                    # Apply smart content filtering (always on but less aggressive)
                    filter_conditions = (working_df["genre_match_score"] > 0)

                    # Only completely exclude the most explicit content
                    filter_conditions &= (~working_df["content_warning"].str.contains("Mature Content", na=False))

                    # Apply all filters
                    filtered_df = working_df[filter_conditions].copy()

                    if not filtered_df.empty:
                        # Calculate content safety score (soft penalty for borderline cases)
                        def content_safety_penalty(row):
                            overview = str(row.get("overview", "")).lower()
                            tag = str(row.get("tag", "")).lower()

                            # Soft penalty for movies with some mature indicators
                            borderline_keywords = {
                                "adult": 0.8,
                                "mature": 0.85,
                                "intense": 0.9,
                                "graphic": 0.8,
                                "sexual": 0.7
                            }

                            penalty = 1.0
                            for kw, weight in borderline_keywords.items():
                                if kw in overview or kw in tag:
                                    penalty = min(penalty, weight)
                            return penalty


                        # Combine scores with soft safety adjustments
                        filtered_df["recommendation_score"] = (
                                0.6 * filtered_df["genre_match_score"] * filtered_df.apply(content_safety_penalty,
                                                                                           axis=1) +
                                0.25 * filtered_df["normalized_rating"] +
                                0.15 * filtered_df["normalized_popularity"]
                        )

                        # Get top recommendations
                        recommendations = filtered_df.sort_values(
                            "recommendation_score",
                            ascending=False
                        ).head(10)

                        # Display results
                        st.markdown("---")
                        st.markdown("### 🎯 Recommended Movies")
                        col1, col2 = st.columns(2)
                        for i, (_, row) in enumerate(recommendations.iterrows()):
                            with col1 if i % 2 == 0 else col2:
                                show_movie_card(row)
                    else:
                        # Fallback with relaxed filtering if no results
                        st.warning("Showing most relevant matches:")

                        fallback_df = working_df[
                            (working_df["genre_match_score"] > 0)
                        ].sort_values("normalized_popularity", ascending=False).head(10)

                        if not fallback_df.empty:
                            col1, col2 = st.columns(2)
                            for i, (_, row) in enumerate(fallback_df.iterrows()):
                                with col1 if i % 2 == 0 else col2:
                                    show_movie_card(row)
                        else:
                            st.error("Could not find any matching movies. Try a different description.")

            except Exception as e:
                st.error(f"Error processing your request: {str(e)}")
    elif submitted:
        st.warning("Please enter a movie description")