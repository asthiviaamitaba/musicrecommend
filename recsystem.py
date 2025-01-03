import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('spotify_tracks_2024.csv')

# Recommendation function
def get_recommendations(track_name, df, n_recommendations=5):
    """Get recommendations based on track name with multiple features"""
    try:
        # Find the track index by name
        track_idx = df[df['name'].str.lower() == track_name.lower()].index[0]

        # Normalize popularity
        scaler = MinMaxScaler()
        df['popularity_scaled'] = scaler.fit_transform(df[['popularity']])

        # Create genre feature matrix
        mlb = MultiLabelBinarizer()
        df['genre_list'] = df['genre'].apply(lambda x: [x] if pd.notnull(x) else ['unknown'])
        genre_matrix = mlb.fit_transform(df['genre_list'])

        # Calculate genre similarity
        genre_similarities = cosine_similarity(genre_matrix)

        # Create popularity matrix (using scaled popularity)
        popularity_matrix = df['popularity_scaled'].values.reshape(-1, 1)

        # Calculate popularity similarity (cosine similarity)
        popularity_similarities = cosine_similarity(popularity_matrix)

        # Combine genre and popularity similarity scores (weight them)
        combined_similarities = 0.7 * genre_similarities + 0.3 * popularity_similarities

        # Get similar tracks based on the combined similarity score
        sim_scores = list(enumerate(combined_similarities[track_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:]  # Skip the first one (itself)

        # Get recommendations
        track_indices = [i[0] for i in sim_scores]
        recommendations = df.iloc[track_indices][['name', 'artist', 'genre', 'popularity']]
        recommendations['similarity_score'] = [i[1] for i in sim_scores]

        return recommendations

    except IndexError:
        return None

# Streamlit UI
st.title("Music Recommendation System 🎵")
st.markdown("Cari rekomendasi lagu berdasarkan nama lagu favoritmu!")

# Input song from user
song_input = st.text_input("Masukkan nama lagu:", "").strip()

# Initialize session state for recommendation display
if "num_recommendations" not in st.session_state:
    st.session_state.num_recommendations = 5
    st.session_state.recommendations = None

# Button to get recommendations
if st.button("Cari Rekomendasi"):
    if song_input:
        recommendations = get_recommendations(song_input, df)
        if recommendations is not None:
            st.session_state.recommendations = recommendations
            st.session_state.num_recommendations = 5  # Reset to initial batch size
        else:
            st.warning(f"Lagu '{song_input}' tidak ditemukan di dataset.")
    else:
        st.warning("Masukkan nama lagu terlebih dahulu.")

# Display recommendations in batches
if st.session_state.recommendations is not None:
    st.subheader("Lagu Rekomendasi:")
    recommendations_to_display = st.session_state.recommendations.head(st.session_state.num_recommendations)
    st.dataframe(recommendations_to_display)

    # Show more recommendations button
    if st.session_state.num_recommendations < len(st.session_state.recommendations):
        if st.button("Tampilkan Lebih Banyak Rekomendasi"):
            st.session_state.num_recommendations += 5
