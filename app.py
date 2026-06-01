import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -------------------------
# LOAD MODEL + ENCODERS
# -------------------------

@st.cache_resource
def load_resources():

    model = tf.keras.models.load_model(
        "models/final_movie_model.keras"
    )

    with open(
        "models/user_encoder.pkl",
        "rb"
    ) as f:
        user_encoder = pickle.load(f)

    with open(
        "models/movie_encoder.pkl",
        "rb"
    ) as f:
        movie_encoder = pickle.load(f)

    return model, user_encoder, movie_encoder


# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data
def load_data():

    ratings = pd.read_csv(
        "data/rating.csv"
    )

    movies = pd.read_csv(
        "data/movie.csv"
    )

    return ratings, movies


model, user_encoder, movie_encoder = load_resources()

ratings, movies = load_data()


# -------------------------
# RECOMMEND FUNCTION
# -------------------------

def recommend_movies(
    user_id,
    top_n=10
):

    encoded_user = user_encoder.transform(
        [user_id]
    )[0]

    # only movies encoder knows
    known_movie_ids = movie_encoder.classes_

    users = np.full(
        len(known_movie_ids),
        encoded_user
    )

    predictions = model.predict(
        [
            users.astype("int32"),
            known_movie_ids.astype("int32")
        ],
        verbose=0
    )

    top_indices = predictions.flatten().argsort()[-top_n:]

    recommended_movie_ids = known_movie_ids[
        top_indices
    ]

    recommendations = movies[
        movies["movieId"].isin(
            recommended_movie_ids
        )
    ][["title"]]

    return recommendations


# -------------------------
# UI
# -------------------------

try:

    st.image(
        "assets/banner.png",
        use_container_width=True
    )

except:

    pass


st.title(
    "🎬 Movie Recommendation System"
)

st.markdown(
    "Deep Learning Based Personalized Movie Recommendations"
)

col1, col2 = st.columns(2)

with col1:

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        value=1
    )

with col2:

    top_n = st.slider(
        "Number of Recommendations",
        1,
        20,
        10
    )


if st.button(
    "Recommend Movies"
):

    try:

        with st.spinner(
            "Finding recommendations..."
        ):

            recommendations = recommend_movies(
                user_id,
                top_n
            )

        st.success(
            "Recommendations Generated"
        )

        st.subheader(
            "Top Recommended Movies"
        )

        for idx, movie in enumerate(
            recommendations["title"].tolist(),
            start=1
        ):

            st.write(
                f"{idx}. {movie}"
            )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )


st.markdown("---")

st.caption(
    "Built using TensorFlow • Streamlit • MovieLens Dataset"
)