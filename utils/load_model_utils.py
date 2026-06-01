import tensorflow as tf
import pickle

def load_files():

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