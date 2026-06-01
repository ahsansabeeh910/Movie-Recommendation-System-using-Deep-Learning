import numpy as np


def recommend_movies(
    user_id,
    model,
    user_encoder,
    movie_encoder,
    ratings,
    movies,
    top_n=10
):

    encoded_user = user_encoder.transform(
        [user_id]
    )[0]

    movie_ids = ratings[
        "movieId"
    ].unique()

    encoded_movies = movie_encoder.transform(
        movie_ids
    )

    users = np.full(
        len(encoded_movies),
        encoded_user
    )

    predictions = model.predict(
        [
            users.astype("int32"),
            encoded_movies.astype("int32")
        ],
        verbose=0
    )

    top_indices = predictions.flatten().argsort()[-top_n:]

    recommended_movie_ids = movie_ids[
        top_indices
    ]

    recommendations = movies[
        movies["movieId"].isin(
            recommended_movie_ids
        )
    ][["title"]]

    return recommendations