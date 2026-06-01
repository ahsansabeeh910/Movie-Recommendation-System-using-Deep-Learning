# 🎬 Movie Recommendation System using Deep Learning

<p align="center">
  <img src="assets/banner.png" width="100%" alt="Movie Recommendation Banner">
</p>

## Overview

A deep learning-powered movie recommendation system that generates personalized movie suggestions using user-movie interaction data from the MovieLens-20M dataset. The project leverages neural collaborative filtering techniques and provides recommendations through an interactive Streamlit frontend.

---

## Features

* Personalized movie recommendations
* Deep learning recommendation engine
* Streamlit interactive UI
* Trained using MovieLens-20M dataset
* Saved model checkpoints
* Encoder persistence using Pickle
* Google Colab training workflow
* Robust handling for model reloads

---

## Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* Google Colab
* MovieLens Dataset

---

## Project Structure

```text
Movie-Recommendation-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   └── banner.png
│
├── models/
│
├── data/
│
└── notebooks/
```

## Installation

```bash
git clone https://github.com/ahsansabeeh910/Movie-Recommendation-System-using-Deep-Learning
pip install -r requirements.txt

streamlit run app.py
```

## Dataset

MovieLens-20M Dataset used for training recommendation embeddings and personalized prediction modeling.

## Model Performance

| Metric | Score |
| ------ | ----- |
| RMSE   | 0.332 |
| MAE    | 0.241 |


---

## Author

**Sabeeh Ahsan**

Deep Learning • Machine Learning • Full Stack Development

---


