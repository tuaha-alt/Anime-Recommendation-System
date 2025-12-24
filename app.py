import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# Load the dataset
anime_data = pd.read_csv("cleaned_data.csv")

# Fill missing tags
anime_data['Tags'] = anime_data['Tags'].fillna('')

# Vectorize the tags
cv = CountVectorizer(stop_words='english', max_features=13500)
anime_vectors = cv.fit_transform(anime_data['Tags'])

# Build Nearest Neighbors model (memory safe)
model = NearestNeighbors(
    n_neighbors=11,   # 1 extra because the anime itself will appear
    metric='cosine',
    algorithm='brute'
)
model.fit(anime_vectors)

# Streamlit UI
st.title('Anime Recommendation System')

selected_anime = st.selectbox(
    'Select an anime:',
    anime_data['Name'].values
)

if st.button('Get Recommendations'):
    anime_index = anime_data[anime_data['Name'] == selected_anime].index[0]

    distances, indices = model.kneighbors(
        anime_vectors[anime_index]
    )

    # Remove the selected anime itself
    recommended_indices = indices[0][1:]

    recommended_anime = anime_data.iloc[recommended_indices]['Name'].values

    st.subheader('Recommended Anime')
    st.table(recommended_anime)
