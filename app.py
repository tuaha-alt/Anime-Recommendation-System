# import pandas as pd
# import streamlit as st
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.neighbors import NearestNeighbors

# # Load the dataset
# anime_data = pd.read_csv("cleaned_data.csv")

# # Fill missing tags
# anime_data['Tags'] = anime_data['Tags'].fillna('')

# # Vectorize the tags
# cv = CountVectorizer(stop_words='english', max_features=13500)
# anime_vectors = cv.fit_transform(anime_data['Tags'])

# # Build Nearest Neighbors model (memory safe)
# model = NearestNeighbors(
#     n_neighbors=11,   # 1 extra because the anime itself will appear
#     metric='cosine',
#     algorithm='brute'
# )
# model.fit(anime_vectors)

# # Streamlit UI
# st.title('Anime Recommendation System')

# selected_anime = st.selectbox(
#     'Select an anime:',
#     anime_data['Name'].values
# )

# if st.button('Get Recommendations'):
#     anime_index = anime_data[anime_data['Name'] == selected_anime].index[0]

#     distances, indices = model.kneighbors(
#         anime_vectors[anime_index]
#     )

#     # Remove the selected anime itself
#     recommended_indices = indices[0][1:]

#     recommended_anime = anime_data.iloc[recommended_indices]['Name'].values

#     st.subheader('Recommended Anime')
#     st.table(recommended_anime)
import pandas as pd
import streamlit as st
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (DARK ANIME THEME)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: #e5e7eb;
}
h1, h2, h3, h4 {
    color: #f8fafc;
}
.card {
    background-color: #020617;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 15px rgba(99,102,241,0.2);
}
.tag {
    display: inline-block;
    background-color: #6366f1;
    color: white;
    padding: 5px 10px;
    border-radius: 10px;
    margin-right: 5px;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
anime_data = pd.read_csv("cleaned_data.csv")

anime_data['Tags'] = anime_data['Tags'].fillna('')
anime_data['Image_URL'] = anime_data['Image_URL'].fillna(
    "https://via.placeholder.com/300x450?text=No+Image"
)

# -----------------------------
# VECTORIZE TAGS
# -----------------------------
cv = CountVectorizer(stop_words='english', max_features=13500)
anime_vectors = cv.fit_transform(anime_data['Tags'])

# -----------------------------
# NEAREST NEIGHBORS MODEL
# -----------------------------
model = NearestNeighbors(
    n_neighbors=11,
    metric='cosine',
    algorithm='brute'
)
model.fit(anime_vectors)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🎌 Anime Recommender")
st.sidebar.markdown("Find anime with similar **themes, genres, and tags**.")

selected_anime = st.sidebar.selectbox(
    "🎬 Select an Anime",
    anime_data['Name'].values
)

show_tags = st.sidebar.checkbox("Show Tags", value=True)
show_similarity = st.sidebar.checkbox("Show Similarity Score", value=True)

# -----------------------------
# MAIN TITLE
# -----------------------------
st.title("✨ Anime Recommendation System")
st.caption("🔍 Content-based recommendations using cosine similarity")

# -----------------------------
# BUTTON ACTION
# -----------------------------
if st.sidebar.button("🔥 Get Recommendations"):
    with st.spinner("Finding the best anime for you..."):
        time.sleep(1)

        anime_index = anime_data[
            anime_data['Name'] == selected_anime
        ].index[0]

        distances, indices = model.kneighbors(
            anime_vectors[anime_index]
        )

        recommended_indices = indices[0][1:]
        similarity_scores = 1 - distances[0][1:]

    st.success("🎉 Recommendations Ready!")

    # -----------------------------
    # SELECTED ANIME CARD
    # -----------------------------
    st.markdown("## 🎯 Because you liked")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(anime_data.loc[anime_index, 'Image_URL'], width=220)

    with col2:
        st.markdown(f"### {anime_data.loc[anime_index, 'Name']}")
        st.write(anime_data.loc[anime_index, 'Tags'])

    st.divider()

    # -----------------------------
    # RECOMMENDATIONS
    # -----------------------------
    st.markdown("## 🌟 Recommended For You")

    for i, idx in enumerate(recommended_indices):
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)

            col1, col2 = st.columns([1, 4])

            with col1:
                st.image(anime_data.loc[idx, 'Image_URL'], width=160)

            with col2:
                st.markdown(f"### 🎬 {anime_data.loc[idx, 'Name']}")

                if show_similarity:
                    st.write(f"**Similarity:** {similarity_scores[i]*100:.1f}%")

                if show_tags:
                    tags = anime_data.loc[idx, 'Tags'].split(',')[:6]
                    for tag in tags:
                        st.markdown(
                            f"<span class='tag'>{tag.strip()}</span>",
                            unsafe_allow_html=True
                        )

                st.caption(
                    "🔎 Recommended because it shares similar genres & themes"
                )

            st.markdown('</div>', unsafe_allow_html=True)
