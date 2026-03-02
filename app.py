import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.title("🎵 Amazon Music Clustering App")
st.write("Unsupervised Learning using KMeans")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("single_genre_artists.csv")
    
    # Drop unnecessary columns
    columns_to_drop = [
        'id_songs', 'name_song', 'id_artists',
        'name_artists', 'release_date', 'genres',
        'time_signature', 'kmeans_cluster', 'dbscan_cluster'
    ]
    
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# Select Features for Clustering
# ---------------------------
features = [
    'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness',
    'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms'
]

X = df[features]

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Sidebar - Select K
# ---------------------------
st.sidebar.header("Model Settings")
k = st.sidebar.slider("Select Number of Clusters (K)", 2, 8, 3)

# ---------------------------
# Apply KMeans
# ---------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, df["cluster"])

st.subheader("Silhouette Score")
st.write(f"Silhouette Score: {round(sil_score,3)}")

# ---------------------------
# Cluster Distribution
# ---------------------------
st.subheader("Cluster Distribution")
st.write(df["cluster"].value_counts())

# ---------------------------
# PCA Visualization
# ---------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], 
                     c=df["cluster"], cmap="viridis")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title("Cluster Visualization (PCA)")
st.pyplot(fig)

# ---------------------------
# Cluster Interpretation
# ---------------------------
st.subheader("Cluster Feature Averages")
cluster_profile = df.groupby("cluster")[features].mean()
st.dataframe(cluster_profile)