🎵 Amazon Music Clustering Using Machine Learning
Audio Feature–Based Unsupervised Learning Project

📌 Project Summary

This project applies unsupervised machine learning techniques (K-Means and DBSCAN) to group Amazon Music tracks based on audio features, including:

Danceability
Energy
Loudness
Speechiness
Acousticness
Instrumentalness
Liveness
Valence
Tempo
Duration

The goal is to understand natural patterns in music and support use cases such as:

🎧 playlist recommendations
🎵 music similarity search
🧩 genre-based clustering
📊 artist trend analysis

📁 Repository Structure
│
├── data/                       # Dataset used for clustering
├── notebooks/                  # Jupyter Notebook with full workflow
├── outputs/                    # Plots generated from analysis
├── presentation/               # Final PPT for project submission
├── report/                     # Written report (PDF/DOCX)
├── src/                        # Python scripts (optional)
└── README.md                   # Project documentation

🚀 Technologies Used
Component	Library
Data Analysis	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Clustering	Scikit-Learn
Dimensionality Reduction	PCA
Environment	Google Colab
Presentation	Google Slides / PowerPoint

🧹 Data Preprocessing

Load dataset
Drop unnecessary columns (id_songs, name_song, etc.)
Select audio features
Handle missing values
Scale features using StandardScaler

🔍 Exploratory Analysis

PCA Plot (Before Clustering)
Shows the natural shape of the data in 2D.
Correlation Study
Audio features show meaningful relationships:
Danceability ↔ Energy
Acousticness ↔ Instrumentalness

🤖 Clustering Techniques Applied

1️⃣ K-Means Clustering

Tested K = 2 to 10
Elbow method suggested K = 3
Silhouette score was highest at K = 3
Final K-Means Clusters
Cluster	Characteristics
C0	Low energy, acoustic, softer music
C1	High tempo, energetic, upbeat tracks
C2	High speechiness, unique vocal-heavy tracks

🌋 DBSCAN Clustering

Tried eps values (0.5, 1.0, 1.5, 2.0, 3.0)
Best meaningful structure at eps = 2.0
Identified many noise points → reflects dataset density

📊 Key Visualizations (found in /outputs/)

PCA Before Clustering
Elbow Curve
Silhouette Score Results
PCA K-Means (K=3) Clusters
Feature Means (Bar Chart)
Feature Heatmap
DBSCAN PCA Plot

📌 Conclusion

✔ Best performing clustering: K-Means with K=3
✔ Clear separation of musical styles based on audio features
✔ DBSCAN reveals noise and density variations
✔ PCA shows strong cluster structure after K-Means
✔ Useful for recommendation engines & playlist generation

📝 Future Enhancements

Try Gaussian Mixture Models
Apply t-SNE for deeper visualization
Use Spotify API for real-time audio track expansion
Build a recommendation system using nearest neighbors
