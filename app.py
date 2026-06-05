import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(
    page_title="Hierarchical Clustering",
    layout="wide"
)

st.title("Hierarchical Clustering Visualization")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    numerical_cols = df.select_dtypes(
        include=['int64','float64']
    ).columns

    x_col = st.selectbox(
    "Select Feature 1",
    numerical_cols,
    index=0
)

    y_col = st.selectbox(
    "Select Feature 2",
    numerical_cols,
    index=1 if len(numerical_cols) > 1 else 0
)

    n_clusters = st.slider(
        "Number of Clusters",
        2,
        10,
        5
    )

    X = df[[x_col, y_col]]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )

    labels = model.fit_predict(X_scaled)

    df['Cluster'] = labels

    # Clustered Data
    st.subheader("Clustered Data")
    st.dataframe(df.head())

    # Scatter Plot
    st.subheader("Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        df[x_col],
        df[y_col],
        c=df['Cluster']
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Hierarchical Clusters")

    st.pyplot(fig)

    # Cluster Distribution
    st.subheader("Cluster Distribution")

    cluster_count = df['Cluster'].value_counts()

    fig2, ax2 = plt.subplots()

    ax2.bar(
        cluster_count.index.astype(str),
        cluster_count.values
    )

    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Count")
    ax2.set_title("Cluster Counts")

    st.pyplot(fig2)

    # Dendrogram
    st.subheader("Dendrogram")

    linked = linkage(
        X_scaled,
        method='ward'
    )

    fig3, ax3 = plt.subplots(
        figsize=(12,6)
    )

    dendrogram(
        linked,
        truncate_mode='lastp',
        p=30,
        leaf_rotation=90,
        leaf_font_size=10,
        ax=ax3
    )

    ax3.set_title(
        "Hierarchical Clustering Dendrogram"
    )

    st.pyplot(fig3)
