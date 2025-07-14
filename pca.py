import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pca_from_scratch(X, n_components=None):
    # Step 1: Manual Standardization (Z-score)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_standardized = (X - mean) / std

    # Step 2: Covariance Matrix
    cov_mat = np.cov(X_standardized, rowvar=False)

    # Step 3: Eigen Decomposition
    eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

    # Step 4: Sort eigenvalues and eigenvectors
    sorted_idx = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorted_idx]
    eigen_vecs = eigen_vecs[:, sorted_idx]

    # Step 5: Keep only top k eigenvectors
    if n_components is not None:
        eigen_vecs = eigen_vecs[:, :n_components]

    # Step 6: Project the data
    X_reduced = np.dot(X_standardized, eigen_vecs)

    return X_reduced, eigen_vals


def pca_visualizer():
    st.title("PCA Visualizer (From Scratch using Eigen Decomposition)")
    #uploading csv
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file:
        #reading the csv
        df = pd.read_csv(file)
        #preview csv
        st.write("Data Preview", df.head())
        #selecting only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols)
        #numeric columns more than 1 
        if len(features) >= 2:
            #drop missing values 
            X = df[features].dropna().values
            #maximum numeric column= so accessing column
            max_components = X.shape[1]
            
            n_components = st.number_input(
                f"Number of Principal Components (max = {max_components})",
                min_value=1,
                max_value=max_components,
                value=min(2, max_components),
                step=1
            )
            
            # PCA from scratch
            X_reduced, eigen_vals = pca_from_scratch(X, n_components)

            # Scree Plot
            st.subheader("🔍 Explained Variance (Eigenvalues)")
            fig1, ax1 = plt.subplots()
            total = np.sum(eigen_vals)
            explained = [(i / total) for i in eigen_vals]
            ax1.plot(np.cumsum(explained), marker='o')
            ax1.set_xlabel("Number of Components")
            ax1.set_ylabel("Cumulative Explained Variance")
            st.pyplot(fig1)

            # 📊 PCA Scatter Plot
            st.subheader("📊 PCA Scatter Plot")
            if n_components >= 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c='blue')
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_zlabel("PC3")
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c='blue')
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                st.pyplot(fig)

            st.subheader("✅ Shapes")
            st.write(f"Original shape: {X.shape}")
            st.write(f"PCA-reduced shape: {X_reduced.shape}")

            # Create a DataFrame of reduced data
            reduced_df = pd.DataFrame(X_reduced, columns=[f"PC{i+1}" for i in range(n_components)])

            # Downloadable CSV
            st.subheader("Download PCA-Reduced Data")
            csv = reduced_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Compressed CSV",
                data=csv,
                file_name="pca_compressed.csv",
                mime='text/csv'
            )
