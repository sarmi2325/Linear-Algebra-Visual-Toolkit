import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# PCA via Eigen Decomposition
def pca_eigen(X, n_components=None):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_standardized = (X - mean) / std

    cov_mat = np.cov(X_standardized, rowvar=False)
    st.write("Covariance Matrix", cov_mat)

    eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

    sorted_idx = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorted_idx]
    eigen_vecs = eigen_vecs[:, sorted_idx]

    st.write("Eigen Vectors", eigen_vecs)
    st.write("Eigen Values", eigen_vals)

    if n_components is not None:
        eigen_vecs = eigen_vecs[:, :n_components]

    X_reduced = np.dot(X_standardized, eigen_vecs)
    return X_reduced, eigen_vals

# PCA via SVD
def pca_svd(X, n_components=None):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_standardized = (X - mean) / std

    U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)

    X_reduced = np.dot(U[:, :n_components], np.diag(S[:n_components]))
    eigen_vals = (S ** 2) / (len(X_standardized) - 1)

    st.write("Singular Values", S)
    st.write("Approximated Eigenvalues", eigen_vals)

    return X_reduced, eigen_vals


def pca_visualizer():
    st.title("PCA Visualizer")

    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Data Preview", df.head())

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols)

        if len(features) >= 2:
            X = df[features].dropna().values
            max_components = X.shape[1]

            n_components = st.number_input(
                f"Number of Principal Components (max = {max_components})",
                min_value=1,
                max_value=max_components,
                value=min(2, max_components),
                step=1
            )

            method = st.radio("Choose PCA Method", ["Eigen Decomposition", "SVD"])

            if method == "Eigen Decomposition":
                X_reduced, eigen_vals = pca_eigen(X, n_components)
            else:
                X_reduced, eigen_vals = pca_svd(X, n_components)

            total = np.sum(eigen_vals)
            explained_ratio = eigen_vals / total
            cumulative_ratio = np.cumsum(explained_ratio)

            st.subheader("Explained & Cumulative Variance Table")
            variance_table = pd.DataFrame({
                "Principal Component": [f"PC{i+1}" for i in range(len(explained_ratio))],
                "Explained Variance (%)": [f"{evr*100:.2f}" for evr in explained_ratio],
                "Cumulative Variance (%)": [f"{cum*100:.2f}" for cum in cumulative_ratio]
            })

            st.dataframe(
                variance_table.style.set_properties(**{'text-align': 'center'}),
                use_container_width=True
            )

            # Scree Plot
            fig1, ax1 = plt.subplots()
            ax1.plot(np.arange(1, len(explained_ratio)+1), cumulative_ratio, marker='o', label='Cumulative')
            ax1.bar(np.arange(1, len(explained_ratio)+1), explained_ratio, alpha=0.6, label='Individual')
            ax1.axhline(0.80, color='gray', linestyle='--', label='80% Threshold')
            ax1.axhline(0.90, color='gray', linestyle='-.', label='90% Threshold')
            ax1.set_title("Scree Plot")
            ax1.set_xlabel("Number of Components")
            ax1.set_ylabel("Variance Ratio")
            ax1.set_ylim(0, 1.05)
            ax1.legend()
            st.pyplot(fig1)

            suggested_k = np.argmax(cumulative_ratio >= 0.90) + 1
            st.success(f"Suggestion: Choose at least **{suggested_k} components** to retain ≥90% of the variance.")

            # PCA Scatter Plot
            st.subheader("PCA Scatter Plot")
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

            # Shape Info
            st.subheader("✅ Shapes")
            st.write(f"Original shape: {X.shape}")
            st.write(f"PCA-reduced shape: {X_reduced.shape}")

            # CSV Download
            reduced_df = pd.DataFrame(X_reduced, columns=[f"PC{i+1}" for i in range(n_components)])
            csv = reduced_df.to_csv(index=False).encode('utf-8')
            st.subheader("Download PCA-Reduced Data")
            st.download_button(
                label="Download Compressed CSV",
                data=csv,
                file_name="pca_compressed.csv",
                mime='text/csv'
            )
