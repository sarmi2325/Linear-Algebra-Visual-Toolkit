import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

def plot_2d(matrix, vectors):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    for v in vectors:
        ax.quiver(0, 0, *v, angles='xy', scale_units='xy', scale=1, color='blue')
        vt = matrix @ v
        ax.quiver(0, 0, *vt, angles='xy', scale_units='xy', scale=1, color='red')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    st.pyplot(fig)

def plot_3d_plotly(matrix, vectors):
    fig = go.Figure()

    for i, v in enumerate(vectors):
        vt = matrix @ v

        # Original vector (blue)
        fig.add_trace(go.Scatter3d(
            x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
            mode='lines+text',
            line=dict(color='blue', width=6),
            text=[f"", f"v{i+1}"],
            textposition='top right'
        ))

        # Transformed vector (red)
        fig.add_trace(go.Scatter3d(
            x=[0, vt[0]], y=[0, vt[1]], z=[0, vt[2]],
            mode='lines+text',
            line=dict(color='red', width=6, dash='dash'),
            text=[f"", f"T(v{i+1})"],
            textposition='top right'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-10, 10], title='X'),
            yaxis=dict(range=[-10, 10], title='Y'),
            zaxis=dict(range=[-10, 10], title='Z'),
        ),
        title="3D Matrix Transformation (Interactive)",
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig

def transformation_visualizer():
    st.title("Matrix Transformation Visualizer")
    dim = st.selectbox("Select Dimension", [2, 3])
    matrix = []
    #matrix input
    for i in range(dim):
        cols = st.columns(dim)
        row = [cols[j].number_input(f"A[{i+1},{j+1}]", value=1.0 if i == j else 0.0, key=f"A{i}{j}") for j in range(dim)]
        matrix.append(row)
    matrix = np.array(matrix)

    st.markdown("### Vectors to Transform")
    vectors = []
    #vector input
    n = st.slider("Number of vectors", 1, 3, 2)
    for i in range(n):
        cols = st.columns(dim)
        vec = [cols[j].number_input(f"v{i+1}[{j+1}]", value=1.0 if i == j else 0.0, key=f"v_{i}_{j}") for j in range(dim)]
        vectors.append(np.array(vec))
    #assigning plot for 2D and 3D
    if st.button("Apply Transformation"):
        if dim == 2:
            plot_2d(matrix, vectors)
        else:
            fig = plot_3d_plotly(matrix, vectors)
            st.plotly_chart(fig, use_container_width=True)
