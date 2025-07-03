import numpy as np
import streamlit as st
import sympy as sp

def gaussian_elimination(A, b):
    A = A.astype(float)
    b = b.astype(float)
    m, n = A.shape
    if m != n:
        raise ValueError("Only square systems (m = n) can be solved.")

    for i in range(n):
        max_row = np.argmax(abs(A[i:, i])) + i
        if A[max_row, i] == 0:
            raise ValueError("Matrix is singular or no unique solution.")
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

def linear_equation_solver():
    st.title("Linear Equation Solver with Gaussian Elimination")
    m = st.number_input("Number of Equations (m)", min_value=1, max_value=10, value=3)
    n = st.number_input("Number of Unknowns (n)", min_value=1, max_value=10, value=3)

    A = []
    st.markdown("### Coefficient Matrix A")
    for i in range(m):
        row = []
        cols = st.columns(n)
        for j in range(n):
            val = cols[j].number_input(f"A[{i+1},{j+1}]", value=0.0, key=f"A_{i}_{j}")
            row.append(val)
        A.append(row)
    A = np.array(A)

    b = []
    st.markdown("### Right-Hand Side Vector b")
    cols_b = st.columns(m)
    for i in range(m):
        val = cols_b[i].number_input(f"b[{i+1}]", value=0.0, key=f"b_{i}")
        b.append(val)
    b = np.array(b).reshape(-1, 1)

    st.markdown("### Augmented Matrix [A | b]")
    # Display augmented matrix using LaTeX
    latex_matrix = r"\begin{bmatrix}"
    for i in range(m):
        row_left = " & ".join([f"{A[i,j]:.2f}" for j in range(n)])
        row = row_left + r" & \big| & " + f"{b[i,0]:.2f}"
        latex_matrix += row + r" \\ "
    latex_matrix += r"\end{bmatrix}"
    st.latex(latex_matrix)


    if st.button("Solve"):
        try:
            if m != n:
                st.error("Only square systems (m = n) are supported.")
            else:
                x = gaussian_elimination(A.copy(), b.copy())
                st.success("✅ Solution Found:")
                st.latex(r"x = " + sp.latex(sp.Matrix(x.reshape(-1, 1))))
        except ValueError as e:
            st.error(str(e))

