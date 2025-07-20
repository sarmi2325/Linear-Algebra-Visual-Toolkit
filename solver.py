import numpy as np
import streamlit as st
import sympy as sp

def gaussian_elimination_steps(A, b, to_rref=False):
    A = A.astype(float)
    b = b.astype(float)
    ref_steps = []
    rref_steps = []

    m, n = A.shape
    if m != n:
        raise ValueError("Only square systems (m = n) can be solved.")

    # Forward elimination (REF)
    for i in range(n):
        max_row = np.argmax(abs(A[i:, i])) + i
        if A[max_row, i] == 0:
            raise ValueError("Matrix is singular or has no unique solution.")
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

        aug = np.hstack((A, b))
        latex_step = r"\text{REF Step " + f"{i+1}" + r"}: \quad \begin{bmatrix}" + \
            r"\\ ".join([" & ".join([f"{num:.2f}" for num in row]) for row in aug]) + \
            r"\end{bmatrix}"
        ref_steps.append(latex_step)

    # Stop here if only REF requested
    if not to_rref:
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        return x, ref_steps

    # Continue to Gauss-Jordan elimination (RREF)
    for i in range(n-1, -1, -1):
        divisor = A[i, i]
        A[i] = A[i] / divisor
        b[i] = b[i] / divisor
        for j in range(i):
            factor = A[j, i]
            A[j] -= factor * A[i]
            b[j] -= factor * b[i]

        aug = np.hstack((A, b))
        latex_step = r"\text{RREF Step " + f"{n - i}" + r"}: \quad \begin{bmatrix}" + \
            r"\\ ".join([" & ".join([f"{num:.2f}" for num in row]) for row in aug]) + \
            r"\end{bmatrix}"
        rref_steps.append(latex_step)

    x = b.flatten()  # After RREF, x is directly b
    return x, rref_steps


def linear_equation_solver():
    st.title("Linear Equation Solver (Gaussian / Gauss-Jordan Elimination)")

    m = st.number_input("Number of Equations (m)", min_value=1, max_value=10, value=2)
    n = st.number_input("Number of Unknowns (n)", min_value=1, max_value=10, value=2)

    ref_type = st.selectbox("Form to Display", 
                            ["REF (Row Echelon) - Gaussian Elimination", 
                             "RREF (Reduced Row Echelon) - Gauss-Jordan elimination"])

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

    # Display Augmented Matrix
    st.markdown("### Augmented Matrix $[A | b]$")
    latex_matrix = r"\begin{bmatrix}"
    for i in range(m):
        row_left = " & ".join([f"{A[i,j]:.2f}" for j in range(n)])
        row = row_left + r" & \big| & " + f"{b[i,0]:.2f}"
        latex_matrix += row + r" \\ "
    latex_matrix += r"\end{bmatrix}"
    st.latex(latex_matrix)

    if st.button("🔍 Solve"):
        try:
            if m != n:
                st.error("❌ Only square systems (m = n) are supported.")
            else:
                to_rref = (ref_type == "RREF (Reduced Row Echelon) - Gauss-Jordan elimination")
                x, steps = gaussian_elimination_steps(A.copy(), b.copy(), to_rref=to_rref)

                st.success("✅ Solution Found:")
                st.latex(r"x = " + sp.latex(sp.Matrix(x.reshape(-1, 1))))

                with st.expander("Step-by-step Matrix Transformation"):
                    for step in steps:
                        st.latex(step)

        except ValueError as e:
            st.error(f"⚠️ {str(e)}")

