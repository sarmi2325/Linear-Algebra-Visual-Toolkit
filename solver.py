import numpy as np
import streamlit as st
import sympy as sp

def gaussian_elimination_steps(A, b, to_rref=False):
    #converting into float for prevent division errors
    A = A.astype(float)
    b = b.astype(float)
    steps = []  # Store LaTeX steps
    #A should be a square matrix
    m, n = A.shape
    if m != n:
        raise ValueError("Only square systems (m = n) can be solved.")
    #loop through columns, find max absolute of each column
    for i in range(n):
        max_row = np.argmax(abs(A[i:, i])) + i
        #the pivot should not be zero, for maintaining the non-singularity
        if A[max_row, i] == 0:
            raise ValueError("Matrix is singular or has no unique solution.")
        #swap the rows,that max_row should be in diagonal
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        #make every element below pivot to zero
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

        # Save current matrix state
        aug = np.hstack((A, b))
        latex_step = r"\begin{bmatrix}" + \
            r"\\ ".join([" & ".join([f"{num:.2f}" for num in row]) for row in aug]) + \
            r"\end{bmatrix}"
        steps.append(latex_step)

    # RREF - Normalize pivot to 1 and clear above
    if to_rref:
        for i in range(n-1, -1, -1):
            A[i] = A[i] / A[i, i]
            b[i] = b[i] / A[i, i]
            for j in range(i):
                factor = A[j, i]
                A[j] -= factor * A[i]
                b[j] -= factor * b[i]
            aug = np.hstack((A, b))
            latex_step = r"\begin{bmatrix}" + \
                r"\\ ".join([" & ".join([f"{num:.2f}" for num in row]) for row in aug]) + \
                r"\end{bmatrix}"
            steps.append(latex_step)

    # Back-substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x, steps

def linear_equation_solver():
    st.title("Linear Equation Solver")

    m = st.number_input("Number of Equations (m)", min_value=1, max_value=10, value=3, help="m must equal n (square system) for a unique solution.")
    n = st.number_input("Number of Unknowns (n)", min_value=1, max_value=10, value=3, help="Choose the number of variables (n) to match equations (m).")

    ref_type = st.selectbox("Form to Display", ["REF (Row Echelon) - Gaussian Elimination", "RREF (Reduced Row Echelon) - Gauss-Jordan elimination"],
                            help="Choose whether to stop at upper triangular form (REF) or fully reduce to RREF.")

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
                st.error("Only square systems (m = n) are supported.")
            else:
                rref = (ref_type == "RREF (Reduced Row Echelon)")
                x, steps = gaussian_elimination_steps(A.copy(), b.copy(), to_rref=rref)
                st.success("✅ Solution Found:")
                st.latex(r"x = " + sp.latex(sp.Matrix(x.reshape(-1, 1))))

                with st.expander("Step-by-step Matrix Transformation"):
                    for idx, step in enumerate(steps):
                        st.latex(f"Step {idx+1}: " + step)
        except ValueError as e:
            st.error(str(e))


