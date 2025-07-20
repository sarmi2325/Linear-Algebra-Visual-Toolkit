import numpy as np
import streamlit as st
import sympy as sp

# Format matrix for LaTeX display
def format_augmented_matrix(A, b):
    latex = r"\begin{bmatrix}"
    for i in range(len(A)):
        row_left = " & ".join([f"{A[i,j]:.2f}" for j in range(A.shape[1])])
        row = row_left + r" & \big| & " + f"{b[i,0]:.2f}"
        latex += row + r" \\ "
    latex += r"\end{bmatrix}"
    return latex

# Gaussian Elimination (REF)
def gaussian_elimination_detailed(A, b):
    A = A.astype(float)
    b = b.astype(float)
    steps = []

    m, n = A.shape
    if m != n:
        raise ValueError("Only square systems (m = n) can be solved.")

    for i in range(n):
        max_row = np.argmax(abs(A[i:, i])) + i
        if A[max_row, i] == 0:
            raise ValueError("Matrix is singular or has no unique solution.")
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
            steps.append(rf"\text{{Swap Row {i+1} $\leftrightarrow$ Row {max_row+1}}}")
            steps.append(format_augmented_matrix(A, b))

        pivot = A[i, i]
        if pivot != 1:
            A[i] = A[i] / pivot
            b[i] = b[i] / pivot
            steps.append(rf"\text{{R_{i+1} = R_{i+1} / {pivot:.2f}}}")
            steps.append(format_augmented_matrix(A, b))

        for j in range(i + 1, n):
            factor = A[j, i]
            if abs(factor) > 1e-8:
                A[j] -= factor * A[i]
                b[j] -= factor * b[i]
                steps.append(rf"\text{{R_{j+1} = R_{j+1} - ({factor:.2f}) × R_{i+1}}}")
                steps.append(format_augmented_matrix(A, b))

    # Back-substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x, steps

# Gauss-Jordan Elimination (RREF)
def gauss_jordan_elimination_detailed(A, b):
    A = A.astype(float)
    b = b.astype(float)
    steps = []

    m, n = A.shape
    if m != n:
        raise ValueError("Only square systems (m = n) can be solved.")

    for i in range(n):
        max_row = np.argmax(abs(A[i:, i])) + i
        if A[max_row, i] == 0:
            raise ValueError("Matrix is singular or has no unique solution.")
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
            steps.append(rf"\text{{Swap Row {i+1} $\leftrightarrow$ Row {max_row+1}}}")
            steps.append(format_augmented_matrix(A, b))

        pivot = A[i, i]
        if pivot != 1:
            A[i] = A[i] / pivot
            b[i] = b[i] / pivot
            steps.append(rf"\text{{R_{i+1} = R_{i+1} / {pivot:.2f}}}")
            steps.append(format_augmented_matrix(A, b))

        for j in range(n):
            if j != i:
                factor = A[j, i]
                if abs(factor) > 1e-8:
                    A[j] -= factor * A[i]
                    b[j] -= factor * b[i]
                    steps.append(rf"\text{{R_{j+1} = R_{j+1} - ({factor:.2f}) × R_{i+1}}}")
                    steps.append(format_augmented_matrix(A, b))

    return b.flatten(), steps  # In RREF, b becomes the solution

# Streamlit App
def linear_equation_solver():
    st.title("Linear Equation Solver")
    st.write("Solve systems using Gaussian (REF) or Gauss-Jordan (RREF) Elimination with step-by-step matrix view.")

    m = st.number_input("Number of Equations (m)", min_value=1, max_value=10, value=2)
    n = st.number_input("Number of Unknowns (n)", min_value=1, max_value=10, value=2)

    ref_type = st.selectbox("Elimination Method", 
                            ["REF (Row Echelon) - Gaussian Elimination", 
                             "RREF (Reduced Row Echelon) - Gauss-Jordan Elimination"])

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
    st.latex(format_augmented_matrix(A, b))

    if st.button("🔍 Solve"):
        try:
            if m != n:
                st.error("❌ Only square systems (m = n) are supported.")
                return

            if ref_type == "REF (Row Echelon) - Gaussian Elimination":
                x, steps = gaussian_elimination_detailed(A.copy(), b.copy())
            else:
                x, steps = gauss_jordan_elimination_detailed(A.copy(), b.copy())

            st.success("✅ Solution Found:")
            st.latex(r"x = " + sp.latex(sp.Matrix(x.reshape(-1, 1))))

            with st.expander("Step-by-step Matrix Transformation"):
                for i in range(0, len(steps), 2):
                    st.markdown(f"### Step {i//2 + 1}")
                    st.markdown(f"$$ {steps[i]} $$")         
                    st.markdown(f"$$ {steps[i+1]} $$")      


        except ValueError as e:
            st.error(f"⚠️ {str(e)}")



