import streamlit as st
from solver import linear_equation_solver
from transform import transformation_visualizer
from pca import pca_visualizer

st.set_page_config(page_title='Linear Algebra Playground',layout='wide')

st.sidebar.title('Linear Algebra Playground')
module = st.sidebar.radio('choose a module',[
    "Linear Equation Solver",
    "Matrix Transformation Visualizer",
    "PCA Visualizer"
])

if module == 'Linear Equation Solver':
    linear_equation_solver()
elif module == "Matrix Transformation Visualizer":
    transformation_visualizer()
elif module == "PCA Visualizer":
    pca_visualizer()

