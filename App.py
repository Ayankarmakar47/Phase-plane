
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import io
import warnings

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)
from scipy.integrate import solve_ivp
from scipy.optimize import root

# =========================================================
# 1. STREAMLIT CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Phase Portrait Generator",
    layout="wide"
)
st.title("2D Phase Portrait & Equilibrium Analysis")

st.info(
    "Supports linear and nonlinear systems. "
    "Equilibria are searched symbolically and numerically."
)

# =========================================================
# 2. SIDEBAR INPUTS
# =========================================================
st.sidebar.header("System of Equations")

expression1 = st.sidebar.text_input(
    "Enter dx/dt:",
    value="-y"
)
expression2 = st.sidebar.text_input(
    "Enter dy/dt:",
    value="x"
)

st.sidebar.markdown("---")
st.sidebar.header("Grid Limits")

a = st.sidebar.number_input("x min", value=-2.0, step=0.5)
b = st.sidebar.number_input("x max", value=2.0, step=0.5)
c = st.sidebar.number_input("y min", value=-2.0, step=0.5)
d = st.sidebar.number_input("y max", value=2.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("Trajectory")

show_trajectory = st.sidebar.checkbox(
    "Enable trajectory plotting", value=False
)
x0 = st.sidebar.number_input("Starting x", value=1.0, step=0.1)
y0 = st.sidebar.number_input("Starting y", value=1.0, step=0.1)
trajectory_time = st.sidebar.slider(
    "Trajectory integration time", min_value=1, max_value=100, value=25, step=1
)
trajectories_per_equilibrium = st.sidebar.slider(
    "Trajectories per equilibrium", min_value=4, max_value=24, value=8, step=2
)
perturbation = st.sidebar.number_input(
    "Perturbation around each equilibrium", min_value=0.001,
    max_value=10.0, value=0.15, step=0.05, format="%.3f"
)
plot_equilibrium_trajectories = st.sidebar.checkbox(
    "Plot trajectories around each equilibrium", value=True
)
plot_user_trajectory = st.sidebar.checkbox(
    "Plot trajectory from the chosen starting point", value=True
)

st.sidebar.markdown("---")
st.sidebar.header("Numerical Search")

search_points = st.sidebar.slider(
    "Number of initial guesses per axis",
    min_value=5, max_value=25, value=11
)

# =========================================================
# 3. PARSE THE EQUATIONS
# =========================================================
x, y = sp.symbols("x y")
transformations = (
    standard_transformations
    + (implicit_multiplication_application,)
)

try:
    func1 = parse_expr(
