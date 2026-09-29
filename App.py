import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application
)

# 1. Set up the Web Page Title
st.title("Phase Portrait Generator")

# Define symbols
x, y = sp.symbols("x y")

# 2. Get user input via Web Text Boxes (with default examples)
expression1 = st.text_input("Enter dx/dt (e.g. -y or x - y):", "-y")
expression2 = st.text_input("Enter dy/dt (e.g. x or x + y):", "x")

# 3. Create a Web Button to trigger the calculation
if st.button("Generate Phase Portrait"):
    
    # Parse input
    transformations = standard_transformations + (implicit_multiplication_application,)
    
    try:
        func1 = parse_expr(expression1, transformations=transformations)
        func2 = parse_expr(expression2, transformations=transformations)
    except Exception as e:
        st.error(f"Error parsing the equations. Please check your syntax. Details: {e}")
        st.stop() # Stops execution if there's a typo

    # Convert SymPy expressions to NumPy functions
    f1_num = sp.lambdify((x, y), func1, modules='numpy')
    f2_num = sp.lambdify((x, y), func2, modules='numpy')

    # CHECK FOR LINEARITY
    def check_linear(expr, *vars):
        try:
            return sp.Poly(expr, *vars).is_linear
        except sp.PolynomialError:
            return False

    is_f1_linear = check_linear(func1, x, y)
    is_f2_linear = check_linear(func2, x, y)

    st.write(f"Is $dx/dt$ linear? **{is_f1_linear}**")  
    st.write(f"Is $dy/dt$ linear? **{is_f2_linear}**")  

    # FINDING THE NATURE OF EQUILIBRIUM POINT
    if is_f1_linear and is_f2_linear: 
        A, b = sp.linear_eq_to_matrix([func1, func2], [x, y])

        st.write("Matrix A (Coefficients):")
        # Renders the matrix nicely formatted using LaTeX on the website
        st.latex(sp.latex(A)) 

        # CONVERT MATRIX A TO NUMPY ARRAY
        A_num = np.array(A, dtype=float)
        ev = np.linalg.eigvals(A_num)

        # CLASSIFICATION OF FIXED POINTS
        if ev[0].imag == 0:
            if ev[0].real > 0 and ev[1].real > 0:
                st.success("The nature of the equilibrium point is a Source (Unstable)")
            elif ev[0].real < 0 and ev[1].real < 0:
                st.success("The nature of the equilibrium point is a Sink (Stable)")
            elif ev[0].real * ev[1].real < 0:
                st.success("The equilibrium point is a Saddle point")
        
        if ev[0].imag != 0:
            if ev[0].real == 0:
                st.success("The Equilibrium point is a Centre")
            elif ev[0].real != 0:
                st.success("The Equilibrium point is a Spiral")
                if ev[0].real > 0:
                    st.warning("The spiral is unstable")
                else:
                    st.success("The spiral is stable")

    # PLOTTING THE PHASE PORTRAIT
    x1 = np.linspace(-2.0, 2.0, 20)
    x2 = np.linspace(-2.0, 2.0, 20)
    X1, X2 = np.meshgrid(x1, x2)

    U = f1_num(X1, X2)
    V = f2_num(X1, X2)

    # BUG FIX: Required in Streamlit to prevent plotting crashes on constant inputs
    U = np.broadcast_to(U, X1.shape).astype(float)
    V = np.broadcast_to(V, X2.shape).astype(float)

    magnitude = np.hypot(U, V)
    magnitude[magnitude == 0] = 1.0  # Prevent division by zero
    U_norm = U / magnitude
    V_norm = V / magnitude

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.quiver(X1, X2, U_norm, V_norm, color='darkblue', 
              scale=30, width=0.003, headwidth=3, headlength=5, pivot='mid')

    ax.set_title("Phase Portrait", fontsize=14)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12) 
    ax.set_xlim([-2.1, 2.1])
    ax.set_ylim([-2.1, 2.1])
    ax.grid(True, linestyle='--', alpha=0.5)

    # 4. Display the plot on the website instead of a local window
    st.pyplot(fig)
