import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application
)

# 1. Streamlit Page Configuration
st.set_page_config(page_title="3D Phase Portrait Generator", layout="wide")
st.title("3D Phase Portrait & System Analysis")

# 2. Sidebar for Inputs
st.sidebar.header("System of Equations")
expression1 = st.sidebar.text_input("Enter dx/dt (e.g. -x + y):", value="-x + y")
expression2 = st.sidebar.text_input("Enter dy/dt (e.g. -y + z):", value="-y + z")
expression3 = st.sidebar.text_input("Enter dz/dt (e.g. x - z):", value="x - z")

# 3. Define symbols and parse input
x, y, z = sp.symbols("x y z")
transformations = standard_transformations + (implicit_multiplication_application,)

try:
    func1 = parse_expr(expression1, transformations=transformations)
    func2 = parse_expr(expression2, transformations=transformations)
    func3 = parse_expr(expression3, transformations=transformations)
except Exception as e:
    st.error(f"Error parsing equations. Please check your syntax. ({e})")
    st.stop()

# Convert SymPy expressions to NumPy functions
f1_num = sp.lambdify((x, y, z), func1, modules='numpy')
f2_num = sp.lambdify((x, y, z), func2, modules='numpy')
f3_num = sp.lambdify((x, y, z), func3, modules='numpy')

def check_linear(expr, *vars):
    try:
        return sp.Poly(expr, *vars).is_linear
    except sp.PolynomialError:
        return False

is_lin_1 = check_linear(func1, x, y, z)
is_lin_2 = check_linear(func2, x, y, z)
is_lin_3 = check_linear(func3, x, y, z)

# 4. Create Dashboard Layout
col1, col2 = st.columns([1, 2.5])

with col1:
    st.subheader("System Analysis")
    st.write(f"**dx/dt is linear:** {is_lin_1}")
    st.write(f"**dy/dt is linear:** {is_lin_2}")
    st.write(f"**dz/dt is linear:** {is_lin_3}")

    if is_lin_1 and is_lin_2 and is_lin_3: 
        A, b = sp.linear_eq_to_matrix([func1, func2, func3], [x, y, z])

        st.markdown("**Matrix A (Coefficients):**")
        st.latex(sp.latex(A))

        A_num = np.array(A, dtype=float)
        ev = np.linalg.eigvals(A_num)
        
        st.write("**Eigenvalues:**")
        st.write(f"`λ1 = {np.round(ev[0], 3)}`")
        st.write(f"`λ2 = {np.round(ev[1], 3)}`")
        st.write(f"`λ3 = {np.round(ev[2], 3)}`")

        # 3D Classification logic
        reals = ev.real
        imags = ev.imag
        has_imaginary = np.any(np.abs(imags) > 1e-8) 
        
        classification = ""
        if np.all(reals < -1e-8):
            if has_imaginary:
                classification = "Spiral Sink (Stable)"
            else:
                classification = "Sink (Stable)"
                
        elif np.all(reals > 1e-8):
            if has_imaginary:
                classification = "Spiral Source (Unstable)"
            else:
                classification = "Source (Unstable)"
                
        elif np.any(reals > 1e-8) and np.any(reals < -1e-8):
            if has_imaginary:
                classification = "Spiral Saddle (Unstable)"
            else:
                classification = "Saddle point (Unstable)"
                
        elif np.all(np.abs(reals) <= 1e-8) and has_imaginary:
             classification = "Center (Neutral/Oscillatory)"
             
        if classification:
            st.success(f"**Equilibrium Nature:**\n{classification}")

with col2:
    st.subheader("3D Phase Portrait")
    
    # Set up the 3D grid 
    grid_res = 8
    x1 = np.linspace(-2.0, 2.0, grid_res)
    x2 = np.linspace(-2.0, 2.0, grid_res)
    x3 = np.linspace(-2.0, 2.0, grid_res)
    X1, X2, X3 = np.meshgrid(x1, x2, x3)

    # Compute vector components numerically
    U = f1_num(X1, X2, X3)
    V = f2_num(X1, X2, X3)
    W = f3_num(X1, X2, X3)

    # Broadcast to prevent plotting crashes on constant inputs
    U = np.broadcast_to(U, X1.shape).astype(float)
    V = np.broadcast_to(V, X2.shape).astype(float)
    W = np.broadcast_to(W, X3.shape).astype(float)

    # Normalize arrows
    magnitude = np.sqrt(U**2 + V**2 + W**2)
    magnitude[magnitude == 0] = 1.0  
    U_norm = U 
    V_norm = V 
    W_norm = W 

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(X1, X2, X3, U_norm, V_norm, W_norm, color='darkblue', 
              length=0.4, normalize=False, alpha=0.6)

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_zlabel("$z$", fontsize=12)

    ax.set_xlim([-2.1, 2.1])
    ax.set_ylim([-2.1, 2.1])
    ax.set_zlim([-2.1, 2.1])

    # Render plot in Streamlit
    st.pyplot(fig)
    # --- DOWNLOAD BUTTON LOGIC ---
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    
    st.download_button(
        label="📥 Download Plot as PNG",
        data=buf,
        file_name="2D_phase_portrait_custom_grid.png",
        mime="image/png"
    )
