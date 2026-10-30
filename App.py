import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import io  # Added for image downloading
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application
)
from scipy.integrate import solve_ivp

# 1. Streamlit Page Configuration
st.set_page_config(page_title="Phase Portrait Generator", layout="wide")
st.title("2D Phase Portrait & Trajectory Generator")

# 2. Sidebar for Inputs
st.sidebar.header("System of Equations")
expression1 = st.sidebar.text_input("Enter dx/dt (e.g. -y or x - y):", value="-y")
expression2 = st.sidebar.text_input("Enter dy/dt (e.g. x or x + y):", value="x")

st.sidebar.markdown("---")
st.sidebar.header("Grid Limits (a, b, c, d)")
a = st.sidebar.number_input("x min (a)", value=-2.0, step=0.5)
b = st.sidebar.number_input("x max (b)", value=2.0, step=0.5)
c = st.sidebar.number_input("y min (c)", value=-2.0, step=0.5)
d = st.sidebar.number_input("y max (d)", value=2.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("Plot Trajectory")
show_trajectory = st.sidebar.checkbox("Enable Trajectory Plotting", value=False)
x0 = st.sidebar.number_input("Starting x coordinate", value=1.0, step=0.1)
y0 = st.sidebar.number_input("Starting y coordinate", value=1.0, step=0.1)

# 3. Define symbols and parse input
x, y = sp.symbols("x y")
transformations = standard_transformations + (implicit_multiplication_application,)

try:
    func1 = parse_expr(expression1, transformations=transformations)
    func2 = parse_expr(expression2, transformations=transformations)
except Exception as e:
    st.error(f"Error parsing equations. Please check your syntax. ({e})")
    st.stop()

# Convert SymPy expressions to NumPy functions
f1_num = sp.lambdify((x, y), func1, modules='numpy')
f2_num = sp.lambdify((x, y), func2, modules='numpy')

def check_linear(expr, *vars):
    try:
        return sp.Poly(expr, *vars).is_linear
    except sp.PolynomialError:
        return False

is_lin_1 = check_linear(func1, x, y)
is_lin_2 = check_linear(func2, x, y)

# 4. Create Dashboard Layout (Two Columns)
col1, col2 = st.columns([1, 2.5])

with col1:
    st.subheader("System Analysis")
    st.write(f"**dx/dt is linear:** {is_lin_1}")
    st.write(f"**dy/dt is linear:** {is_lin_2}")

    # Equilibrium Point Classification
    if is_lin_1 and is_lin_2: 
        A, b_vec = sp.linear_eq_to_matrix([func1, func2], [x, y])
        
        st.markdown("**Matrix A (Coefficients):**")
        # Render Matrix A nicely using LaTeX in Streamlit
        st.latex(sp.latex(A))

        A_num = np.array(A, dtype=float)
        ev = np.linalg.eigvals(A_num)
        
        st.write(f"**Eigenvalues:** `{np.round(ev[0], 3)}`, `{np.round(ev[1], 3)}`")

        # Classification Logic
        classification = ""
        if ev[0].imag == 0:
            if ev[0].real > 0 and ev[1].real > 0:
                classification = "Source (Unstable)"
            elif ev[0].real < 0 and ev[1].real < 0:
                classification = "Sink (Stable)"
            elif ev[0].real * ev[1].real < 0:
                classification = "Saddle point"
        else:
            if ev[0].real == 0:
                classification = "Center (Neutral)"
            else:
                if ev[0].real > 0:
                    classification = "Spiral (Unstable)"
                else:
                    classification = "Spiral (Stable)"
        
        st.success(f"**Equilibrium Nature:**\n{classification}")

with col2:
    st.subheader("Phase Portrait")

    # Set up the grid using user inputs (a, b, c, d)
    x1 = np.linspace(a, b, 20)
    x2 = np.linspace(c, d, 20)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute vector components
    U = f1_num(X1, X2)
    V = f2_num(X1, X2)

    # Broadcast to prevent plotting crashes on constant inputs
    U = np.broadcast_to(U, X1.shape).astype(float)
    V = np.broadcast_to(V, X2.shape).astype(float)

    # Calculate magnitude and normalize vectors for better visualization
    magnitude = np.hypot(U, V)
    magnitude[magnitude == 0] = 1.0  
    U_norm = U 
    V_norm = V 

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.quiver(X1, X2, U_norm, V_norm, color='darkblue', 
              scale=30, width=0.003, headwidth=3, headlength=5, pivot='mid')

    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    
    # Update plot limits dynamically based on the grid bounds
    ax.set_xlim([a - 0.1, b + 0.1])
    ax.set_ylim([c - 0.1, d + 0.1])
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- TRAJECTORY PLOTTING LOGIC ---
    if show_trajectory:
        def ode_system(t, state):
            curr_x, curr_y = state
            dx = float(f1_num(curr_x, curr_y))
            dy = float(f2_num(curr_x, curr_y))
            return [dx, dy]
            
        t_eval_f = np.linspace(0, 25, 2000)
        t_eval_b = np.linspace(0, -25, 2000)
        
        try:
            sol_f = solve_ivp(ode_system, (0, 25), [x0, y0], t_eval=t_eval_f, method='RK45')
            ax.plot(sol_f.y[0], sol_f.y[1], color='red', linewidth=2, label='Forward Time')
            
            sol_b = solve_ivp(ode_system, (0, -25), [x0, y0], t_eval=t_eval_b, method='RK45')
            ax.plot(sol_b.y[0], sol_b.y[1], color='magenta', linewidth=2, label='Backward Time')
            
            # Plot the starting point
            ax.plot(x0, y0, 'ko', markersize=6, label='Start Point')
            ax.legend(loc='upper right')
            
        except Exception as e:
            st.warning(f"Failed to calculate trajectory: {e}")

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
