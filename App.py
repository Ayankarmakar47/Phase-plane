import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import io # <-- Added for image downloading
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application
)
from scipy.integrate import solve_ivp

# 1. Streamlit Page Configuration
st.set_page_config(page_title="3D Phase Portrait Generator", layout="wide")
st.title("3D Phase Portrait & System Analysis")

# 2. Sidebar for Inputs
st.sidebar.header("System of Equations")
expression1 = st.sidebar.text_input("Enter dx/dt (e.g. -x + y):", value="-x + y")
expression2 = st.sidebar.text_input("Enter dy/dt (e.g. -y + z):", value="-y + z")
expression3 = st.sidebar.text_input("Enter dz/dt (e.g. x - z):", value="x - z")

st.sidebar.markdown("---")
st.sidebar.header("Plot Trajectory")
show_trajectory = st.sidebar.checkbox("Enable Trajectory Plotting", value=False)
x0 = st.sidebar.number_input("Starting x coordinate", value=1.0, step=0.1)
y0 = st.sidebar.number_input("Starting y coordinate", value=1.0, step=0.1)
z0 = st.sidebar.number_input("Starting z coordinate", value=1.0, step=0.1)

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
    
    # Set up the 3D grid (Resolution kept low at 8x8x8 to prevent browser lag)
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
    U_norm = U / magnitude
    V_norm = V / magnitude
    W_norm = W / magnitude

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(X1, X2, X3, U_norm, V_norm, W_norm, color='darkblue', 
              length=0.4, normalize=False, alpha=0.3)

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_zlabel("$z$", fontsize=12)

    ax.set_xlim([-2.1, 2.1])
    ax.set_ylim([-2.1, 2.1])
    ax.set_zlim([-2.1, 2.1])

    # --- TRAJECTORY PLOTTING LOGIC ---
    if show_trajectory:
        def ode_system(t, state):
            curr_x, curr_y, curr_z = state
            dx = float(f1_num(curr_x, curr_y, curr_z))
            dy = float(f2_num(curr_x, curr_y, curr_z))
            dz = float(f3_num(curr_x, curr_y, curr_z))
            return [dx, dy, dz]
            
        t_eval_f = np.linspace(0, 15, 1000)
        t_eval_b = np.linspace(0, -15, 1000)
        
        try:
            sol_f = solve_ivp(ode_system, (0, 15), [x0, y0, z0], t_eval=t_eval_f, method='RK45')
            ax.plot(sol_f.y[0], sol_f.y[1], sol_f.y[2], color='red', linewidth=2.5, label='Forward Time')
            
            sol_b = solve_ivp(ode_system, (0, -15), [x0, y0, z0], t_eval=t_eval_b, method='RK45')
            ax.plot(sol_b.y[0], sol_b.y[1], sol_b.y[2], color='magenta', linewidth=2.5, label='Backward Time')
            
            # Plot the starting point
            ax.scatter([x0], [y0], [z0], color='black', s=50, label='Start Point')
            ax.legend(loc='upper left')
            
        except Exception as e:
            st.warning(f"Failed to calculate trajectory: {e}")

    # Render plot in Streamlit
    st.pyplot(fig)
    
    # --- DOWNLOAD BUTTON LOGIC ---
    # Save the figure to an in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0) # Reset buffer position to the beginning
    
    # Create the download button
    st.download_button(
        label="📥 Download Plot as PNG",
        data=buf,
        file_name="3D_phase_portrait.png",
        mime="image/png"
    )
