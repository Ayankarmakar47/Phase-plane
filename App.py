import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import io  # Required for downloading the plot
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application
)

# 1. Set up the Web Page Title
st.title("Phase Portrait Generator")

# Define symbols
x, y = sp.symbols("x y")

# 2. Get user input via Web Text Boxes
expression1 = st.text_input("Enter dx/dt (e.g. -y, x - y, or sin(x)):", "-y")
expression2 = st.text_input("Enter dy/dt (e.g. x, x + y, or cos(y)):", "x")

# 3. Create a Web Button to trigger the calculation
if st.button("Generate Phase Portrait"):
    
    # Parse input
    transformations = standard_transformations + (implicit_multiplication_application,)
    
    # Map 'i', 'I', or 'j' to SymPy's imaginary unit to allow complex inputs natively
    complex_mapping = {'i': sp.I, 'I': sp.I, 'j': sp.I}
    
    try:
        # Pass the local_dict mapping to parser
        func1 = parse_expr(expression1, transformations=transformations, local_dict=complex_mapping)
        func2 = parse_expr(expression2, transformations=transformations, local_dict=complex_mapping)
        
        # ERROR HANDLING: Check if user typed 'sinx' instead of 'sin(x)'
        valid_symbols = {x, y}
        # (sp.I is a constant, so it safely bypasses this check)
        invalid_syms1 = func1.free_symbols - valid_symbols
        invalid_syms2 = func2.free_symbols - valid_symbols
        
        if invalid_syms1 or invalid_syms2:
            bad_syms = ", ".join([str(s) for s in invalid_syms1 | invalid_syms2])
            st.error(f"Unrecognized variable detected: **{bad_syms}**. If you meant a mathematical function, you must use parentheses (e.g., type `sin(x)` instead of `sinx`).")
            st.stop() # Stops the script from crashing NumPy
            
    except Exception as e:
        st.error(f"Error parsing the equations. Please check your syntax. Details: {e}")
        st.stop()

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

    # FINDING THE NATURE OF EQUILIBRIUM POINT (Linear Systems Only)
    if is_f1_linear and is_f2_linear: 
        A, b = sp.linear_eq_to_matrix([func1, func2], [x, y])

        st.write("Matrix A (Coefficients):")
        st.latex(sp.latex(A)) 

        # CONVERT MATRIX A TO NUMPY ARRAY
        A_num = np.array(A, dtype=complex)
        ev = np.linalg.eigvals(A_num)

        # CLASSIFICATION OF FIXED POINTS
        real_part = ev[0].real
        imag_part = ev[0].imag
        
        if np.isclose(imag_part, 0, atol=1e-8): 
            if ev[0].real > 0 and ev[1].real > 0:
                st.success("The nature of the equilibrium point is a Source (Unstable)")
            elif ev[0].real < 0 and ev[1].real < 0:
                st.success("The nature of the equilibrium point is a Sink (Stable)")
            elif ev[0].real * ev[1].real < 0:
                st.success("The equilibrium point is a Saddle point")
            else:
                st.info("The equilibrium point has a zero eigenvalue (Marginal/Degenerate).")
        
        else: # Imaginary part is not zero
            if np.isclose(real_part, 0, atol=1e-8):
                st.success("The Equilibrium point is a Centre")
            else:
                st.success("The Equilibrium point is a Spiral")
                if real_part > 0:
                    st.warning("The spiral is unstable")
                else:
                    st.success("The spiral is stable")


    # PLOTTING THE PHASE PORTRAIT
    x1 = np.linspace(-2.0, 2.0, 20)
    x2 = np.linspace(-2.0, 2.0, 20)
    X1, X2 = np.meshgrid(x1, x2)

    U = f1_num(X1, X2)
    V = f2_num(X1, X2)

    # SAFELY HANDLE COMPLEX OUTPUTS
    if np.iscomplexobj(U) or np.iscomplexobj(V):
        st.warning("Complex outputs detected in the vector field. Plotting the real parts only.")
        U = np.real(U)
        V = np.real(V)

    # Broadcast to prevent crashes if user enters a constant (e.g. dx/dt = 1)
    U = np.broadcast_to(U, X1.shape)
    V = np.broadcast_to(V, X2.shape)
    
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)

    # Replace NaNs with 0
    U = np.nan_to_num(U)
    V = np.nan_to_num(V)

    magnitude = np.hypot(U, V)
    magnitude[magnitude == 0] = 1.0  # Prevent division by zero
    U_norm = U / magnitude
    V_norm = V / magnitude

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.quiver(X1, X2, U_norm, V_norm, color='darkblue', 
              scale=30, width=0.003, headwidth=3, headlength=5, pivot='mid')

    ax.set_title("Phase Portrait", fontsize=14)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12) 
    ax.set_xlim([-2.1, 2.1])
    ax.set_ylim([-2.1, 2.1])
    ax.grid(True, linestyle='--', alpha=0.5)

    # 4. Display the plot on the website
    st.pyplot(fig)

    # 5. Add a Download Button for the Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    
    st.download_button(
        label="Download Plot as PNG",
        data=buf,
        file_name="phase_portrait.png",
        mime="image/png"
    )
