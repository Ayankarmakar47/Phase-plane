
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
        expression1,
        local_dict={"x": x, "y": y},
        transformations=transformations
    )
    func2 = parse_expr(
        expression2,
        local_dict={"x": x, "y": y},
        transformations=transformations
    )

    # Restrict the application to real-valued expressions
    if func1.has(sp.I) or func2.has(sp.I):
        raise ValueError("Use real-valued equations.")

    f1_num = sp.lambdify((x, y), func1, modules="numpy")
    f2_num = sp.lambdify((x, y), func2, modules="numpy")

except Exception as e:
    st.error(f"Could not parse the equations: {e}")
    st.stop()

if a >= b or c >= d:
    st.error("The minimum grid limits must be smaller than the maximum limits.")
    st.stop()


def evaluate_scalar(func, xx, yy):
    """Evaluate a vector-field component as a real scalar."""
    with np.errstate(all="ignore"):
        value = np.asarray(func(xx, yy))

    if value.size != 1:
        raise ValueError("The equation did not return a scalar.")

    value = complex(value.item())

    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise ValueError("Non-finite value.")

    if abs(value.imag) > 1e-8:
        raise ValueError("The vector field is not real-valued here.")

    return float(value.real)


def vector_field(xx, yy):
    return np.array([
        evaluate_scalar(f1_num, xx, yy),
        evaluate_scalar(f2_num, xx, yy)
    ], dtype=float)


def check_linear(expr):
    """True for affine expressions, including constants."""
    try:
        poly = sp.Poly(expr, x, y)
        return poly.total_degree() <= 1
    except sp.PolynomialError:
        return False


is_linear = check_linear(func1) and check_linear(func2)

# =========================================================
# 4. FIXED-POINT SEARCH
# =========================================================

def inside_grid(point, tolerance=1e-7):
    xx, yy = point
    return (
        a - tolerance <= xx <= b + tolerance
        and c - tolerance <= yy <= d + tolerance
    )


def residual_is_small(point, tolerance=1e-7):
    try:
        residual = vector_field(point[0], point[1])
        return np.linalg.norm(residual, ord=np.inf) < tolerance
    except Exception:
        return False


def add_unique(points, candidate, tolerance=1e-5):
    """Avoid storing the same root more than once."""
    candidate = np.asarray(candidate, dtype=float)

    if not np.all(np.isfinite(candidate)):
        return

    if not inside_grid(candidate):
        return

    if not residual_is_small(candidate):
        return

    if not any(
        np.linalg.norm(candidate - old) < tolerance
        for old in points
    ):
        points.append(candidate)


def find_fixed_points():
    points = []
    exact_count = 0
    symbolic_message = ""

    # ---- A. Attempt exact symbolic solving ----
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solutions = sp.solve(
                [func1, func2],
                (x, y),
                dict=True
            )

        for sol in solutions:
            if x not in sol or y not in sol:
                continue

            sx = sp.simplify(sol[x])
            sy = sp.simplify(sol[y])

            # Discard parametric or otherwise unresolved solutions
            if sx.free_symbols or sy.free_symbols:
                continue

            if sx.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                continue
            if sy.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                continue

            try:
                nx = complex(sp.N(sx))
                ny = complex(sp.N(sy))

                if not (
                    np.isfinite(nx.real)
                    and np.isfinite(ny.real)
                    and abs(nx.imag) < 1e-8
                    and abs(ny.imag) < 1e-8
                ):
                    continue

                candidate = np.array([nx.real, ny.real])
                before = len(points)
                add_unique(points, candidate)

                if len(points) > before:
                    exact_count += 1

            except (TypeError, ValueError, OverflowError):
                continue

    except Exception as e:
        symbolic_message = str(e)

    # ---- B. Multistart numerical root-finding ----
    # This may discover additional roots not returned symbolically.
    x_guesses = np.linspace(a, b, search_points)
    y_guesses = np.linspace(c, d, search_points)

    def residual(z):
        return vector_field(z[0], z[1])

    for gx in x_guesses:
        for gy in y_guesses:
            try:
                result = root(
                    residual,
                    np.array([gx, gy]),
                    method="hybr"
                )

                if result.success:
                    add_unique(points, result.x)

                # Some valid roots are returned even if the solver
                # reports that its convergence criterion was not met.
                elif residual_is_small(result.x):
                    add_unique(points, result.x)

            except Exception:
                continue

    points.sort(key=lambda p: (p[0], p[1]))

    return points, exact_count, symbolic_message


# =========================================================
# 5. JACOBIAN AND LOCAL CLASSIFICATION
# =========================================================

jacobian = sp.Matrix([
    [sp.diff(func1, x), sp.diff(func1, y)],
    [sp.diff(func2, x), sp.diff(func2, y)]
])


def jacobian_at(point):
    xx, yy = point
    values = jacobian.subs({x: xx, y: yy}).evalf()
    return np.array(values, dtype=float)


def classify_equilibrium(J):
    eigenvalues = np.linalg.eigvals(J)
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)

    tol = 1e-7

    if np.all(real_parts < -tol):
        if np.any(np.abs(imag_parts) > tol):
            label = "Spiral sink (locally asymptotically stable)"
        else:
            label = "Sink (locally asymptotically stable)"

    elif np.all(real_parts > tol):
        if np.any(np.abs(imag_parts) > tol):
            label = "Spiral source (unstable)"
        else:
            label = "Source (unstable)"

    elif np.any(real_parts < -tol) and np.any(real_parts > tol):
        label = "Saddle (unstable)"

    elif np.all(np.abs(real_parts) <= tol):
        label = (
            "Non-hyperbolic: linearization is inconclusive; "
            "a center is possible"
        )

    else:
        label = (
            "Non-hyperbolic or borderline: "
            "nonlinear analysis may be needed"
        )

    return eigenvalues, label


fixed_points, exact_count, symbolic_message = find_fixed_points()

# =========================================================
# 6. SYSTEM ANALYSIS
# =========================================================

col1, col2 = st.columns([1, 2.5])

with col1:
    st.subheader("System Analysis")

    st.latex(r"\dot{x}=f(x,y)")
    st.latex(r"\dot{y}=g(x,y)")

    st.write(f"**First equation affine-linear:** {check_linear(func1)}")
    st.write(f"**Second equation affine-linear:** {check_linear(func2)}")

    st.markdown("**Equilibrium condition**")
    st.latex(r"f(x^*,y^*)=0,\quad g(x^*,y^*)=0")

    st.write(
        f"**Symbolically found within the grid:** {exact_count}"
    )
    st.write(
        f"**Distinct equilibria found in total:** {len(fixed_points)}"
    )

    if symbolic_message:
        st.caption(
            "The symbolic solver could not complete normally. "
            "Numerical search was still attempted."
        )

    if fixed_points:
        for i, point in enumerate(fixed_points, start=1):
            xx, yy = point
            st.markdown(f"### Fixed point {i}")
            st.write(f"x* = {xx:.8g}")
            st.write(f"y* = {yy:.8g}")

            try:
                J = jacobian_at(point)
                eigvals, nature = classify_equilibrium(J)

                st.write("Jacobian matrix:")
                st.latex(sp.latex(jacobian.subs({
                    x: sp.Float(xx),
                    y: sp.Float(yy)
                })))

                st.write(
                    "Eigenvalues: "
                    + ", ".join(
                        f"{ev.real:.5g}"
                        + (
                            f"{ev.imag:+.5g}i"
                            if abs(ev.imag) > 1e-8 else ""
                        )
                        for ev in eigvals
                    )
                )

                st.write(f"**Local classification:** {nature}")

            except Exception as e:
                st.warning(f"Could not classify this point: {e}")

    else:
        st.warning(
            "No fixed points were found in the selected grid. "
            "This does not prove that no equilibria exist."
        )

    st.caption(
        "Classification is based on Jacobian eigenvalues. "
        "For non-hyperbolic points, higher-order analysis may be required."
    )

# =========================================================
# 7. PHASE PORTRAIT
# =========================================================

with col2:
    st.subheader("Phase Portrait")

    x_grid = np.linspace(a, b, 20)
    y_grid = np.linspace(c, d, 20)
    X, Y = np.meshgrid(x_grid, y_grid)

    U = np.full_like(X, np.nan, dtype=float)
    V = np.full_like(Y, np.nan, dtype=float)

    # Evaluate field point-by-point so singularities do not crash plotting
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                U[i, j], V[i, j] = vector_field(X[i, j], Y[i, j])
            except Exception:
                pass

    magnitude = np.hypot(U, V)
    magnitude[magnitude == 0] = 1.0

    U_norm = U / magnitude
    V_norm = V / magnitude

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.quiver(
        X, Y, U_norm, V_norm,
        color="darkblue",
        scale=30,
        width=0.003,
        headwidth=3,
        headlength=5,
        pivot="mid"
    )

    # Mark all discovered fixed points
    for i, point in enumerate(fixed_points, start=1):
        ax.plot(
            point[0], point[1],
            marker="o",
            color="green",
            markersize=8,
            linestyle="None",
            label=f"Equilibrium {i}"
        )

        ax.annotate(
            f"E{i}",
            (point[0], point[1]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    # Eigenvector lines are valid globally for an affine-linear system.
    # For a nonlinear system, Jacobian eigenvectors are local directions
    # and should not be interpreted as global invariant lines.
    if is_linear and fixed_points:
        try:
            J = jacobian_at(fixed_points[0])
            eigvals, eigvecs = np.linalg.eig(J)
            eq = fixed_points[0]

            for i in range(2):
                v = eigvecs[:, i]

                if np.max(np.abs(np.imag(v))) < 1e-8:
                    v = np.real(v)

                    if np.linalg.norm(v) > 1e-10:
                        ax.axline(
                            eq,
                            (eq[0] + v[0], eq[1] + v[1]),
                            color="orange",
                            linestyle="--",
                            linewidth=1.8,
                            label=f"Eigenvector λ={eigvals[i].real:.3g}"
                        )
        except Exception:
            pass

    # -----------------------------------------------------
    # 8. FORWARD AND BACKWARD TRAJECTORIES
    # -----------------------------------------------------
    if show_trajectory:

        def ode_system(t, state):
            return vector_field(state[0], state[1])

        t_forward = np.linspace(0, 25, 2000)
        t_backward = np.linspace(0, -25, 2000)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                sol_f = solve_ivp(
                    ode_system,
                    (0, 25),
                    [x0, y0],
                    t_eval=t_forward,
                    method="RK45",
                    rtol=1e-7,
                    atol=1e-9
                )

                sol_b = solve_ivp(
                    ode_system,
                    (0, -25),
                    [x0, y0],
                    t_eval=t_backward,
                    method="RK45",
                    rtol=1e-7,
                    atol=1e-9
                )

            if sol_f.y.size:
                ax.plot(
                    sol_f.y[0], sol_f.y[1],
                    color="red",
                    linewidth=2,
                    label="Forward time"
                )

            if sol_b.y.size:
                ax.plot(
                    sol_b.y[0], sol_b.y[1],
                    color="magenta",
                    linewidth=2,
                    label="Backward time"
                )

            ax.plot(
                x0, y0, "ko",
                markersize=6,
                label="Starting point"
            )

            if not sol_f.success or not sol_b.success:
                st.warning(
                    "A trajectory integration stopped early, possibly "
                    "because of a singularity or numerical difficulty."
                )

        except Exception as e:
            st.warning(f"Trajectory calculation failed: {e}")

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_xlim(a - 0.1, b + 0.1)
    ax.set_ylim(c - 0.1, d + 0.1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Phase Portrait with Fixed Points")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Avoid duplicate legend entries
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best")

    st.pyplot(fig)

    # =====================================================
    # 9. DOWNLOAD PNG
    # =====================================================
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="Download Phase Portrait as PNG",
        data=buf.getvalue(),
        file_name="phase_portrait.png",
        mime="image/png"
    )

    plt.close(fig)


# =========================================================
# 8. MULTIPLE TRAJECTORIES AROUND ALL FIXED POINTS
# =========================================================

if show_trajectory:

    def ode_system(t, state):
        return vector_field(state[0], state[1])

    # Store all initial conditions
    initial_conditions = []

    # Generate multiple initial conditions around each equilibrium
    for i, eq in enumerate(fixed_points):

        angles = np.linspace(
            0,
            2 * np.pi,
            trajectories_per_equilibrium,
            endpoint=False
        )

        for angle in angles:
            dx = perturbation * np.cos(angle)
            dy = perturbation * np.sin(angle)

            initial_conditions.append({
                "point": np.array(eq) + np.array([dx, dy]),
                "equilibrium": i,
                "label": f"Near E{i + 1}"
            })

    # Add the user-defined initial condition, if requested
    if plot_user_trajectory:
        initial_conditions.append({
            "point": np.array([x0, y0], dtype=float),
            "equilibrium": None,
            "label": "User initial condition"
        })

    t_forward = np.linspace(0, trajectory_time, 1500)
    t_backward = np.linspace(0, -trajectory_time, 1500)

    successful_trajectories = 0

    for j, item in enumerate(initial_conditions):

        start = item["point"]
        eq_index = item["equilibrium"]

        # Use the same color for trajectories around the same equilibrium
        if eq_index is not None:
            color = f"C{eq_index % 10}"
            label = (
                f"Trajectories near E{eq_index + 1}"
                if j == next(
                    k for k, entry in enumerate(initial_conditions)
                    if entry["equilibrium"] == eq_index
                )
                else None
            )
        else:
            color = "black"
            label = "User initial condition trajectory"

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                sol_forward = solve_ivp(
                    ode_system,
                    (0, trajectory_time),
                    start,
                    t_eval=t_forward,
                    method="RK45",
                    rtol=1e-7,
                    atol=1e-9
                )

                sol_backward = solve_ivp(
                    ode_system,
                    (0, -trajectory_time),
                    start,
                    t_eval=t_backward,
                    method="RK45",
                    rtol=1e-7,
                    atol=1e-9
                )

            plotted = False

            if sol_forward.y.shape[1] > 0:
                ax.plot(
                    sol_forward.y[0],
                    sol_forward.y[1],
                    color=color,
                    linewidth=1.3,
                    alpha=0.8,
                    label=label
                )
                plotted = True

            if sol_backward.y.shape[1] > 0:
                ax.plot(
                    sol_backward.y[0],
                    sol_backward.y[1],
                    color=color,
                    linewidth=1.3,
                    alpha=0.8,
                    linestyle="--",
                    label=None
                )
                plotted = True

            if plotted:
                successful_trajectories += 1

            # Mark each perturbed initial point
            ax.plot(
                start[0],
                start[1],
                ".",
                color=color,
                markersize=4
            )

        except Exception:
            continue

    # Mark the original user-selected point
    if plot_user_trajectory:
        ax.plot(
            x0, y0,
            marker="*",
            color="black",
            markersize=12,
            label="User starting point"
        )

    st.write(
        f"Generated {len(initial_conditions)} initial conditions; "
        f"{successful_trajectories} produced plotted trajectories."
    )

    st.caption(
        "Solid lines represent forward time; dashed lines represent "
        "backward time. Colors distinguish the equilibria. "
        "Trajectories may overlap or leave the visible plotting region."
    )
