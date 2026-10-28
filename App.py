        def ode_system(t, state):
            return vector_field(state[0], state[1])

        initial_conditions = []

        # Generate several nearby initial conditions for each equilibrium.
        for eq_index, eq in enumerate(fixed_points):
            angles = np.linspace(
                0, 2 * np.pi,
                trajectories_per_equilibrium,
                endpoint=False
            )

            for angle_index, angle in enumerate(angles):
                offset = perturbation * np.array([
                    np.cos(angle), np.sin(angle)
                ])
                initial_conditions.append({
                    "point": np.asarray(eq, dtype=float) + offset,
                    "equilibrium": eq_index,
                    "label": (
                        f"Trajectories near E{eq_index + 1}"
                        if angle_index == 0 else None
                    )
                })

        if plot_user_trajectory:
            initial_conditions.append({
                "point": np.array([x0, y0], dtype=float),
                "equilibrium": None,
                "label": "User starting point"
            })

        t_forward = np.linspace(0, trajectory_time, 1500)
        t_backward = np.linspace(0, -trajectory_time, 1500)
        successful_trajectories = 0

        for item in initial_conditions:
            start_point = item["point"]
            eq_index = item["equilibrium"]
            color = "black" if eq_index is None else f"C{eq_index % 10}"

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sol_forward = solve_ivp(
                        ode_system, (0, trajectory_time), start_point,
                        t_eval=t_forward, method="RK45",
                        rtol=1e-7, atol=1e-9
                    )
                    sol_backward = solve_ivp(
                        ode_system, (0, -trajectory_time), start_point,
                        t_eval=t_backward, method="RK45",
                        rtol=1e-7, atol=1e-9
                    )

                plotted = False

                if sol_forward.y.shape[1] > 0:
                    ax.plot(
                        sol_forward.y[0], sol_forward.y[1],
                        color=color, linewidth=1.3, alpha=0.8,
                        label=item["label"]
                    )
                    item["label"] = None
                    plotted = True

                if sol_backward.y.shape[1] > 0:
                    ax.plot(
                        sol_backward.y[0], sol_backward.y[1],
                        color=color, linewidth=1.3, alpha=0.8,
                        linestyle="--", label=None
                    )
                    plotted = True

                if plotted:
                    successful_trajectories += 1

                ax.plot(
                    start_point[0], start_point[1],
                    ".", color=color, markersize=4
                )

            except Exception:
                continue

        if plot_user_trajectory:
            ax.plot(
                x0, y0, marker="*", color="black",
                markersize=12, linestyle="None",
                label="User starting point"
            )

        st.write(
            f"Generated {len(initial_conditions)} initial conditions; "
            f"{successful_trajectories} produced plotted trajectories."
        )
        st.caption(
            "Solid lines show forward time; dashed lines show backward time. "
            "Colors distinguish trajectories initialized near different "
            "equilibria. Numerical integration may stop early near singularities."
        )

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
