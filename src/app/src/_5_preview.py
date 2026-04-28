import os

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from src.utils.helper import make_config_from_session_state, debugger  # noqa
from trajgen.trajectory_generator import TrajectoryGenerator  # noqa
from src.method_overview import (
    ALL_SPATIAL_METHODS,
    ALL_TEMPORAL_METHODS,
    ALL_COMBINED_METHODS,
    ALL_RESAMPLING_METHODS,
)  # noqa
from trajgen.trajectory import Trajectory
from trajgen.config import Config  # noqa
from trajgen.point_generator import (  # noqa
    RandomPointGenerator2D,
    RandomPointGenerator3D,
    GridPointGenerator2D,
    GridPointGenerator3D,
    StartEndPointGenerator,
)


def show_preview_step():

    st.header("👁️ Step 5: Preview")
    st.write(
        "Review your configuration and preview generated trajectories. "
        "Use the slider to browse different trajectory IDs."
    )

    # --- Selection overview (read-only KPIs) ----------------------------------
    spatial_dim = st.session_state.get("config_spatial_dimension", "–")
    dim_type = st.session_state.get("config_spatial_dim_type", "–")
    spatial_method = st.session_state.get("selected_method", "–")
    temporal_method = st.session_state.get("selected_temporal_method", "None")
    resampling_method = st.session_state.get("selected_resampling_method", "None")

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Dimension", spatial_dim)
    kpi_cols[1].metric("Dim Type", dim_type)
    kpi_cols[2].metric("Spatial Method", spatial_method)
    kpi_cols[3].metric("Temporal Method", temporal_method)
    kpi_cols[4].metric("Resampling", resampling_method)

    st.divider()

    debugger()

    config = make_config_from_session_state()
    construct_trajectory_generator(config)

    st.slider(
        "Select Trajectory ID for Preview",
        min_value=0,
        max_value=100,
        key="current_id",
        help="Select a trajectory ID to preview how the configured parameters affect the generated trajectory.",
    )

    generate_trajectory(config)

    if st.session_state.get("selected_method") == "OSM Sampling":
        plot_osm_trajectory()
    else:
        plot_trajectory()

    # Navigation buttons
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ Back to Resampling Method Selection"):
            st.session_state.current_step = 4
            st.session_state.point_properties_locked = False
            st.rerun()

    with col3:
        if st.button("Next: Generation ➡️", type="primary"):
            st.session_state.current_step = 6
            st.rerun()


def construct_trajectory_generator(config: Config) -> Trajectory:
    selected_method = st.session_state.get("selected_method", None)

    # --- Resolve the spatial strategy class and its requirements ---------------
    if selected_method in ALL_SPATIAL_METHODS:
        strategy_cls = ALL_SPATIAL_METHODS[selected_method]
    elif selected_method in ALL_COMBINED_METHODS:
        strategy_cls = ALL_COMBINED_METHODS[selected_method]
    else:
        raise ValueError(
            f"Selected method '{selected_method}' not found in available methods"
        )

    spatial_dim = st.session_state.get("config_spatial_dimension", "2D")
    if hasattr(strategy_cls, "get_requirements"):
        try:
            requirements = strategy_cls.get_requirements(spatial_dim)
        except TypeError:
            requirements = strategy_cls.get_requirements()
    else:
        requirements = {}

    # --- Derive bounding-box ranges from requirements / session state ----------
    def _range_from_req(lo_key: str, hi_key: str, default_lo: float, default_hi: float):
        lo = float(st.session_state.get(f"config_{lo_key}", default_lo))
        hi = float(st.session_state.get(f"config_{hi_key}", default_hi))
        return (lo, hi)

    has_bbox = "get_next_x_min" in requirements
    if has_bbox:
        x_range = _range_from_req("get_next_x_min", "get_next_x_max", 0.0, 1.0)
        y_range = _range_from_req("get_next_y_min", "get_next_y_max", 0.0, 1.0)
    else:
        x_range = (config.x_min, config.x_max)
        y_range = (config.y_min, config.y_max)

    # --- Build the core PointGenerator ----------------------------------------
    spatial_dim_type = st.session_state.get("config_spatial_dim_type", "continuous")
    seed = config.seed

    if spatial_dim == "3D":
        if "get_next_z_min" in requirements:
            z_range = _range_from_req(
                "get_next_z_min",
                "get_next_z_max",
                0.0,
                1.0,
            )
        else:
            z_range = (config.z_min, config.z_max)
        if spatial_dim_type == "discrete":
            grid_res = float(st.session_state.get("config_grid_resolution", 1.0))
            point_gen = GridPointGenerator3D(x_range, y_range, z_range, grid_res)
        else:
            point_gen = RandomPointGenerator3D(x_range, y_range, z_range, seed)
    else:  # 2D
        if spatial_dim_type == "discrete":
            grid_res = float(st.session_state.get("config_grid_resolution", 1.0))
            point_gen = GridPointGenerator2D(x_range, y_range, grid_res)
        else:
            point_gen = RandomPointGenerator2D(x_range, y_range, seed)

    # --- Optionally wrap with StartEndPointGenerator --------------------------
    from shapely.geometry import Point as ShapelyPoint

    is_3d = spatial_dim == "3D"
    start_point = None
    end_point = None

    if "get_start_point" in requirements:
        mode = st.session_state.get("config_get_start_point_mode", "undefined")
        if mode == "fixed for dataset":
            x = float(st.session_state.get("config_get_start_point_x", 0.0))
            y = float(st.session_state.get("config_get_start_point_y", 0.0))
            if is_3d:
                z = float(st.session_state.get("config_get_start_point_z", 0.0))
                start_point = ShapelyPoint(x, y, z)
            else:
                start_point = ShapelyPoint(x, y)

    if "get_end_point" in requirements:
        mode = st.session_state.get("config_get_end_point_mode", "undefined")
        if mode == "fixed for dataset":
            x = float(st.session_state.get("config_get_end_point_x", 0.0))
            y = float(st.session_state.get("config_get_end_point_y", 0.0))
            if is_3d:
                z = float(st.session_state.get("config_get_end_point_z", 0.0))
                end_point = ShapelyPoint(x, y, z)
            else:
                end_point = ShapelyPoint(x, y)

    if start_point is not None or end_point is not None:
        point_gen = StartEndPointGenerator(point_gen, start=start_point, end=end_point)

    config.point_generator = point_gen

    # --- Instantiate strategies -----------------------------------------------
    if selected_method in ALL_SPATIAL_METHODS:
        if selected_method == "OSM Sampling":
            _bar = st.progress(0, "Downloading OSM road network…")
            _status = st.empty()

            def _osm_progress(pct: int, msg: str) -> None:
                _bar.progress(pct, msg)
                _status.caption(msg)

            from trajgen.spatial_strategy.osm_sampling import OsmSamplingStrategy
            spatial_strategy = OsmSamplingStrategy(config, progress_callback=_osm_progress)
            _bar.empty()
            _status.empty()
        else:
            spatial_strategy = ALL_SPATIAL_METHODS[selected_method](config)
        temporal_strategy = (
            ALL_TEMPORAL_METHODS[
                st.session_state.get("selected_temporal_method", None)
            ](config)
            if st.session_state.get("selected_temporal_method") in ALL_TEMPORAL_METHODS
            and st.session_state.get("selected_temporal_method") != "None"
            else None
        )
    elif selected_method in ALL_COMBINED_METHODS:
        spatial_strategy = ALL_COMBINED_METHODS[selected_method](config)
        temporal_strategy = (
            None  # Temporal strategy is determined by the combined method
        )
    else:
        raise ValueError("Selected method not found in available methods")

    resampling_strategy = (
        ALL_RESAMPLING_METHODS[
            st.session_state.get("selected_resampling_method", None)
        ](config)
        if st.session_state.get("selected_resampling_method") in ALL_RESAMPLING_METHODS
        and st.session_state.get("selected_resampling_method") != "None"
        else None
    )

    generator = TrajectoryGenerator(
        config,
        spatial_strategy,
        temporal_strategy=temporal_strategy,
        resampling_strategy=resampling_strategy,
    )
    st.session_state["current_generator"] = generator


def generate_and_plot_trajectory(config: Config) -> None:
    generate_trajectory(config)
    plot_trajectory()


def generate_trajectory(config: Config) -> None:
    import random as _random

    id = st.session_state["current_id"]
    if os.getenv("DEBUG", "False").lower() == "true":
        print(f"Generating trajectory with ID {id} using current configuration...")
    generator = st.session_state["current_generator"]

    # Derive a per-trajectory seed so each ID produces a unique trajectory
    # while remaining reproducible for the same (base_seed, id) pair.
    base_seed = config.seed
    traj_seed = base_seed + id
    config._rng = _random.Random(traj_seed)

    # Reseed the point generator's RNG, traversing through any wrapper
    # (e.g. StartEndPointGenerator) to reach the inner generator.
    pg = config._state.get("config_point_generator")
    while pg is not None:
        if hasattr(pg, "rng"):
            pg.rng = _random.Random(traj_seed)
            break
        pg = getattr(pg, "inner", None)

    trajectory = generator.generate_trajectory(id)

    st.session_state["current_trajectory"] = trajectory


def plot_osm_trajectory() -> None:
    """Show an OSM trajectory on an interactive Leaflet map with PT hotspots."""
    import folium
    from streamlit_folium import st_folium

    trajectory = st.session_state["current_trajectory"]
    generator = st.session_state["current_generator"]
    strategy = generator.spatial_strategy

    # Trajectory coords are (lon, lat) — Folium needs [lat, lon]
    coords = list(trajectory.ls.coords)
    traj_locations = [[c[1], c[0]] for c in coords]

    # Centre map on the trajectory
    center_lat = sum(c[0] for c in traj_locations) / len(traj_locations)
    center_lon = sum(c[1] for c in traj_locations) / len(traj_locations)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # PT hotspot markers — red, small
    G = strategy._G
    for node_id in strategy._hotspots:
        nd = G.nodes[node_id]
        folium.CircleMarker(
            location=[nd["y"], nd["x"]],
            radius=5,
            color="red",
            fill=True,
            fill_opacity=0.7,
            tooltip=f"PT hotspot {node_id}",
        ).add_to(m)

    # Trajectory polyline — blue
    folium.PolyLine(
        locations=traj_locations,
        color="blue",
        weight=4,
        opacity=0.85,
        tooltip=f"Trajectory {trajectory.id}",
    ).add_to(m)

    # Start / end markers
    folium.Marker(
        location=traj_locations[0],
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        location=traj_locations[-1],
        tooltip="End",
        icon=folium.Icon(color="darkred", icon="stop"),
    ).add_to(m)

    columns = st.columns(3)
    with columns[1]:
        st_folium(m, height=500, use_container_width=True)
        with st.expander("Show Trajectory Details"):
            st.write(f"Trajectory ID: {trajectory.id}")
            st.write(f"Number of points: {len(coords)}")
            st.write(f"PT hotspots in area: {len(strategy._hotspots)}")
            df_points = pd.DataFrame(coords, columns=["lon", "lat"])
            if trajectory.t is not None:
                df_points["time"] = trajectory.t
            st.dataframe(df_points, width="stretch")


def plot_trajectory(title: str | None = None) -> None:
    """
    Plot a Trajectory object in Streamlit using Matplotlib.

    Converts LineString to x,y arrays and plots path with points.
    Supports time `t` as colored markers.
    Uses a 3D axes when the configured spatial dimension is 3D.
    """
    trajectory = st.session_state["current_trajectory"]
    coords = list(trajectory.ls.coords)
    dim = st.session_state.get("config_spatial_dimension", "2D")
    is_3d = dim == "3D"

    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]
    z_coords = [c[2] for c in coords] if is_3d and len(coords[0]) >= 3 else None

    fig: Figure = plt.figure(figsize=(10, 8))

    if is_3d and z_coords is not None:
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            x_coords,
            y_coords,
            z_coords,
            "b-",
            linewidth=2,
            label=f"Trajectory {trajectory.id}",
        )
        ax.scatter(x_coords, y_coords, z_coords, c="red", s=40, label="Points")

        if trajectory.t is not None:
            sc = ax.scatter(
                x_coords,
                y_coords,
                z_coords,
                c=trajectory.t,
                cmap="viridis",
                s=80,
                edgecolors="black",
                linewidth=0.5,
                label="Time points",
            )
            fig.colorbar(sc, ax=ax, label="Time (s)", shrink=0.6)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        ax = fig.add_subplot(111)

        ax.plot(
            x_coords, y_coords, "b-", linewidth=3, label=f"Trajectory {trajectory.id}"
        )
        ax.plot(x_coords, y_coords, "ro", markersize=8, label="Points")

        if trajectory.t is not None:
            sc = ax.scatter(
                x_coords,
                y_coords,
                c=trajectory.t,
                cmap="viridis",
                s=100,
                edgecolors="black",
                linewidth=1,
                label="Time points",
            )
            fig.colorbar(sc, label="Time (s)")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    ax.set_title(title or f"Trajectory ID {trajectory.id}")
    ax.legend()

    columns = st.columns(3)
    with columns[1]:
        st.pyplot(fig)
        with st.expander("Show Trajectory Details"):
            st.write(f"Trajectory ID: {trajectory.id}")
            st.write(f"Number of points: {len(coords)}")
            if trajectory.t is not None:
                st.write(
                    f"Time range: {min(trajectory.t):.2f} to {max(trajectory.t):.2f} seconds"
                )
            cols = ["x", "y"] + (["z"] if is_3d else [])
            df_points = pd.DataFrame(coords, columns=cols)
            if trajectory.t is not None:
                df_points["time"] = trajectory.t
            st.dataframe(df_points, width="stretch")

    plt.close(fig)  # Prevent memory leak


def get_method_specific_parameters():
    selected_point_generator = st.session_state.get("selected_point_generator", None)
    selected_method = st.session_state.get("selected_method", None)
    selected_temporal_method = st.session_state.get("selected_temporal_method", None)
    selected_resampling_method = st.session_state.get(
        "selected_resampling_method", None
    )

    st.subheader("Selected Methods")
    columns = st.columns(4)
    with columns[0]:
        st.metric("- Point Sampling:", selected_point_generator)
    with columns[1]:
        st.metric("- Spatial/Combined Method:", selected_method)
    with columns[2]:
        st.metric("- Temporal Method:", selected_temporal_method)
    with columns[3]:
        st.metric("- Resampling Method:", selected_resampling_method)
