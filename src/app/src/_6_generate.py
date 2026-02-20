import streamlit as st
import random
import json
import io
import csv
import os
from datetime import datetime
from timeit import default_timer as timer
from src.utils.helper import make_config_from_session_state  # noqa
from src._5_preview import construct_trajectory_generator  # noqa


def show_generation_step():
    st.header("🚀 Step 6: Generate Trajectories")

    num_trajectories_to_generate = st.number_input(
        "Number of trajectories to generate",
        min_value=1,
        step=1,
        value=st.session_state.get("config_num_trajectories", 100),
    )

    st.session_state["config_num_trajectories"] = num_trajectories_to_generate
    st.write(
        f"Ready to generate **{num_trajectories_to_generate}** trajectories "
        "with the selected configuration and methods."
    )

    columns = st.columns(2)
    with columns[0]:
        st.button("🎉 Generate Trajectories", on_click=generate_trajectories)
    with columns[1]:
        st.button("⏱️ Generate Trajectories Timed", on_click=generate_trajectories_timed)

    if "generated_trajectories" in st.session_state:
        st.info(
            f"{len(st.session_state['generated_trajectories'])} trajectories "
            "available."
        )

    # Download buttons rendered inline so Streamlit can serve the files
    dl_cols = st.columns(2)
    with dl_cols[0]:
        download_configuration()
    with dl_cols[1]:
        download_trajectories()

    # Navigation buttons
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ Back to Configuration Step"):
            st.session_state.current_step = 5
            st.session_state.point_properties_locked = False
            st.rerun()


def generate_trajectories_timed():
    # Ensure logs directory exists
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    times = []
    for i in range(20):
        start = timer()
        generate_trajectories()
        end = timer()
        elapsed = end - start
        times.append(elapsed)

    # Build a per-run log line with datetime and times
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines = [f"{now_str} | run {i+1:02d} | {t:.6f} s" for i, t in enumerate(times)]

    log_path = os.path.join(logs_dir, f"{now_str}_logs.txt")
    # Append so you keep a history
    with open(log_path, "a", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    avg_time = sum(times) / len(times)
    st.success(
        f"Trajectory generation completed. "
        f"Runs: {len(times)}, avg: {avg_time:.4f}s, "
        f"min: {min(times):.4f}s, max: {max(times):.4f}s"
    )
    # Save config as JSON
    config_dict = generate_config_dict()
    config_dict["generation_times"] = times

    config_path = os.path.join(logs_dir, f"{now_str}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, default=str, ensure_ascii=True)


def generate_trajectories():
    num = st.session_state.get("config_num_trajectories", 100)
    config = make_config_from_session_state()

    construct_trajectory_generator(config)
    generator = st.session_state.get("current_generator")
    if generator is None:
        st.error("No generator configured. Please complete the previous steps first.")
        return

    base_seed = config.seed
    trajectories = []
    progress = st.progress(0, text="Generating trajectories...")
    for i in range(num):
        traj_seed = base_seed + i
        config._rng = random.Random(traj_seed)
        if hasattr(config, "point_generator") and hasattr(
            config.point_generator, "rng"
        ):
            config.point_generator.rng = random.Random(traj_seed)
        traj = generator.generate_trajectory(i)
        trajectories.append(traj)
        progress.progress((i + 1) / num, text=f"Generated {i + 1}/{num} trajectories")

    st.session_state["generated_trajectories"] = trajectories
    progress.empty()
    st.success(f"Successfully generated {len(trajectories)} trajectories!")


def generate_config_dict():
    # Collect all config_* keys from session state into a JSON-serialisable dict
    config_dict = {}
    for key, value in st.session_state.items():
        if key.startswith("config_") or key.startswith("selected_"):
            try:
                json.dumps(value)  # test serialisability
                config_dict[key] = value
            except (TypeError, ValueError):
                config_dict[key] = str(value)

    return config_dict


def download_configuration():

    config_dict = generate_config_dict()
    json_str = json.dumps(config_dict, indent=2)
    st.download_button(
        label="📥 Download Configuration (JSON)",
        data=json_str,
        file_name="trajectory_config.json",
        mime="application/json",
    )


def download_trajectories():
    trajectories = st.session_state.get("generated_trajectories")
    if not trajectories:
        st.warning(
            "No trajectories generated yet. Click 'Generate Trajectories' first."
        )
        return

    buf = io.StringIO()
    writer = csv.writer(buf)

    # Detect dimensionality and time from the first trajectory
    sample_coords = list(trajectories[0].ls.coords)
    has_z = len(sample_coords[0]) >= 3
    has_time = trajectories[0].t is not None

    header = ["traj_id", "point_idx", "x", "y"]
    if has_z:
        header.append("z")
    if has_time:
        header.append("time")
    writer.writerow(header)

    for traj in trajectories:
        coords = list(traj.ls.coords)
        for idx, coord in enumerate(coords):
            row = [traj.id, idx, coord[0], coord[1]]
            if has_z:
                row.append(coord[2] if len(coord) >= 3 else "")
            if has_time and traj.t is not None:
                row.append(traj.t[idx] if idx < len(traj.t) else "")
            writer.writerow(row)

    st.download_button(
        label="📥 Download Trajectories (CSV)",
        data=buf.getvalue(),
        file_name="trajectories.csv",
        mime="text/csv",
    )
