import streamlit as st
import os
import json
from .utils.helper import show_available_methods_preview


def _reset_discrete_locations(points_count_key: str) -> None:
    """
    Reset all categorical discrete locations.
    Manually clears session state for all location coordinates.
    """
    num_locations = st.session_state.get(points_count_key, 0)
    for i in range(num_locations):
        x_key = f"discrete_location_{i}_x"
        y_key = f"discrete_location_{i}_y"
        z_key = f"discrete_location_{i}_z"

        if x_key in st.session_state:
            del st.session_state[x_key]
        if y_key in st.session_state:
            del st.session_state[y_key]
        if z_key in st.session_state:
            del st.session_state[z_key]

    st.session_state[points_count_key] = 1


def show_point_properties_step():
    st.header("📍 Step 1: Point Properties")
    st.write("Configure the basic properties of your trajectory points.")

    st.write("**Spatial Dimension Properties**")
    col1, col2, col3 = st.columns(3)

    with col1:
        # Dimension selector - manual state management
        current_dimension = st.session_state.get("config_spatial_dimension", "2D")
        dimension = st.selectbox(
            "Spatial Dimension",
            options=["2D", "3D"],
            index=0 if current_dimension == "2D" else 1,
            help="Select the spatial dimensionality of the trajectories",
            disabled=st.session_state.get("point_properties_locked", False),
        )
        # Save to session state immediately
        st.session_state["config_spatial_dimension"] = dimension

        # Spatial dimension type - manual state management
        current_spatial_type = st.session_state.get(
            "config_spatial_dim_type", "continuous"
        )
        spatial_dim_type = st.selectbox(
            "Spatial Dimension Type",
            options=["continuous", "discrete"],
            index=0 if current_spatial_type == "continuous" else 1,
            help="Select whether spatial coordinates are continuous or discrete (grid-based)",
            disabled=st.session_state.get("point_properties_locked", False),
        )
        # Save to session state immediately
        st.session_state["config_spatial_dim_type"] = spatial_dim_type

    with col2:
        # Give more specifics about discrete spatial dimension type
        if spatial_dim_type == "discrete":

            current_discrete_type = st.session_state.get(
                "config_discrete_dim_type", "grid-based"
            )
            discrete_dim_type = st.selectbox(
                "Discrete Spatial Dimension Type",
                options=["grid-based", "categorical"],
                index=0 if current_discrete_type == "grid-based" else 1,
                help="Select the type of discrete spatial dimension. Grid-based means points will be on a regular grid,"
                " while categorical means points will be selected from a set of predefined locations.",
                disabled=st.session_state.get("point_properties_locked", False),
            )
            st.session_state["config_discrete_dim_type"] = discrete_dim_type
            with col3:
                if discrete_dim_type == "grid-based":
                    current_resolution = st.session_state.get(
                        "config_grid_resolution", 10.0
                    )
                    grid_resolution = st.number_input(
                        "Grid Resolution",
                        value=current_resolution,
                        help="Select the resolution of the grid for grid-based discrete spatial dimension. "
                        "This defines the distance between adjacent points on the grid.",
                        disabled=st.session_state.get("point_properties_locked", False),
                    )
                    st.session_state["config_grid_resolution"] = grid_resolution
                elif discrete_dim_type == "categorical":
                    st.write(
                        "Categorical discrete spatial dimension means points will be selected from a predefined set of "
                        "locations."
                        " This is useful for scenarios like selecting from a set of cities or landmarks."
                    )

                    # Initialize session state
                    points_count_key = "discrete_locations_count"
                    if points_count_key not in st.session_state:
                        st.session_state[points_count_key] = 1

                    st.write(
                        "Define discrete locations using one of the methods below:"
                    )

                    # Create tabs for manual input vs CSV upload
                    tab1, tab2 = st.tabs(["Manual Input", "Upload CSV"])

                    with tab1:
                        num_locations = st.number_input(
                            "Number of Locations",
                            value=st.session_state.get(points_count_key, 1),
                            min_value=1,
                            max_value=50,
                            step=1,
                            help="Number of discrete locations to define",
                            disabled=st.session_state.point_properties_locked,
                        )
                        # Manually update session state
                        st.session_state[points_count_key] = int(num_locations)

                        st.write(f"Define {num_locations} discrete locations:")

                        # Create input fields for each location
                        for i in range(num_locations):
                            st.write(f"**Location {i+1}**")

                            if dimension == "2D":
                                col_loc = st.columns(2)
                            else:  # 3D
                                col_loc = st.columns(3)

                            x_key = f"discrete_location_{i}_x"
                            y_key = f"discrete_location_{i}_y"
                            z_key = f"discrete_location_{i}_z"

                            # Initialize session state
                            if x_key not in st.session_state:
                                st.session_state[x_key] = 0.0
                            if y_key not in st.session_state:
                                st.session_state[y_key] = 0.0
                            if z_key not in st.session_state:
                                st.session_state[z_key] = 0.0

                            with col_loc[0]:
                                x_val = st.number_input(
                                    f"Loc{i+1} X",
                                    value=st.session_state.get(x_key, 0.0),
                                    help=f"X coordinate of location {i+1}",
                                    disabled=st.session_state.point_properties_locked,
                                )
                                st.session_state[x_key] = x_val

                            with col_loc[1]:
                                y_val = st.number_input(
                                    f"Loc{i+1} Y",
                                    value=st.session_state.get(y_key, 0.0),
                                    help=f"Y coordinate of location {i+1}",
                                    disabled=st.session_state.point_properties_locked,
                                )
                                st.session_state[y_key] = y_val

                            if dimension == "3D":
                                with col_loc[2]:
                                    z_val = st.number_input(
                                        f"Loc{i+1} Z",
                                        value=st.session_state.get(z_key, 0.0),
                                        help=f"Z coordinate of location {i+1}",
                                        disabled=st.session_state.point_properties_locked,
                                    )
                                    st.session_state[z_key] = z_val

                    with tab2:
                        st.write("Upload a CSV file with one location per line.")
                        if dimension == "2D":
                            st.write("Expected format: `x,y` (one location per row)")
                        else:
                            st.write("Expected format: `x,y,z` (one location per row)")

                        uploaded_file = st.file_uploader(
                            "Choose CSV file",
                            type="csv",
                            key="discrete_locations_csv_upload",
                            disabled=st.session_state.point_properties_locked,
                        )

                        if uploaded_file is not None:
                            try:
                                import csv
                                import io

                                # Read CSV file
                                stream = io.StringIO(
                                    uploaded_file.getvalue().decode("utf8"),
                                    newline=None,
                                )
                                csv_reader = csv.reader(stream)
                                points = []

                                for row_idx, row in enumerate(csv_reader):
                                    if not row or all(
                                        not cell.strip() for cell in row
                                    ):  # Skip empty rows
                                        continue

                                    try:
                                        if dimension == "2D":
                                            if len(row) < 2:
                                                st.warning(
                                                    f"Row {row_idx + 1}: Expected at least 2 values "
                                                    f"(x, y), got {len(row)}"
                                                )
                                                continue
                                            x, y = float(row[0].strip()), float(
                                                row[1].strip()
                                            )
                                            points.append((x, y))
                                        else:  # 3D
                                            if len(row) < 3:
                                                st.warning(
                                                    f"Row {row_idx + 1}: Expected at least 3 values "
                                                    f"(x, y, z), got {len(row)}"
                                                )
                                                continue
                                            x, y, z = (
                                                float(row[0].strip()),
                                                float(row[1].strip()),
                                                float(row[2].strip()),
                                            )
                                            points.append((x, y, z))
                                    except ValueError as e:
                                        st.warning(
                                            f"Row {row_idx + 1}: Could not parse values as numbers - {e}"
                                        )
                                        continue

                                if points:
                                    st.session_state[points_count_key] = len(points)

                                    # Store points in session state with proper keys
                                    for i, point in enumerate(points):
                                        x_key = f"discrete_location_{i}_x"
                                        y_key = f"discrete_location_{i}_y"
                                        z_key = f"discrete_location_{i}_z"

                                        st.session_state[x_key] = point[0]
                                        st.session_state[y_key] = point[1]
                                        if dimension == "3D":
                                            st.session_state[z_key] = point[2]

                                    st.success(
                                        f"✓ Loaded {len(points)} locations from CSV"
                                    )
                                else:
                                    st.error("No valid locations found in CSV")
                            except Exception as e:
                                st.error(f"Error reading CSV file: {e}")

                    # Debug section showing all inputted locations
                    with st.expander("📊 Debug: Discrete Locations", expanded=False):
                        num_locations = st.session_state.get(points_count_key, 0)
                        if num_locations > 0:
                            st.write(f"Total locations: {num_locations}")
                            locations_display = []

                            for i in range(num_locations):
                                x_key = f"discrete_location_{i}_x"
                                y_key = f"discrete_location_{i}_y"
                                z_key = f"discrete_location_{i}_z"

                                x = st.session_state.get(x_key, None)
                                y = st.session_state.get(y_key, None)
                                z = (
                                    st.session_state.get(z_key, None)
                                    if dimension == "3D"
                                    else None
                                )

                                if dimension == "2D":
                                    locations_display.append(
                                        {"Location": i + 1, "X": x, "Y": y}
                                    )
                                else:
                                    locations_display.append(
                                        {"Location": i + 1, "X": x, "Y": y, "Z": z}
                                    )

                            st.dataframe(locations_display, width="stretch")
                        else:
                            st.info("No locations defined yet")

                    # Reset button
                    st.button(
                        "🔄 Reset Locations",
                        on_click=_reset_discrete_locations,
                        args=(points_count_key,),
                        help="Clear all discrete locations",
                        disabled=st.session_state.point_properties_locked,
                    )
    # Temporal dimension type - manual state management
    # col1,col2,col3 = st.columns(3)
    with col1:
        st.write("**Temporal Dimension Properties**")
        current_temporal = st.session_state.get(
            "config_temporal_dim_type", "continuous"
        )
        temporal_dim_type = st.selectbox(
            "Temporal Dimension Type",
            options=["continuous", "discrete"],
            index=0 if current_temporal == "continuous" else 1,
            help="Select whether temporal coordinates are continuous or discrete",
            disabled=st.session_state.get("point_properties_locked", False),
        )
        # Save to session state immediately
        st.session_state["config_temporal_dim_type"] = temporal_dim_type

    show_available_methods_preview()

    st.subheader("📂 Load Configuration")
    st.write("Upload a previously downloaded configuration file to restore settings.")

    uploaded_config = st.file_uploader(
        "Upload Configuration JSON", type="json", max_upload_size=5
    )  # Limit to 5MB

    if uploaded_config is not None:
        try:

            config_data = json.load(uploaded_config)

            # Button to apply
            if st.button("Apply Configuration & Skip to Preview", type="primary"):
                # Update session state with loaded config
                for key, value in config_data.items():
                    # Basic validation or filtering if needed
                    st.session_state[key] = value

                st.session_state.point_properties_locked = True
                st.session_state.current_step = 5
                st.rerun()
            st.success("Configuration file loaded. Click above to apply and proceed.")

        except json.JSONDecodeError:
            st.error("Invalid JSON file.")
        except Exception as e:
            st.error(f"Error loading configuration: {e}")

    st.write("**Or choose from existing configurations:**")

    scenarios_path = os.path.join(
        os.path.dirname(__file__), "..", "assets", "scenarios"
    )

    if os.path.exists(scenarios_path):
        json_files = sorted([f for f in os.listdir(scenarios_path) if f.endswith(".json")])
        if json_files:
            selected_config = st.selectbox(
                "Choose Existing Configuration",
                options=json_files,
                index=None,
                key="existing_config_select",
            )
            if selected_config:
                config_path = os.path.join(scenarios_path, selected_config)
                try:
                    with open(config_path, "r") as f:
                        config_data = json.load(f)
                    if st.button(
                        "Apply Selected Configuration & Skip to Preview",
                        type="primary",
                        key="apply_selected_config",
                    ):
                        for key, value in config_data.items():
                            st.session_state[key] = value
                        st.session_state.point_properties_locked = True
                        st.session_state.current_step = 5
                        st.rerun()
                    st.success(
                        "Configuration file loaded. Click above to apply and proceed."
                    )
                except json.JSONDecodeError:
                    st.error("Invalid JSON file.")
                except Exception as e:
                    st.error(f"Error loading configuration: {e}")
        else:
            st.info("No existing configurations found.")
    else:
        st.info("Scenarios folder not found.")

    # Navigation buttons
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])

    with col3:
        if st.button(
            "Next: Select Method ➡️",
            type="primary",
        ):
            st.session_state.current_step = 2
            st.session_state.point_properties_locked = True
            st.rerun()
