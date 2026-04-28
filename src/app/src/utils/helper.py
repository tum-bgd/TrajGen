import os

from ..method_overview import (
    ALL_COMBINED_METHODS,
    ALL_RESAMPLING_METHODS,
    ALL_SPATIAL_METHODS,
    ALL_TEMPORAL_METHODS,
)
import streamlit as st
from trajgen.config import Config
from trajgen.relevant_properties import Properties


DEBUG = os.getenv("DEBUG", "False").lower() == "true"


def debugger():
    if DEBUG:
        with st.expander("🔍 Debug Info", expanded=False):
            st.write("DEBUG - Session state keys:", st.session_state)


def make_config_from_session_state():
    # Collect all config_ keys from session state
    state = {}
    for key, value in st.session_state.items():
        if isinstance(key, str) and key.startswith("config_"):
            state[key] = value

    # Also include strategy-selection keys (not config_ prefixed)
    for key in [
        "selected_method",
        "selected_temporal_method",
        "selected_resampling_method",
    ]:
        if key in st.session_state:
            state[key] = st.session_state[key]

    seed = int(state.get("config_seed", 42))
    return Config(state, seed=seed)


def get_available_spatial_methods(dimension: str, spatial_dim_type: str) -> list[str]:
    """
    Determine which trajectory generation methods are available based on point properties.
    This is a placeholder implementation that needs to be refined based on actual method constraints.
    """
    all_methods = list(ALL_SPATIAL_METHODS.keys())

    # Placeholder logic - needs refinement
    available = set(all_methods)
    if dimension == "2D":
        available = available
    elif dimension == "3D":
        available = available.intersection(
            ("Random Walk", "Physics Informed", "Freespace")
        )

    # Some methods might only work with certain configurations
    if spatial_dim_type == "discrete":
        available = available.intersection(("Random Walk", "Equal Distribution"))
    elif spatial_dim_type == "continuous":
        available = available.intersection(
            ("Random Walk", "Physics Informed", "Freespace", "OSM Sampling")
        )

    return list(available)  # Remove duplicates


def get_available_temporal_methods(temporal_dim_type: str) -> list[str]:
    all_temporal_methods = list(ALL_TEMPORAL_METHODS.keys())

    if temporal_dim_type == "continuous":
        return all_temporal_methods
    elif temporal_dim_type == "discrete":
        raise NotImplementedError(
            "Temporal methods for discrete temporal dimension type are not implemented yet."
        )


def get_available_combined_methods(
    dimension: str, spatial_dim_type: str, temporal_dim_type: str
) -> list[str]:
    all_combined_methods = list(ALL_COMBINED_METHODS.keys())
    if (
        dimension == "2D"
        or dimension == "3D"
        and spatial_dim_type == "continuous"
        and temporal_dim_type == "continuous"
    ):
        return all_combined_methods
    else:
        return []  # No combined methods available for other configurations


def get_available_resampling_methods(temporal_dim_type: str) -> list[str]:
    all_resampling_methods = list(ALL_RESAMPLING_METHODS.keys())

    if temporal_dim_type == "continuous":
        return all_resampling_methods
    elif temporal_dim_type == "discrete":
        raise NotImplementedError(
            "Resampling methods for discrete temporal dimension type are not implemented yet."
        )


def show_available_methods_preview():
    with st.expander("🔍 Available Methods Preview", expanded=False):
        # Show available methods preview
        dimension = st.session_state["config_spatial_dimension"]
        spatial_dim_type = st.session_state["config_spatial_dim_type"]
        temporal_dim_type = st.session_state["config_temporal_dim_type"]
        available_spatial_methods = get_available_spatial_methods(
            dimension, spatial_dim_type
        )

        available_temporal_methods = get_available_temporal_methods(temporal_dim_type)

        available_combined_methods = get_available_combined_methods(
            dimension, spatial_dim_type, temporal_dim_type
        )

        st.subheader("Available Spatial Methods Preview")
        if available_spatial_methods:
            st.success(
                f"**{len(available_spatial_methods)}** methods available for this configuration:"
            )
            method_cols = st.columns(min(5, len(available_spatial_methods)))
            for i, method in enumerate(available_spatial_methods):
                with method_cols[i % 5]:
                    st.info(f"{method}")
        else:
            st.error(
                "⚠️ No methods available for this combination. Please adjust your selection."
            )

        st.subheader("Available Temporal Methods Preview")
        if available_temporal_methods:
            st.success(
                f"**{len(available_temporal_methods)}** methods available for this configuration:"
            )
            method_cols = st.columns(min(5, len(available_temporal_methods)))
            for i, method in enumerate(available_temporal_methods):
                with method_cols[i % 5]:
                    st.info(f"{method}")
        else:
            st.error(
                "⚠️ No temporal methods available for this combination. Please adjust your selection."
            )

        st.subheader("Available Combined Spatial-Temporal Methods Preview")
        if available_combined_methods:
            st.success(
                f"**{len(available_combined_methods)}** combined methods available for this configuration:"
            )
            method_cols = st.columns(min(5, len(available_combined_methods)))
            for i, method in enumerate(available_combined_methods):
                with method_cols[i % 5]:
                    st.info(f"{method}")


def universal_user_input_method(input_name: str, input_properties: dict):
    columns = st.columns(3)
    config_mode_name = "config_" + input_name + "_mode"
    config_distribution_name = "config_" + input_name + "_distribution"
    config_value_name = "config_" + input_name

    # Use default_mode / default from the requirement dict when available
    _default_mode = input_properties.get("default_mode", "undefined")
    _default_value = input_properties.get("default", None)

    # Initialize session state keys to ensure they persist.
    # Also upgrade stale "undefined" / None when the requirement supplies a
    # concrete default (covers existing sessions started before a default was
    # added).
    # Skip for get_point_function – it uses its own "_point_mode" key scheme.
    if input_properties["type"] != "get_point_function":
        if config_mode_name not in st.session_state:
            st.session_state[config_mode_name] = _default_mode
        elif (
            st.session_state[config_mode_name] == "undefined"
            and _default_mode != "undefined"
        ):
            st.session_state[config_mode_name] = _default_mode

        if config_distribution_name not in st.session_state:
            st.session_state[config_distribution_name] = None

        if config_value_name not in st.session_state:
            st.session_state[config_value_name] = _default_value
        elif st.session_state[config_value_name] is None and _default_value is not None:
            st.session_state[config_value_name] = _default_value

    # Get attributes of the property to get user input for
    # Get int function
    if input_properties["type"] == "get_int_function":
        # Fix level of setting
        if getattr(Properties, config_mode_name, None) is None:
            mode_properties = {
                "short_name": input_properties["short_name"] + " Mode",
                "description": f"Select the mode for {input_properties['short_name']}",
                "options": {
                    "undefined": [],
                    "fixed for dataset": [],
                    "fixed for trajectory": [],
                },
            }
        else:
            mode_properties = getattr(Properties, config_mode_name, [])

        with columns[0]:
            get_fixation_level(config_mode_name, mode_properties)

        # Either fix value or distribution
        if st.session_state[config_mode_name] == "fixed for dataset":
            with columns[1]:
                value_properties = {
                    "short_name": input_properties["short_name"],
                    "description": f"Set the value for {input_properties['short_name']}",
                    "type": "int",
                }
                get_int_value(config_value_name, value_properties)

        elif st.session_state[config_mode_name] == "fixed for trajectory":
            if getattr(Properties, config_distribution_name, None) is None:
                distribution_properties = {
                    "short_name": input_properties["short_name"] + " Distribution",
                    "description": f"Select the distribution for {input_properties['short_name']}",
                    "options": {
                        "uniform": ["min", "max"],
                        "normal": ["mean", "std"],
                    },
                }
            else:
                distribution_properties = getattr(Properties, config_distribution_name)

            with columns[1]:
                get_distribution(config_distribution_name, distribution_properties)

            # Fix values
            with columns[2]:
                parameters_to_get = distribution_properties["options"].get(
                    st.session_state.get(config_distribution_name, None), []
                )
                for param in parameters_to_get:
                    param_config_name = "config_" + input_name + "_" + param
                    # Initialize parameter session state
                    if param_config_name not in st.session_state:
                        st.session_state[param_config_name] = None

                    parameter_properties = getattr(Properties, param_config_name, None)
                    if parameter_properties is None:
                        # Create default properties if not found
                        parameter_properties = {
                            "short_name": param.capitalize(),
                            "description": f"Set the {param} for {input_properties['short_name']}",
                            "type": (
                                "float"
                                if param in ["mean", "std", "min", "max"]
                                else "int"
                            ),
                        }

                    if parameter_properties.get("type") == "int":
                        get_int_value(param_config_name, parameter_properties)
                    elif parameter_properties.get("type") == "float":
                        get_float_value(param_config_name, parameter_properties)

    elif input_properties["type"] == "get_float_function":
        # Fix level of setting
        if getattr(Properties, config_mode_name, None) is None:
            mode_properties = {
                "short_name": input_properties["short_name"] + " Mode",
                "description": f"Select the mode for {input_properties['short_name']}",
                "options": {
                    "undefined": [],
                    "fixed for dataset": [],
                    "fixed for trajectory": [],
                },
            }
        else:
            mode_properties = getattr(Properties, config_mode_name, [])

        with columns[0]:
            get_fixation_level(config_mode_name, mode_properties)

        # Either fix value or distribution
        if st.session_state[config_mode_name] == "fixed for dataset":
            with columns[1]:
                value_properties = {
                    "short_name": input_properties["short_name"],
                    "description": f"Set the value for {input_properties['short_name']}",
                    "type": "float",
                }
                get_float_value(config_value_name, value_properties)

        elif st.session_state[config_mode_name] == "fixed for trajectory":
            if getattr(Properties, config_distribution_name, None) is None:
                distribution_properties = {
                    "short_name": input_properties["short_name"] + " Distribution",
                    "description": f"Select the distribution for {input_properties['short_name']}",
                    "options": {
                        "uniform": ["min", "max"],
                        "normal": ["mean", "std"],
                    },
                }
            else:
                distribution_properties = getattr(Properties, config_distribution_name)

            with columns[1]:
                get_distribution(config_distribution_name, distribution_properties)

            # Fix values
            with columns[2]:
                parameters_to_get = distribution_properties["options"].get(
                    st.session_state.get(config_distribution_name, None), []
                )
                for param in parameters_to_get:
                    param_config_name = "config_" + input_name + "_" + param
                    # Initialize parameter session state
                    if param_config_name not in st.session_state:
                        st.session_state[param_config_name] = None

                    parameter_properties = getattr(Properties, param_config_name, None)
                    if parameter_properties is None:
                        # Create default properties if not found
                        parameter_properties = {
                            "short_name": param.capitalize(),
                            "description": f"Set the {param} for {input_properties['short_name']}",
                            "type": (
                                "float"
                                if param in ["mean", "std", "min", "max"]
                                else "int"
                            ),
                        }

                    if parameter_properties.get("type") == "int":
                        get_int_value(param_config_name, parameter_properties)
                    elif parameter_properties.get("type") == "float":
                        get_float_value(param_config_name, parameter_properties)

    elif input_properties["type"] == "get_point_function":
        # Strip trailing "_point" suffix if already present to avoid double-suffix
        base_name = input_name[:-6] if input_name.endswith("_point") else input_name
        point_config_name = "config_" + base_name + "_point"

        # Initialize session state keys
        point_mode_key = point_config_name + "_mode"
        point_distribution_key = point_config_name + "_distribution"

        if point_mode_key not in st.session_state:
            st.session_state[point_mode_key] = "undefined"
        if point_distribution_key not in st.session_state:
            st.session_state[point_distribution_key] = None

        # Get spatial dimension and type
        spatial_dimension = st.session_state.get("config_spatial_dimension", "2D")
        spatial_dim_type = st.session_state.get("config_spatial_dim_type", "continuous")

        # COLUMN 0: Fixation Level Selection
        with columns[0]:
            st.session_state[point_mode_key] = st.selectbox(
                f"{input_properties['short_name']} Mode",
                options=["undefined", "fixed for dataset", "fixed for trajectory"],
                index=(
                    ["undefined", "fixed for dataset", "fixed for trajectory"].index(
                        st.session_state.get(point_mode_key, "undefined")
                    )
                    if st.session_state.get(point_mode_key, "undefined")
                    in ["undefined", "fixed for dataset", "fixed for trajectory"]
                    else 0
                ),
                key=point_mode_key + "_element",
                help="Select whether the point is fixed for the dataset or varies per trajectory",
            )

        # COLUMN 1 & 2: Based on mode
        if st.session_state[point_mode_key] == "fixed for dataset":
            # Fixed point - user inputs specific coordinates
            get_fixed_point_input(
                point_config_name,
                input_properties,
                spatial_dimension,
                spatial_dim_type,
                columns[1],
                columns[2],
            )

        elif st.session_state[point_mode_key] == "fixed for trajectory":
            # Variable point - user defines distribution
            with columns[1]:
                # Select distribution type
                if spatial_dim_type == "continuous":
                    dist_options = ["uniform", "normal"]
                else:  # discrete
                    dist_options = ["uniform", "normal", "discrete set"]

                st.session_state[point_distribution_key] = st.selectbox(
                    f"{input_properties['short_name']} Distribution",
                    options=dist_options,
                    index=(
                        dist_options.index(
                            st.session_state.get(
                                point_distribution_key, dist_options[0]
                            )
                        )
                        if st.session_state.get(point_distribution_key) in dist_options
                        else 0
                    ),
                    key=point_distribution_key + "_element",
                    help="Select the distribution for generating points",
                )

            # Column 2: Distribution parameters
            with columns[2]:
                distribution = st.session_state.get(point_distribution_key)
                if distribution == "uniform":
                    get_point_uniform_distribution(
                        point_config_name, spatial_dimension, spatial_dim_type
                    )
                elif distribution == "normal":
                    get_point_normal_distribution(point_config_name, spatial_dimension)
                elif distribution == "discrete set":
                    get_point_discrete_set(point_config_name, spatial_dimension)

    # Get bool function
    elif input_properties["type"] == "get_bool_function":
        # Initialize session state
        if config_mode_name not in st.session_state:
            st.session_state[config_mode_name] = "fixed for dataset"
        if config_value_name not in st.session_state:
            st.session_state[config_value_name] = False

        config_probability_name = "config_" + input_name + "_probability"
        if config_probability_name not in st.session_state:
            st.session_state[config_probability_name] = 0.5

        with columns[0]:
            st.session_state[config_mode_name] = st.selectbox(
                f"{input_properties['short_name']} Mode",
                options=[
                    "fixed for dataset",
                    "fixed for trajectory",
                ],
                index=(
                    0
                    if st.session_state.get(config_mode_name, "fixed for dataset")
                    == "fixed for dataset"
                    else (
                        1
                        if st.session_state.get(config_mode_name, "fixed for dataset")
                        == "fixed for trajectory"
                        else 0
                    )
                ),
                help=f"Toggle {input_properties['short_name']} on or off",
            )

        # Show appropriate input in column 1 based on mode
        if st.session_state[config_mode_name] == "fixed for dataset":
            with columns[1]:
                st.session_state[config_value_name] = st.selectbox(
                    f"{input_properties['short_name']} Value",
                    options=[True, False],
                    index=(0 if st.session_state.get(config_value_name, False) else 1),
                    help=f"Set the fixed boolean value for {input_properties['short_name']}",
                )
        elif st.session_state[config_mode_name] == "fixed for trajectory":
            with columns[1]:
                st.session_state[config_probability_name] = st.number_input(
                    f"{input_properties['short_name']} Probability",
                    value=st.session_state.get(config_probability_name, 0.5),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    help=f"Set the probability (0.0-1.0) of True for {input_properties['short_name']}",
                )

    # Get str function
    elif input_properties["type"] == "get_str_function":
        # Initialize session state assuming we want a simple selection fixed for the dataset
        if config_mode_name not in st.session_state:
            st.session_state[config_mode_name] = "fixed for dataset"

        # Get options from properties
        options = input_properties.get("options", [])
        default_value = input_properties.get("default", options[0] if options else "")

        if config_value_name not in st.session_state:
            st.session_state[config_value_name] = default_value

        with columns[0]:
            # Currently we only support fixed for dataset for strings for simplicity
            # But we keep the UI consistent
            st.session_state[config_mode_name] = st.selectbox(
                f"{input_properties['short_name']} Mode",
                options=["fixed for dataset"],
                index=0,
                disabled=True,  # Lock to fixed for dataset for now
                help=f"Mode for {input_properties['short_name']} (currently only fixed supported)",
            )

        # Show appropriate input in column 1
        with columns[1]:
            current_val = st.session_state.get(config_value_name, default_value)
            idx = options.index(current_val) if current_val in options else 0

            st.session_state[config_value_name] = st.selectbox(
                f"{input_properties['short_name']}",
                options=options,
                index=idx,
                help=f"Select the value for {input_properties['short_name']}",
            )


def get_fixation_level(config_mode_name: str, properties: dict):

    st.session_state[config_mode_name] = st.selectbox(
        f"{properties['short_name']}",
        options=list(properties["options"].keys()),
        index=(
            list(properties["options"].keys()).index(
                st.session_state.get(config_mode_name, "undefined")
            )
            if st.session_state.get(config_mode_name, "undefined")
            in properties["options"]
            else 0
        ),
        help=f"Select the mode for {properties['short_name']}",
    )


def get_distribution(config_distribution_name: str, distribution_properties: dict):
    st.session_state[config_distribution_name] = st.selectbox(
        f"{distribution_properties['short_name']}",
        options=distribution_properties["options"].keys(),
        index=(
            list(distribution_properties["options"].keys()).index(
                st.session_state.get(config_distribution_name, "undefined")
            )
            if st.session_state.get(config_distribution_name, "undefined")
            in distribution_properties["options"]
            else 0
        ),
        help=f"Select the distribution for {distribution_properties['short_name']}",
    )


def get_int_value(config_name: str, properties: dict):
    st.session_state[config_name] = st.number_input(
        label=properties["short_name"],
        value=st.session_state.get(config_name, 0),
        help=f"Set the value for {properties['short_name']}",
    )


def get_float_value(config_name: str, properties: dict):
    st.session_state[config_name] = st.number_input(
        label=properties["short_name"],
        value=st.session_state.get(config_name, 0.0),
        help=f"Set the value for {properties['short_name']}",
    )


def get_fixed_point_input(
    point_config_name: str,
    input_properties: dict,
    spatial_dimension: str,
    spatial_dim_type: str,
    col1,
    col2,
):
    """
    Get fixed point coordinates for dataset.
    User inputs x, y (and z for 3D) coordinates.
    For discrete grids, validates that coordinates are multiples of grid resolution.
    """
    print(point_config_name)
    point_x_key = point_config_name + "_x"
    point_y_key = point_config_name + "_y"
    point_z_key = point_config_name + "_z"
    grid_resolution = st.session_state.get("config_grid_resolution", 1.0)

    # Initialize session state
    if point_x_key not in st.session_state:
        st.session_state[point_x_key] = 0.0
    if point_y_key not in st.session_state:
        st.session_state[point_y_key] = 0.0
    if point_z_key not in st.session_state:
        st.session_state[point_z_key] = 0.0

    with col1:
        st.session_state[point_x_key] = st.number_input(
            "X Coordinate",
            value=st.session_state.get(point_x_key, 0.0),
            help="X coordinate of the fixed point",
            key=point_x_key + "_element",
        )
        st.session_state[point_y_key] = st.number_input(
            "Y Coordinate",
            value=st.session_state.get(point_y_key, 0.0),
            help="Y coordinate of the fixed point",
            key=point_y_key + "_element",
        )

        if spatial_dimension == "3D":
            st.session_state[point_z_key] = st.number_input(
                "Z Coordinate",
                value=st.session_state.get(point_z_key, 0.0),
                help="Z coordinate of the fixed point",
                key=point_z_key + "_element",
            )

    # Validation for discrete dimensions
    with col2:
        if spatial_dim_type == "discrete":
            discrete_type = st.session_state.get(
                "config_discrete_dim_type", "grid-based"
            )
            x_val = st.session_state.get(point_x_key, 0.0)
            y_val = st.session_state.get(point_y_key, 0.0)
            z_val = (
                st.session_state.get(point_z_key, 0.0)
                if spatial_dimension == "3D"
                else 0.0
            )

            if discrete_type == "grid-based":
                # Validate grid alignment
                x_valid = (
                    (x_val % grid_resolution) == 0 if grid_resolution != 0 else True
                )
                y_valid = (
                    (y_val % grid_resolution) == 0 if grid_resolution != 0 else True
                )
                z_valid = (
                    (z_val % grid_resolution) == 0 if grid_resolution != 0 else True
                )

                if x_valid and y_valid and z_valid:
                    st.success(f"✓ Valid grid point (resolution: {grid_resolution})")
                else:
                    st.error(
                        f"✗ Coordinates must be multiples of grid resolution {grid_resolution}"
                    )

            elif discrete_type == "categorical":
                # Validate against set of allowed discrete points
                # Collect all defined discrete points
                allowed_points = []
                points_count_key = point_config_name + "_discrete_count"
                num_points = st.session_state.get(points_count_key, 0)

                for i in range(num_points):
                    pt_x_key = f"{point_config_name}_discrete_point_{i}_x"
                    pt_y_key = f"{point_config_name}_discrete_point_{i}_y"
                    pt_z_key = f"{point_config_name}_discrete_point_{i}_z"

                    pt_x = st.session_state.get(pt_x_key, None)
                    pt_y = st.session_state.get(pt_y_key, None)
                    pt_z = (
                        st.session_state.get(pt_z_key, None)
                        if spatial_dimension == "3D"
                        else None
                    )

                    if pt_x is not None and pt_y is not None:
                        if spatial_dimension == "2D":
                            allowed_points.append((pt_x, pt_y))
                        else:
                            allowed_points.append((pt_x, pt_y, pt_z))

                # Check if current point is in allowed set
                if spatial_dimension == "2D":
                    current_point = (x_val, y_val)
                else:
                    current_point = (x_val, y_val, z_val)

                # Use approximate comparison for floats
                point_found = any(
                    all(
                        abs(current_point[j] - allowed[j]) < 1e-6
                        for j in range(len(allowed))
                    )
                    for allowed in allowed_points
                )

                if point_found:
                    st.success("✓ Valid discrete point (found in set)")
                else:
                    if allowed_points:
                        st.error(
                            f"✗ Point not in allowed discrete set. Allowed points: {allowed_points}"
                        )
                    else:
                        st.warning("⚠️ No discrete points defined yet")
        else:
            st.info("Continuous space - any coordinates allowed")


def get_point_uniform_distribution(
    point_config_name: str, spatial_dimension: str, spatial_dim_type: str
):
    """
    Get uniform distribution parameters (bounding box) for point generation.
    """
    bbox_x_min_key = point_config_name + "_uniform_x_min"
    bbox_x_max_key = point_config_name + "_uniform_x_max"
    bbox_y_min_key = point_config_name + "_uniform_y_min"
    bbox_y_max_key = point_config_name + "_uniform_y_max"
    bbox_z_min_key = point_config_name + "_uniform_z_min"
    bbox_z_max_key = point_config_name + "_uniform_z_max"

    # Initialize session state
    for key in [
        bbox_x_min_key,
        bbox_x_max_key,
        bbox_y_min_key,
        bbox_y_max_key,
        bbox_z_min_key,
        bbox_z_max_key,
    ]:
        if key not in st.session_state:
            if "min" in key:
                st.session_state[key] = 0.0
            else:
                st.session_state[key] = 1.0

    st.subheader("Bounding Box Parameters")

    col_x = st.columns(2)
    with col_x[0]:
        st.session_state[bbox_x_min_key] = st.number_input(
            "X Min",
            value=st.session_state.get(bbox_x_min_key, 0.0),
            help="Minimum X coordinate",
            key=bbox_x_min_key + "_element",
        )
    with col_x[1]:
        st.session_state[bbox_x_max_key] = st.number_input(
            "X Max",
            value=st.session_state.get(bbox_x_max_key, 1.0),
            help="Maximum X coordinate",
            key=bbox_x_max_key + "_element",
        )

    col_y = st.columns(2)
    with col_y[0]:
        st.session_state[bbox_y_min_key] = st.number_input(
            "Y Min",
            value=st.session_state.get(bbox_y_min_key, 0.0),
            help="Minimum Y coordinate",
            key=bbox_y_min_key + "_element",
        )
    with col_y[1]:
        st.session_state[bbox_y_max_key] = st.number_input(
            "Y Max",
            value=st.session_state.get(bbox_y_max_key, 1.0),
            help="Maximum Y coordinate",
            key=bbox_y_max_key + "_element",
        )

    if spatial_dimension == "3D":
        col_z = st.columns(2)
        with col_z[0]:
            st.session_state[bbox_z_min_key] = st.number_input(
                "Z Min",
                value=st.session_state.get(bbox_z_min_key, 0.0),
                help="Minimum Z coordinate",
                key=bbox_z_min_key + "_element",
            )
        with col_z[1]:
            st.session_state[bbox_z_max_key] = st.number_input(
                "Z Max",
                value=st.session_state.get(bbox_z_max_key, 1.0),
                help="Maximum Z coordinate",
                key=bbox_z_max_key + "_element",
            )


def get_point_normal_distribution(point_config_name: str, spatial_dimension: str):
    """
    Get normal distribution parameters (mean and std) for point generation in all dimensions.
    """
    mean_x_key = point_config_name + "_normal_mean_x"
    std_x_key = point_config_name + "_normal_std_x"
    mean_y_key = point_config_name + "_normal_mean_y"
    std_y_key = point_config_name + "_normal_std_y"
    mean_z_key = point_config_name + "_normal_mean_z"
    std_z_key = point_config_name + "_normal_std_z"

    # Initialize session state
    for key in [mean_x_key, mean_y_key, mean_z_key]:
        if key not in st.session_state:
            st.session_state[key] = 0.5
    for key in [std_x_key, std_y_key, std_z_key]:
        if key not in st.session_state:
            st.session_state[key] = 0.1

    st.subheader("Normal Distribution Parameters")

    st.write("**X Dimension**")
    col_x = st.columns(2)
    with col_x[0]:
        st.session_state[mean_x_key] = st.number_input(
            "X Mean",
            value=st.session_state.get(mean_x_key, 0.5),
            help="Mean of X distribution",
            key=mean_x_key + "_element",
        )
    with col_x[1]:
        st.session_state[std_x_key] = st.number_input(
            "X Std Dev",
            value=st.session_state.get(std_x_key, 0.1),
            min_value=0.01,
            help="Standard deviation of X distribution",
            key=std_x_key + "_element",
        )

    st.write("**Y Dimension**")
    col_y = st.columns(2)
    with col_y[0]:
        st.session_state[mean_y_key] = st.number_input(
            "Y Mean",
            value=st.session_state.get(mean_y_key, 0.5),
            help="Mean of Y distribution",
            key=mean_y_key + "_element",
        )
    with col_y[1]:
        st.session_state[std_y_key] = st.number_input(
            "Y Std Dev",
            value=st.session_state.get(std_y_key, 0.1),
            min_value=0.01,
            help="Standard deviation of Y distribution",
            key=std_y_key + "_element",
        )

    if spatial_dimension == "3D":
        st.write("**Z Dimension**")
        col_z = st.columns(2)
        with col_z[0]:
            st.session_state[mean_z_key] = st.number_input(
                "Z Mean",
                value=st.session_state.get(mean_z_key, 0.5),
                help="Mean of Z distribution",
                key=mean_z_key + "_element",
            )
        with col_z[1]:
            st.session_state[std_z_key] = st.number_input(
                "Z Std Dev",
                value=st.session_state.get(std_z_key, 0.1),
                min_value=0.01,
                help="Standard deviation of Z distribution",
                key=std_z_key + "_element",
            )


def _reset_discrete_points(points_count_key: str, points_key: str) -> None:
    """
    Reset all discrete points for a specific input.
    Manually clears session state for all point coordinates.
    """
    num_points = st.session_state.get(points_count_key, 0)
    for i in range(num_points):
        pt_key = f"{points_key}_point_{i}"
        x_key = f"{pt_key}_x"
        y_key = f"{pt_key}_y"
        z_key = f"{pt_key}_z"

        if x_key in st.session_state:
            del st.session_state[x_key]
        if y_key in st.session_state:
            del st.session_state[y_key]
        if z_key in st.session_state:
            del st.session_state[z_key]

    st.session_state[points_count_key] = 1


def get_point_discrete_set(point_config_name: str, spatial_dimension: str):
    """
    Get discrete set of points for point generation.
    User can define multiple points to choose from using manual input or CSV upload.
    """
    points_key = point_config_name + "_discrete_points"
    points_count_key = point_config_name + "_discrete_count"

    # Initialize session state
    if points_count_key not in st.session_state:
        st.session_state[points_count_key] = 1
    if points_key not in st.session_state:
        st.session_state[points_key] = {}

    st.subheader("Discrete Point Set")

    # Create tabs for manual input vs CSV upload
    tab1, tab2 = st.tabs(["Manual Input", "Upload CSV"])

    with tab1:
        # Manual point input
        num_points = st.number_input(
            "Number of Points",
            value=st.session_state.get(points_count_key, 1),
            min_value=1,
            step=1,
            help="How many discrete points to define",
        )
        # Manually update session state
        st.session_state[points_count_key] = int(num_points)

        # Create input fields for each point
        for i in range(num_points):
            pt_key = f"{points_key}_point_{i}"
            st.write(f"**Point {i+1}**")

            col_coords = st.columns(3 if spatial_dimension == "3D" else 2)

            x_key = f"{pt_key}_x"
            y_key = f"{pt_key}_y"
            z_key = f"{pt_key}_z"

            if x_key not in st.session_state:
                st.session_state[x_key] = 0.0
            if y_key not in st.session_state:
                st.session_state[y_key] = 0.0
            if z_key not in st.session_state:
                st.session_state[z_key] = 0.0

            with col_coords[0]:
                x_val = st.number_input(
                    f"P{i+1} X",
                    value=st.session_state.get(x_key, 0.0),
                    help=f"X coordinate of point {i+1}",
                )
                st.session_state[x_key] = x_val

            with col_coords[1]:
                y_val = st.number_input(
                    f"P{i+1} Y",
                    value=st.session_state.get(y_key, 0.0),
                    help=f"Y coordinate of point {i+1}",
                )
                st.session_state[y_key] = y_val

            if spatial_dimension == "3D":
                with col_coords[2]:
                    z_val = st.number_input(
                        f"P{i+1} Z",
                        value=st.session_state.get(z_key, 0.0),
                        help=f"Z coordinate of point {i+1}",
                    )
                    st.session_state[z_key] = z_val

    with tab2:
        # CSV upload option
        st.write("Upload a CSV file with one point per line.")
        if spatial_dimension == "2D":
            st.write("Expected format: `x,y` (one point per row)")
        else:
            st.write("Expected format: `x,y,z` (one point per row)")

        uploaded_file = st.file_uploader(
            "Choose CSV file", type="csv", key=f"{point_config_name}_csv_upload"
        )

        if uploaded_file is not None:
            try:
                import csv
                import io

                # Read CSV file
                stream = io.StringIO(
                    uploaded_file.getvalue().decode("utf8"), newline=None
                )
                csv_reader = csv.reader(stream)
                points = []

                for row_idx, row in enumerate(csv_reader):
                    if not row or all(
                        not cell.strip() for cell in row
                    ):  # Skip empty rows
                        continue

                    try:
                        if spatial_dimension == "2D":
                            if len(row) < 2:
                                st.warning(
                                    f"Row {row_idx + 1}: Expected at least 2 values (x, y), got {len(row)}"
                                )
                                continue
                            x, y = float(row[0].strip()), float(row[1].strip())
                            points.append((x, y))
                        else:  # 3D
                            if len(row) < 3:
                                st.warning(
                                    f"Row {row_idx + 1}: Expected at least 3 values (x, y, z), got {len(row)}"
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
                        pt_key = f"{points_key}_point_{i}"
                        x_key = f"{pt_key}_x"
                        y_key = f"{pt_key}_y"
                        z_key = f"{pt_key}_z"

                        st.session_state[x_key] = point[0]
                        st.session_state[y_key] = point[1]
                        if spatial_dimension == "3D":
                            st.session_state[z_key] = point[2]

                    st.success(f"✓ Loaded {len(points)} points from CSV")
                else:
                    st.error("No valid points found in CSV")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

    # Debug section showing all inputted points
    with st.expander("📊 Debug: Input Points", expanded=False):
        num_points = st.session_state.get(points_count_key, 0)
        if num_points > 0:
            st.write(f"Total points: {num_points}")
            points_display = []

            for i in range(num_points):
                pt_key = f"{points_key}_point_{i}"
                x_key = f"{pt_key}_x"
                y_key = f"{pt_key}_y"
                z_key = f"{pt_key}_z"

                x = st.session_state.get(x_key, None)
                y = st.session_state.get(y_key, None)
                z = (
                    st.session_state.get(z_key, None)
                    if spatial_dimension == "3D"
                    else None
                )

                if spatial_dimension == "2D":
                    points_display.append({"Point": i + 1, "X": x, "Y": y})
                else:
                    points_display.append({"Point": i + 1, "X": x, "Y": y, "Z": z})

            st.dataframe(points_display, width="stretch")
        else:
            st.info("No points defined yet")

    # Reset button
    st.button(
        "🔄 Reset Points",
        on_click=_reset_discrete_points,
        args=(points_count_key, points_key),
        help="Clear all points for this specific input",
    )
