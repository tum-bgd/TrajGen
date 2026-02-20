import streamlit as st
from shapely.geometry import Point
from src.configuration import show_configuration_step

ALL_SPATIAL_METHODS = [
    "Random Walk",
    # "Constrained Random Walk",
    "Equal Distribution",
    # "Polynomial Curves",
    "Physics Informed",
    "Freespace",
    "OSM Sampling",
]

ALL_TEMPORAL_METHODS = [
    "None",
    "Constant Time",
    "Constant Velocity",
    "Variable Velocity",
    "Constant Acceleration",
    "Variable Acceleration",
    "Constant Time Step",
    "Variable Time Step",
]

ALL_COMBINED_METHODS = ["Physics Informed"]

ALL_RESAMPLING_METHODS = [
    "None",
    "Constant Length",
    "Constant Temporal Step",
    "Constant Spatial Step",
]

ALL_METHOD_DESCRIPTIONS = {
    "Random Walk": "Generates trajectories through random movement steps from point to point.",
    "Constrained Random Walk": "Random walk with constraints to prevent going outside defined boundaries.",
    "Equal Distribution": "Distributes points equally across a discrete grid, ensuring balanced spatial coverage.",
    "Polynomial Curves": "Creates smooth trajectories using polynomial mathematical functions.",
    "Freespace": "Generates trajectories in open/free space without obstacles.",
    "OSM Sampling": "Samples trajectories from real-world OpenStreetMap road networks.",
    "Physics Informed": "Uses physics principles (velocity, acceleration) to create realistic movement patterns.",
}


def main():
    st.set_page_config(page_title="TrajGen", page_icon="📈", layout="wide")

    st.title("📈TrajGen Trajectory Generator")
    st.write(
        "Generate synthetic trajectory data using various spatial and temporal strategies."
    )

    # Initialize session state for step navigation
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1
    if "point_properties_locked" not in st.session_state:
        st.session_state.point_properties_locked = False

    # Progress indicator
    progress_steps = [
        "Point Properties",
        "Spatial Method",
        "Temporal Method",
        "Resampling Method",
        "Configuration",
        "Generation",
    ]
    current_step = st.session_state.current_step

    # Create progress bar
    progress_cols = st.columns(len(progress_steps))
    for i, step_name in enumerate(progress_steps):
        with progress_cols[i]:
            if i + 1 == current_step:
                st.markdown(f"**🔄 {i+1}. {step_name}**")
            elif i + 1 < current_step:
                st.markdown(f"✅ {i+1}. {step_name}")
            else:
                st.markdown(f"⭕ {i+1}. {step_name}")

    st.divider()

    # Step 1: Point Property Selector
    if current_step == 1:
        show_point_properties_step()

    # Step 2: Method Selection
    elif current_step == 2:
        show_spatial_method_selection_step()

    # Step 3: Temporal Method Selection
    elif current_step == 3:
        show_temporal_method_selection_step()

    # Step 4: Resampling Method Selection
    elif current_step == 4:
        show_resampling_method_selection_step()

    # Step 5: Configuration
    elif current_step == 5:
        show_configuration_step()

    # Step 6: Generation
    elif current_step == 6:
        show_generation_step()


def show_point_properties_step():
    st.header("📍 Step 1: Point Properties")
    st.write("Configure the basic properties of your trajectory points.")

    # Get current values from session state or set defaults
    current_dimension = st.session_state.get("config_spatial_dimension", "2D")
    current_spatial_type = st.session_state.get("config_spatial_dim_type", "continuous")
    current_temporal_type = st.session_state.get("config_temporal_dim_type", "continuous")

    col1, col2 = st.columns(2)

    with col1:
        # Dimension selector
        dimension = st.selectbox(
            "Spatial Dimension",
            options=["2D", "3D"],
            index=0 if current_dimension == "2D" else 1,
            help="Select the spatial dimensionality of the trajectories",
            disabled=st.session_state.point_properties_locked,
        )

        # Temporal dimension type
        temporal_dim_type = st.selectbox(
            "Temporal Dimension Type",
            options=["continuous", "discrete"],
            index=0 if current_temporal_type == "continuous" else 1,
            help="Select whether temporal coordinates are continuous or discrete",
            disabled=st.session_state.point_properties_locked,
        )

    with col2:
        # Spatial dimension type
        spatial_dim_type = st.selectbox(
            "Spatial Dimension Type",
            options=["continuous", "discrete"],
            index=0 if current_spatial_type == "continuous" else 1,
            help="Select whether spatial coordinates are continuous or discrete (grid-based)",
            disabled=st.session_state.point_properties_locked,
        )

    # Store selections in session state
    st.session_state["config_spatial_dimension"] = dimension
    st.session_state["config_spatial_dim_type"] = spatial_dim_type
    st.session_state["config_temporal_dim_type"] = temporal_dim_type

    # Give more specifics about discrete spatial dimension type
    if spatial_dim_type == "discrete":
        st.write("Discrete Spatial Dimension Details")
        col1, col2 = st.columns(2)
        with col1:
            discrete_dim_type = st.selectbox(
                "Discrete Spatial Dimension Type",
                options=["grid-based", "categorical"],
                help="Select the type of discrete spatial dimension. Grid-based means points will be on a regular grid,"
                " while categorical means points will be selected from a set of predefined locations.",
                disabled=st.session_state.point_properties_locked,
            )
        with col2:
            if discrete_dim_type == "grid-based":
                grid_resolution = st.slider(
                    "Grid Resolution",
                    min_value=0.01,
                    max_value=100.0,
                    step=0.01,
                    value=10.0,
                    help="Select the resolution of the grid for grid-based discrete spatial dimension. This defines the"
                    " distance between adjacent points on the grid.",
                    disabled=st.session_state.point_properties_locked,
                )
                st.session_state["grid_resolution"] = grid_resolution
            elif discrete_dim_type == "categorical":
                st.write(
                    "Categorical discrete spatial dimension means points will be selected from a predefined set of "
                    "locations. This is useful for scenarios like selecting from a set of cities or landmarks."
                )
                num_locations = st.number_input(
                    "Number of Locations",
                    min_value=2,
                    max_value=50,
                    value=5,
                    help="Number of discrete locations to define",
                    disabled=st.session_state.point_properties_locked,
                )

                # Initialize locations in session state if not present
                if "discrete_locations" not in st.session_state:
                    st.session_state.discrete_locations = []

                st.write(f"Define {num_locations} discrete locations:")

                # TODO  - Add functionality to remove/edit existing locations
                # Create input fields for each location
                for i in range(num_locations):
                    if dimension == "2D":
                        col21, col22 = st.columns(2)
                    elif dimension == "3D":
                        col21, col22, col23 = st.columns(3)

                    # Get existing values or defaults
                    existing_location = (
                        st.session_state.discrete_locations[i]
                        if i < len(st.session_state.discrete_locations)
                        else {}
                    )

                    with col21:
                        x = st.number_input(
                            f"Location {i+1} - X",
                            value=getattr(existing_location, "x", 0.0),
                            disabled=st.session_state.point_properties_locked,
                        )
                    with col22:
                        y = st.number_input(
                            f"Location {i+1} - Y",
                            value=getattr(existing_location, "y", 0.0),
                            disabled=st.session_state.point_properties_locked,
                        )
                    if dimension == "3D":
                        with col23:
                            z = st.number_input(
                                f"Location {i+1} - Z",
                                value=getattr(existing_location, "z", 0.0),
                                disabled=st.session_state.point_properties_locked,
                            )
                    # Update session state with new location
                    st.session_state.discrete_locations.append(
                        Point(x, y, z) if dimension == "3D" else Point(x, y)
                    )

            st.session_state["discrete_dim_type"] = discrete_dim_type
    
    show_available_methods_preview()
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


def show_available_methods_preview():
    with st.expander("🔍 Available Methods Preview", expanded=False):
        # Show available methods preview
        dimension = st.session_state["config_spatial_dimension"]
        spatial_dim_type = st.session_state.get("config_spatial_dim_type")
        temporal_dim_type = st.session_state.get("config_temporal_dim_type")
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


def show_spatial_method_selection_step():
    st.header("📐 Step 2: Spatial Method Selection")

    # Show locked point properties summary
    st.subheader("🔒 Point Properties (Locked)")
    locked_cols = st.columns(3)
    with locked_cols[0]:
        st.metric("Dimension", st.session_state.get("config_spatial_dimension", "2D"))
    with locked_cols[1]:
        st.metric("Spatial Type", st.session_state.get("config_spatial_dim_type", "continuous"))
    with locked_cols[2]:
        st.metric("Temporal Type", st.session_state.get("config_temporal_dim_type", "continuous"))

    # Method selection
    available_spatial_methods = get_available_spatial_methods(
        st.session_state.get("config_spatial_dimension", "2D"),
        st.session_state.get("config_spatial_dim_type", "continuous"),
    )

    available_combined_methods = get_available_combined_methods(
        st.session_state.get("config_spatial_dimension", "2D"),
        st.session_state.get("config_spatial_dim_type", "continuous"),
        st.session_state.get("config_temporal_dim_type", "continuous"),
    )
    st.subheader("🎯 Select Generation Method")

    # Get current selection from session state
    current_method = st.session_state.get(
        "selected_method",
        available_spatial_methods[0] if available_spatial_methods else None,
    )

    # Method selection with descriptions

    if available_spatial_methods and available_combined_methods:
        st.info(
            "Both spatial and combined spatial-temporal methods are available. Combined methods may consider temporal"
            " properties in their generation process."
        )
        available_methods = available_spatial_methods + available_combined_methods
    elif available_spatial_methods:
        available_methods = available_spatial_methods
    elif available_combined_methods:
        available_methods = available_combined_methods
    else:
        available_methods = []

    if available_methods:
        selected_spatial_method = st.radio(
            "Choose a spatial or combined spatial-temporal generation method:",
            available_methods,
            index=(
                available_spatial_methods.index(current_method)
                if current_method in available_spatial_methods
                else 0
            ),
            help="Select the algorithm that will generate your trajectories",
        )

        # Show method description
        if selected_spatial_method in ALL_METHOD_DESCRIPTIONS:
            st.info(
                f"**{selected_spatial_method}**: {ALL_METHOD_DESCRIPTIONS[selected_spatial_method]}"
            )

        # Store selection
        st.session_state["selected_method"] = selected_spatial_method

        # Navigation buttons
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("⬅️ Back to Properties"):
                st.session_state.current_step = 1
                st.session_state.point_properties_locked = False
                st.rerun()

        with col3:
            if st.button("Next: Select Temporal Method ➡️", type="primary"):
                st.session_state.current_step = 3
                st.rerun()

    else:
        st.error("No methods available. Please go back and adjust point properties.")
        if st.button("⬅️ Back to Properties"):
            st.session_state.current_step = 1
            st.session_state.point_properties_locked = False
            st.rerun()


def show_temporal_method_selection_step():
    st.header("⏱️ Step 3: Temporal Method Selection")
    st.write("Select the temporal generation method for your trajectories.")

    # Get current selection from session state
    if st.session_state.get("selected_method") in ALL_COMBINED_METHODS:
        st.info(
            "Since you selected a combined spatial-temporal method, the temporal method will be determined by the "
            "combined method's logic. You can skip this step."
        )

    available_temporal_methods = get_available_temporal_methods(
        st.session_state.get("config_temporal_dim_type", "continuous")
    )

    # Get current selection from session state
    current_method = st.session_state.get(
        "selected_temporal_method",
        available_temporal_methods[0] if available_temporal_methods else None,
    )

    if available_temporal_methods:
        selected_temporal_method = st.radio(
            "Choose a temporal generation method:",
            available_temporal_methods,
            index=(
                available_temporal_methods.index(current_method)
                if current_method in available_temporal_methods
                else 0
            ),
            help="Select the algorithm that will generate the time steps for your trajectories",
        )

        # Show method description
        if selected_temporal_method in ALL_METHOD_DESCRIPTIONS:
            st.info(
                f"**{selected_temporal_method}**: {ALL_METHOD_DESCRIPTIONS[selected_temporal_method]}"
            )

        # Store selection
        st.session_state["selected_temporal_method"] = selected_temporal_method
    if selected_temporal_method == "Constant Time":
        pass

    # Navigation buttons
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ Back to Spatial Method Selection"):
            st.session_state.current_step = 2
            st.session_state.point_properties_locked = False
            st.rerun()

    with col3:
        if st.button("Next: Configure Method ➡️", type="primary"):
            st.session_state.current_step = 4
            st.rerun()


def show_resampling_method_selection_step():
    st.header("⏱️ Step 4: Resampling Method Selection")
    st.write("Select the a resampling method for your trajectories.")

    # Get current selection from session state

    available_resampling_methods = get_available_resampling_methods(
        st.session_state.get("config_temporal_dim_type", "continuous")
    )

    # Get current selection from session state
    current_method = st.session_state.get(
        "selected_resampling_method",
        available_resampling_methods[0] if available_resampling_methods else None,
    )

    if available_resampling_methods:
        selected_resampling_method = st.radio(
            "Choose a resampling method:",
            available_resampling_methods,
            index=(
                available_resampling_methods.index(current_method)
                if current_method in available_resampling_methods
                else 0
            ),
            help="Select the algorithm that will resample your trajectories",
        )

        # Show method description
        if selected_resampling_method in ALL_METHOD_DESCRIPTIONS:
            st.info(
                f"**{selected_resampling_method}**: {ALL_METHOD_DESCRIPTIONS[selected_resampling_method]}"
            )

        # Store selection
        st.session_state["selected_resampling_method"] = selected_resampling_method

    # Navigation buttons
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ Back to Temporal Method Selection"):
            st.session_state.current_step = 3
            st.session_state.point_properties_locked = False
            st.rerun()

    with col3:
        if st.button("Next: Configuration ➡️", type="primary"):
            st.session_state.current_step = 5
            st.rerun()


def show_generation_step():
    # Navigation buttons
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ Back to Configuration Step"):
            st.session_state.current_step = 4
            st.session_state.point_properties_locked = False
            st.rerun()


##################################################
# Helper Functions
##################################################


def get_available_spatial_methods(dimension: str, spatial_dim_type: str) -> list[str]:
    """
    Determine which trajectory generation methods are available based on point properties.
    This is a placeholder implementation that needs to be refined based on actual method constraints.
    """
    all_methods = ALL_SPATIAL_METHODS

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
    all_temporal_methods = ALL_TEMPORAL_METHODS

    if temporal_dim_type == "continuous":
        return all_temporal_methods
    elif temporal_dim_type == "discrete":
        raise NotImplementedError(
            "Temporal methods for discrete temporal dimension type are not implemented yet."
        )


def get_available_combined_methods(
    dimension: str, spatial_dim_type: str, temporal_dim_type: str
) -> list[str]:
    all_combined_methods = ALL_COMBINED_METHODS
    if (
        dimension == "2D"
        and spatial_dim_type == "continuous"
        and temporal_dim_type == "continuous"
    ):
        return all_combined_methods
    else:
        return []  # No combined methods available for other configurations


def get_available_resampling_methods(temporal_dim_type: str) -> list[str]:
    all_resampling_methods = ALL_RESAMPLING_METHODS

    if temporal_dim_type == "continuous":
        return all_resampling_methods
    elif temporal_dim_type == "discrete":
        raise NotImplementedError(
            "Resampling methods for discrete temporal dimension type are not implemented yet."
        )


if __name__ == "__main__":
    main()
