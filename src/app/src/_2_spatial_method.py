import streamlit as st
from src.utils.helper import (
    get_available_spatial_methods,
    get_available_combined_methods,
    debugger,
    universal_user_input_method,
)  # noqa

from src.method_overview import ALL_METHOD_DESCRIPTIONS, ALL_METHODS  # noqa


def show_spatial_method_selection_step():
    st.header("📐 Step 2: Spatial Method Selection")

    debugger()

    # Show locked point properties summary
    st.subheader("🔒 Point Properties (Locked)")
    locked_cols = st.columns(3)
    with locked_cols[0]:
        st.metric("Dimension", st.session_state.get("config_spatial_dimension", None))
    with locked_cols[1]:
        st.metric(
            "Spatial Type",
            st.session_state.get("config_spatial_dim_type", None),
        )
    with locked_cols[2]:
        st.metric(
            "Temporal Type",
            st.session_state.get("config_temporal_dim_type", None),
        )

    # Method selection
    available_spatial_methods = get_available_spatial_methods(
        st.session_state.get("config_spatial_dimension", None),
        st.session_state.get("config_spatial_dim_type", None),
    )

    available_combined_methods = get_available_combined_methods(
        st.session_state.get("config_spatial_dimension", None),
        st.session_state.get("config_spatial_dim_type", None),
        st.session_state.get("config_temporal_dim_type", None),
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
        # Manual session state management for method selection
        current_method = st.session_state.get("selected_method", available_methods[0])

        selected_spatial_method = st.radio(
            "Choose a spatial or combined spatial-temporal generation method:",
            available_methods,
            index=(
                available_methods.index(current_method)
                if current_method in available_methods
                else 0
            ),
            help="Select the algorithm that will generate your trajectories",
        )

        # Save to session state immediately
        st.session_state["selected_method"] = selected_spatial_method

        # Show method description
        if selected_spatial_method in ALL_METHOD_DESCRIPTIONS:
            st.info(
                f"**{selected_spatial_method}**: {ALL_METHOD_DESCRIPTIONS[selected_spatial_method]}"
            )

        # Additional Method Properties
        st.subheader("🔧 Additional Method Properties")
        st.write("Configure specific properties for the selected method.")

        spatial_dim = st.session_state.get("config_spatial_dimension", "2D")
        requirements = ALL_METHODS[selected_spatial_method].get_requirements(
            spatial_dim
        )
        for req_name, req_info in requirements.items():
            st.write(f"- **{req_info['short_name']}**: {req_info['description']}")
            universal_user_input_method(req_name, req_info)
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
