import streamlit as st
from src.utils.helper import get_available_temporal_methods, universal_user_input_method
from src.method_overview import (
    ALL_METHOD_DESCRIPTIONS,
    ALL_COMBINED_METHODS,
    ALL_TEMPORAL_METHODS,
)
from src.utils.helper import debugger


def show_temporal_method_selection_step():
    st.header("⏱️ Step 3: Temporal Method Selection")
    st.write("Select the temporal generation method for your trajectories.")

    debugger()

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
    # Get current selection from session state
    if st.session_state.get("selected_method") in ALL_COMBINED_METHODS.keys():
        st.info(
            "Since you selected a combined spatial-temporal method, the temporal method will be determined by the "
            "combined method's logic. You can skip this step."
        )

    else:
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

            # Additional Method Properties
            strategy_cls = ALL_TEMPORAL_METHODS.get(selected_temporal_method)
            if strategy_cls is not None and hasattr(strategy_cls, "get_requirements"):
                st.subheader("🔧 Additional Method Properties")
                st.write(
                    "Configure specific properties for the selected temporal method."
                )
                requirements = strategy_cls.get_requirements()
                for req_name, req_info in requirements.items():
                    st.write(
                        f"- **{req_info['short_name']}**: {req_info['description']}"
                    )
                    universal_user_input_method(req_name, req_info)
                if (
                    selected_temporal_method == "Velocity"
                    or selected_temporal_method == "Acceleration"
                ):
                    col1, _, _ = st.columns([1, 1, 1])
                    with col1:
                        st.session_state["config_distance_function"] = st.selectbox(
                            "Select the distance function to use for velocity and acceleration calculations:",
                            options=["Euclidean", "Manhattan"],
                            index=(
                                0
                                if st.session_state.get("config_distance_function")
                                == "Euclidean"
                                else 1
                            ),
                            help="Choose the distance function that will be used to calculate spatial lengths "
                            "for velocity and acceleration based temporal methods.",
                        )

    # Navigation buttons
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ Back to Spatial Method Selection"):
            st.session_state.current_step = 2
            st.session_state.point_properties_locked = False
            st.rerun()

    with col3:
        if st.button("Next: Resampling Method ➡️", type="primary"):
            st.session_state.current_step = 4
            st.rerun()
