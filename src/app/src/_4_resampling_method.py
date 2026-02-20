import streamlit as st
from src.utils.helper import (
    get_available_resampling_methods,
    universal_user_input_method,
)
from src.method_overview import ALL_METHOD_DESCRIPTIONS, ALL_RESAMPLING_METHODS
from src.utils.helper import debugger


def show_resampling_method_selection_step():
    st.header("⏱️ Step 4: Resampling Method Selection")
    st.write("Select the a resampling method for your trajectories.")

    debugger()
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

        # Additional Method Properties
        strategy_cls = ALL_RESAMPLING_METHODS.get(selected_resampling_method)
        if strategy_cls is not None and hasattr(strategy_cls, "get_requirements"):
            st.subheader("🔧 Additional Method Properties")
            st.write("Configure specific properties for the selected resampling method.")
            requirements = strategy_cls.get_requirements()
            for req_name, req_info in requirements.items():
                st.write(f"- **{req_info['short_name']}**: {req_info['description']}")
                universal_user_input_method(req_name, req_info)

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
