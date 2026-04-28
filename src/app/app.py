import streamlit as st
from collections import OrderedDict
from src._1_point_properties import show_point_properties_step  # noqa
from src._2_spatial_method import show_spatial_method_selection_step  # noqa
from src._3_temporal_method import show_temporal_method_selection_step  # noqa
from src._4_resampling_method import show_resampling_method_selection_step  # noqa
from src._5_preview import show_preview_step  # noqa
from src._6_generate import show_generation_step  # noqa
from src._header import tum_header  # noqa
from src._footer import tum_footer  # noqa
from pathlib import Path


def main():
    page_icon = Path(__file__).parent / "assets/bgd_favicon.png"

    if not page_icon.exists():
        st.error(f"Favicon nicht gefunden unter: {page_icon.resolve()}")
        page_icon = None  # Fallback auf Standard-Icon

    st.set_page_config(page_title="TrajGen", page_icon=page_icon, layout="wide")

    tum_header("TrajGen Trajectory Generator")
    st.title("TrajGen Trajectory Generator")
    st.write(
        "Generate synthetic trajectory data using various spatial and temporal strategies."
    )

    # Initialize session state for step navigation
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1
    if "point_properties_locked" not in st.session_state:
        st.session_state.point_properties_locked = False
    if "max_step_reached" not in st.session_state:
        st.session_state.max_step_reached = 1

    # Update highest reached step whenever the user advances
    st.session_state.max_step_reached = max(
        st.session_state.max_step_reached, st.session_state.current_step
    )

    # Progress indicator
    progress_steps = [
        "Point Properties",
        "Spatial Method",
        "Temporal Method",
        "Resampling Method",
        "Preview",
        "Generation",
    ]
    current_step = st.session_state.current_step
    max_step_reached = st.session_state.max_step_reached

    # Create progress bar
    progress_cols = st.columns(len(progress_steps))
    for i, step_name in enumerate(progress_steps):
        step_number = i + 1
        with progress_cols[i]:
            button_key = f"nav_button_{step_number}"
            if step_number == current_step:
                if st.button(f"**🔄 {step_number}. {step_name}**", key=button_key):
                    st.session_state.current_step = step_number
                    st.rerun()
            elif step_number <= max_step_reached:
                if st.button(f"✅ {step_number}. {step_name}", key=button_key):
                    st.session_state.current_step = step_number
                    if step_number == 1:  # Going back to step 1 unlocks properties
                        st.session_state.point_properties_locked = False
                    st.rerun()
            else:
                st.button(
                    f"⭕ {step_number}. {step_name}",
                    key=button_key,
                    disabled=True,
                )

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
        show_preview_step()

    # Step 6: Generation
    elif current_step == 6:
        show_generation_step()

    tum_footer()


if __name__ == "__main__":
    main()
