import streamlit as st
from src.utils.helper import (
    get_available_spatial_methods,
    get_available_combined_methods,
    debugger,
    universal_user_input_method,
)  # noqa

from src.method_overview import ALL_METHOD_DESCRIPTIONS, ALL_METHODS  # noqa


def _show_osm_bbox_map() -> None:
    """Render an interactive Folium map so the user can draw a bounding box.

    The drawn rectangle updates the session-state keys
    ``config_get_next_x_min/x_max/y_min/y_max`` (longitude / latitude) and
    triggers a rerun so the numeric inputs above reflect the new values.
    """
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium

    st.subheader("Select Area on Map")
    st.write(
        "Draw a rectangle on the map to set the bounding box. "
        "The Longitude / Latitude fields above will update automatically."
    )

    # Read current bbox from session state, fall back to Munich city centre
    lon_min = float(st.session_state.get("config_get_next_x_min", 11.54))
    lon_max = float(st.session_state.get("config_get_next_x_max", 11.62))
    lat_min = float(st.session_state.get("config_get_next_y_min", 48.12))
    lat_max = float(st.session_state.get("config_get_next_y_max", 48.17))

    # Guard against stale generic bbox values (e.g. 0–1 from other methods)
    if not (-180 <= lon_min < lon_max <= 180 and -90 <= lat_min < lat_max <= 90):
        lon_min, lon_max, lat_min, lat_max = 11.54, 11.62, 48.12, 48.17

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    Draw(
        export=False,
        draw_options={
            "rectangle": True,
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
    ).add_to(m)

    # Show current bbox as a semi-transparent blue overlay
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color="blue",
        fill=True,
        fill_opacity=0.1,
    ).add_to(m)

    result = st_folium(m, height=450, use_container_width=True)

    # Update session state when the user finishes drawing a new rectangle
    drawing = result and result.get("last_active_drawing")
    if drawing and drawing.get("geometry", {}).get("type") == "Polygon":
        coords = drawing["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        new_x_min = min(lons)
        new_x_max = max(lons)
        new_y_min = min(lats)
        new_y_max = max(lats)

        # Only rerun when the drawn bbox is actually different from current values
        if (
            st.session_state.get("config_get_next_x_min") != new_x_min
            or st.session_state.get("config_get_next_x_max") != new_x_max
            or st.session_state.get("config_get_next_y_min") != new_y_min
            or st.session_state.get("config_get_next_y_max") != new_y_max
        ):
            st.session_state["config_get_next_x_min"] = new_x_min
            st.session_state["config_get_next_x_max"] = new_x_max
            st.session_state["config_get_next_y_min"] = new_y_min
            st.session_state["config_get_next_y_max"] = new_y_max
            # Ensure mode keys are set so Config.__getattr__ can resolve them
            for key in ("x_min", "x_max", "y_min", "y_max"):
                st.session_state[f"config_get_next_{key}_mode"] = "fixed for dataset"
            st.rerun()


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

        if selected_spatial_method == "OSM Sampling":
            _show_osm_bbox_map()

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
