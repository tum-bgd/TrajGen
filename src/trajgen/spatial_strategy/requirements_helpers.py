"""Shared requirement dictionaries for spatial strategies."""


def bbox_requirements(spatial_dim: str = "2D") -> dict:
    """Return bounding-box requirement entries for the given dimension.

    Parameters
    ----------
    spatial_dim : str
        ``"2D"`` (default) or ``"3D"``.

    Returns
    -------
    dict
        Requirement entries for x/y (and optionally z) min/max.
    """
    reqs: dict = {
        "get_next_x_min": {
            "short_name": "X Min",
            "type": "get_float_function",
            "default": 0.0,
            "default_mode": "fixed for dataset",
            "description": "Bounding box minimum X coordinate.",
            "optional": False,
        },
        "get_next_x_max": {
            "short_name": "X Max",
            "type": "get_float_function",
            "default": 1.0,
            "default_mode": "fixed for dataset",
            "description": "Bounding box maximum X coordinate.",
            "optional": False,
        },
        "get_next_y_min": {
            "short_name": "Y Min",
            "type": "get_float_function",
            "default": 0.0,
            "default_mode": "fixed for dataset",
            "description": "Bounding box minimum Y coordinate.",
            "optional": False,
        },
        "get_next_y_max": {
            "short_name": "Y Max",
            "type": "get_float_function",
            "default": 1.0,
            "default_mode": "fixed for dataset",
            "description": "Bounding box maximum Y coordinate.",
            "optional": False,
        },
    }
    if spatial_dim == "3D":
        reqs["get_next_z_min"] = {
            "short_name": "Z Min",
            "type": "get_float_function",
            "default": 0.0,
            "default_mode": "fixed for dataset",
            "description": "Bounding box minimum Z coordinate.",
            "optional": False,
        }
        reqs["get_next_z_max"] = {
            "short_name": "Z Max",
            "type": "get_float_function",
            "default": 1.0,
            "default_mode": "fixed for dataset",
            "description": "Bounding box maximum Z coordinate.",
            "optional": False,
        }
    return reqs
