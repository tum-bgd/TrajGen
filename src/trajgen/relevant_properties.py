from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class Properties:
    ##### POINT PROPERTIES #####
    config_dimension_mode_options = ["fixed for dataset", "fixed for trajectory"]
    config_dimension = ["2D", "3D"]

    ###### GEOMETRIC PROPERTIES ######
    # Length parameters options
    config_get_next_length_mode = {
        "short_name": "Trajectory Length Mode",
        "type": "get_int_function",
        "description": "Select how the length of trajectories is determined.",
        "options": OrderedDict(
            [
                ("undefined", []),
                ("fixed for dataset", ["config_length"]),
                (
                    "fixed for trajectory",
                    ["config_get_next_length_distribution"],
                ),
            ]
        ),
    }
    config_get_next_length_distribution = {
        "short_name": "Trajectory Length Distribution",
        "type": "get_int_function",
        "description": "Select the distribution of trajectory lengths.",
        "options": OrderedDict(
            [
                ("uniform", ["config_length_min", "config_length_max"]),
                ("normal", ["config_length_mean", "config_length_std"]),
            ]
        ),
    }

    # Spatial step size parameters options
    config_spatial_step_size_mode = {
        "short_name": "Spatial Step Size Mode",
        "type": "get_float_function",
        "description": "Select how the spatial step size of trajectories is determined.",
        "options": OrderedDict(
            [
                ("undefined", []),
                ("fixed for dataset", ["config_spatial_step_size"]),
                ("fixed for trajectory", ["config_spatial_step_size_distribution"]),
            ]
        ),
    }

    config_spatial_step_size_distribution = {
        "short_name": "Spatial Step Size Distribution",
        "type": "get_float_function",
        "description": "Select the distribution of spatial step sizes.",
        "options": OrderedDict(
            [
                (None, []),
                (
                    "uniform",
                    ["config_spatial_step_size_min", "config_spatial_step_size_max"],
                ),
                (
                    "normal",
                    ["config_spatial_step_size_mean", "config_spatial_step_size_std"],
                ),
            ]
        ),
    }

    # Spatial extent parameters options
    config_spatial_extent_mode = {
        "short_name": "Spatial Extent Mode",
        "type": "get_string_function",
        "description": "Select how the spatial extent of trajectories is determined.",
        "options": OrderedDict(
            [
                ("undefined", []),
                ("fixed for dataset", ["config_spatial_extent_mode_form_options"]),
                (
                    "fixed for trajectory",
                    [
                        "config_spatial_extent_mode_form_options",
                        "config_spatial_extent_distribution_options",
                    ],
                ),
            ]
        ),
    }
    config_spatial_extent_mode_form = {
        "short_name": "Spatial Extent Mode Form",
        "type": "get_string_function",
        "description": "Select the form of spatial extent.",
        "options": ["bounding_box"],
    }
    config_spatial_extent_distribution = {
        "short_name": "Spatial Extent Distribution",
        "type": "get_string_function",
        "description": "Select the distribution of spatial extent.",
        "options": [None, "uniform", "normal"],
    }

    # Smoothness parameters options
    config_smoothness_mode = {
        "short_name": "Smoothness Mode",
        "type": "get_string_function",
        "description": "Select how the smoothness of trajectories is determined.",
        "options": OrderedDict(
            [
                ("undefined", []),
                ("degrees_per_distance_unit", ["config_smoothness"]),
            ]
        ),
    }
    config_smoothness_distribution = {
        "short_name": "Smoothness Distribution",
        "type": "get_string_function",
        "description": "Select the distribution of smoothness.",
        "options": OrderedDict(
            [
                (None, []),
                ("uniform", ["config_smoothness_min", "config_smoothness_max"]),
                ("normal", ["config_smoothness_mean", "config_smoothness_std"]),
            ]
        ),
    }

    # Shape parameters options
    config_shape_mode = {
        "short_name": "Shape Mode",
        "type": "get_string_function",
        "description": "Select the shape mode for trajectories.",
        "options": OrderedDict(
            [
                ("Rectlinear", []),
            ]
        ),
    }

    config_shape_closed_loop_options = [True, False]

    ############# KINEMATIC PROPERTIES #############
    # Temporal extent parameters options
    config_start_time_mode_options = OrderedDict(
        [
            ("undefined", []),
            ("fixed for dataset", ["config_start_time"]),
            ("fixed for trajectory", ["config_start_time_distribution_options"]),
        ]
    )
    config_start_time_distribution_options = OrderedDict(
        [
            (None, []),
            ("uniform", ["config_start_time_min", "config_start_time_max"]),
            ("normal", ["config_start_time_mean", "config_start_time_std"]),
        ]
    )

    config_temporal_extent_mode_options = OrderedDict(
        [
            ("undefined", []),
            ("fixed for dataset", ["config_temporal_extent"]),
            (
                "fixed for trajectory",
                [
                    "config_temporal_extent_distribution_options",
                ],
            ),
        ]
    )

    config_temporal_extent_distribution_options = OrderedDict(
        [
            ("undefined", []),
            (
                "uniform-constant_length",
                [
                    "config_temporal_extent_min",
                    "config_temporal_extent_max",
                ],
            ),
            ("normal", ["config_temporal_extent_mean", "config_temporal_extent_std"]),
        ]
    )

    # Lentgth parameters options
    config_length = {
        "short_name": "Trajectory Length",
        "type": "int",
        "description": "Fixed trajectory length for fixed for dataset mode",
    }
    config_length_min = {
        "short_name": "Min Trajectory Length",
        "type": "int",
        "description": "Minimum trajectory length for uniform distribution",
    }
    config_length_max = {
        "short_name": "Max Trajectory Length",
        "type": "int",
        "description": "Maximum trajectory length for uniform distribution",
    }
    config_length_mean = {
        "short_name": "Mean Trajectory Length",
        "type": "float",
        "description": "Mean trajectory length for normal distribution",
    }
    config_length_std = {
        "short_name": "Trajectory Length Std Dev",
        "type": "float",
        "description": "Standard deviation of trajectory length for normal distribution",
    }

    # Freespace strategy specific properties
    config_num_obstacles = {
        "short_name": "Number of Obstacles",
        "type": "int",
        "description": "Number of random obstacles to generate in the space.",
    }
    config_deviation_factor = {
        "short_name": "Deviation Factor",
        "type": "float",
        "description": "Controls how much trajectories deviate from straight line to avoid obstacles.",
    }
