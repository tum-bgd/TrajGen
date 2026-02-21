import sys
import os

# Add src/app to python path to allow importing config
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest

# Now we can import app.config if src is root, or just config if app is root context
# But since file is in src/app, importing config directly might work if we add src/app
# Let's try importing config from .config
try:
    from trajgen.config import Config
except ImportError:
    # If standard import fails, try relative or direct file import approach for test
    sys.path.append(os.path.dirname(__file__))
    from config import Config


class TestFrontendConfig:
    def test_velocity_new_config(self):
        # Setup state with new velocity configuration (fixed for trajectory / uniform)
        state = {
            "config_get_next_velocity_mode": "fixed for trajectory",
            "config_get_next_velocity_distribution": "uniform",
            "config_get_next_velocity_min": 5.0,
            "config_get_next_velocity_max": 10.0,
            "config_seed": 42,
        }
        config = Config(state)

        # Should use the new distribution keys
        v = config.get_next_velocity()
        assert 5.0 <= v <= 10.0

    def test_velocity_dataset_mode(self):
        state = {
            "config_get_next_velocity_mode": "fixed for dataset",
            "config_get_next_velocity": 7.5,
            "config_seed": 42,
        }
        config = Config(state)
        assert config.get_next_velocity() == 7.5

    def test_acceleration_new_config(self):
        state = {
            "config_get_next_acceleration_mode": "fixed for dataset",
            "config_get_next_acceleration": 2.5,
            "config_seed": 42,
        }
        config = Config(state)
        assert config.get_next_acceleration() == 2.5

    def test_tmin_behavior(self):
        # Current app config uses config_tmin directly
        # We should check if we need to align this with start_time
        state = {"config_tmin": 10.0, "config_seed": 42}
        config = Config(state)
        assert config.get_next_tmin() == 10.0

    def test_legacy_min_max_velocity(self):
        # State with old style min/max
        state = {
            "config_min_velocity": 3.0,
            "config_max_velocity": 8.0,
            "config_seed": 42,
        }
        config = Config(state)
        v = config.get_next_velocity()
        assert 3.0 <= v <= 8.0
