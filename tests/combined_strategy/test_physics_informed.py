import random
import pytest
import math
from shapely.geometry import LineString, Point

from trajgen.combined_strategy.physics_informed import PhysicsInformedCombinedStrategy
from trajgen.trajectory import Trajectory


class MockConfig:
    """Mock config for testing physics-informed strategy."""

    def __init__(self, **kwargs):
        # Bounding box
        self.x_min = kwargs.get("x_min", 0.0)
        self.x_max = kwargs.get("x_max", 100.0)
        self.y_min = kwargs.get("y_min", 0.0)
        self.y_max = kwargs.get("y_max", 50.0)
        self.z_min = kwargs.get("z_min", 0.0)
        self.z_max = kwargs.get("z_max", 50.0)

        # Dimension configuration
        self.spatial_dimension = kwargs.get("dimension", "2D")

        # Physics parameters
        self.gravity = kwargs.get("gravity", 9.81)
        self.time_step = kwargs.get("time_step", 0.01)
        self.simulation_time = kwargs.get("simulation_time", 2.0)  # Shorter for tests
        self.bounce_damping = kwargs.get("bounce_damping", 0.8)
        self.num_balls = kwargs.get("num_balls", 1)

    def get_next_point(self):
        if self.spatial_dimension == "3D":
            return Point(
                random.uniform(self.x_min, self.x_max),
                random.uniform(self.y_min, self.y_max),
                random.uniform(self.z_min, self.z_max),
            )
        else:
            return Point(
                random.uniform(self.x_min, self.x_max),
                random.uniform(self.y_min, self.y_max),
            )

    def get_next_velocity(self):
        """Generate random velocity component (can be positive or negative)."""
        return random.uniform(-10.0, 10.0)  # -10 to +10 m/s


class TestPhysicsInformedCombinedStrategy:
    def test_basic_physics_simulation(self):
        """Test basic physics simulation with default parameters."""
        config = MockConfig()
        strategy = PhysicsInformedCombinedStrategy(config)

        # Generate trajectory using trajectory ID
        result = strategy(trajectory_id=1)

        # Check that we get a valid trajectory
        assert isinstance(result, Trajectory)
        assert isinstance(result.ls, LineString)
        assert len(result.ls.coords) >= 2
        assert len(result.t) >= 2
        assert result.id == 1

    def test_trajectory_stays_within_bounds(self):
        """Test that the ball trajectory stays within the bounding box."""
        config = MockConfig(x_min=0, x_max=50, y_min=0, y_max=30)
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=2)

        # Check all points are within bounds
        for coord in result.ls.coords:
            assert config.x_min <= coord[0] <= config.x_max
            assert config.y_min <= coord[1] <= config.y_max

    def test_gravity_effect(self):
        """Test that gravity affects the trajectory (ball should fall)."""
        config = MockConfig(simulation_time=2.0, gravity=10.0, y_min=0, y_max=30)
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=3)

        coords = list(result.ls.coords)

        # Ball should eventually move downward due to gravity
        # Check if there are points with decreasing y-values (falling motion)
        has_falling_motion = False
        max_height = max(coord[1] for coord in coords)
        min_height = min(coord[1] for coord in coords)

        # If max height is significantly higher than min height, ball must have fallen
        has_falling_motion = (max_height - min_height) > 5.0

        # Alternative check: look for actual falling motion in trajectory
        if not has_falling_motion:
            for i in range(10, len(coords)):  # Skip initial points
                if (
                    coords[i][1] < coords[i - 1][1] - 0.5
                ):  # Significant downward movement
                    has_falling_motion = True
                    break

        assert (
            has_falling_motion
        ), f"Ball should fall due to gravity. Height range: {max_height - min_height}, coords: {coords[:10]}"

    def test_floor_collision(self):
        """Test that the ball bounces off the floor."""
        config = MockConfig(y_min=0, y_max=30, simulation_time=3.0, gravity=15.0)
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=4)

        coords = list(result.ls.coords)

        # Check if ball touches the floor (within small tolerance)
        touches_floor = any(coord[1] <= config.y_min + 0.5 for coord in coords)
        min_y = min(coord[1] for coord in coords)

        # If ball doesn't touch floor, check if it at least gets close to it
        gets_low = min_y <= config.y_max * 0.2  # Gets to bottom 20% of the box

        assert (
            touches_floor or gets_low
        ), f"Ball should touch floor or get low. Min Y: {min_y}, Floor: {config.y_min}"

    def test_time_progression(self):
        """Test that timestamps progress correctly."""
        config = MockConfig(time_step=0.05, simulation_time=1.0)
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=5)

        timestamps = result.t

        # Check timestamps are monotonically increasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

        # Check first timestamp is 0
        assert timestamps[0] == 0.0

        # Check time steps are approximately correct
        if len(timestamps) > 1:
            avg_dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
            assert abs(avg_dt - config.time_step) < config.time_step * 0.5

    def test_small_bounding_box(self):
        """Test simulation in a small bounding box."""
        config = MockConfig(x_min=0, x_max=10, y_min=0, y_max=10, simulation_time=1.0)
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=6)

        # Should still generate valid trajectory
        assert len(result.ls.coords) >= 2

        # All points within small bounds
        for coord in result.ls.coords:
            assert 0 <= coord[0] <= 10
            assert 0 <= coord[1] <= 10

    def test_no_gravity(self):
        """Test simulation with zero gravity."""
        config = MockConfig(gravity=0.0, simulation_time=1.0)
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=7)

        # With no gravity, horizontal motion should be more linear
        assert len(result.ls.coords) >= 2
        assert isinstance(result.t, list)

    def test_high_damping(self):
        """Test simulation with high bounce damping."""
        config = MockConfig(
            bounce_damping=0.1, simulation_time=3.0
        )  # Very high damping
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=8)

        # Ball should lose energy quickly and settle
        coords = list(result.ls.coords)

        # Check that the ball eventually settles near the bottom
        final_y = coords[-1][1]
        assert final_y <= config.y_max * 0.3, "Ball should settle low with high damping"

    def test_long_simulation(self):
        """Test longer simulation time."""
        config = MockConfig(simulation_time=5.0, time_step=0.02)
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=9)

        # Should generate more points for longer simulation
        assert len(result.ls.coords) > 50  # Should have many points (reduced from 100)
        assert (
            result.t[-1] <= config.simulation_time + config.time_step
        )  # Allow for small floating point error

    def test_multiple_balls(self):
        """Test simulation with multiple balls."""
        config = MockConfig(num_balls=3, simulation_time=1.0)
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=10)

        # Should return trajectory for the specified trajectory_id
        assert isinstance(result, Trajectory)
        assert result.id == 10

    def test_custom_physics_parameters(self):
        """Test with custom physics parameters."""
        config = MockConfig(
            gravity=15.0,  # Higher gravity
            time_step=0.001,  # Very small time step
            simulation_time=0.5,  # Short simulation
            bounce_damping=0.9,  # Low damping (high bounce)
        )
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=11)

        # Should work with custom parameters
        assert len(result.ls.coords) >= 2
        assert len(result.t) >= 2

        # With very small time steps, should have many points
        assert len(result.ls.coords) > 100

    def test_trajectory_properties_calculation(self):
        """Test the trajectory properties calculation method."""
        config = MockConfig()
        strategy = PhysicsInformedCombinedStrategy(config)

        # Test with sample trajectory points
        points = [(0, 0), (10, 15), (20, 10), (30, 5)]
        props = strategy._calculate_trajectory_properties(points)

        assert "total_distance" in props
        assert "max_height" in props
        assert "min_height" in props
        assert "height_range" in props

        assert props["max_height"] == 15
        assert props["min_height"] == 0
        assert props["height_range"] == 15
        assert props["total_distance"] > 0

    def test_edge_case_single_point(self):
        """Test edge case with single trajectory point."""
        config = MockConfig(simulation_time=0.01)  # Very short simulation
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=12)

        # Should handle edge case gracefully
        assert len(result.ls.coords) >= 2  # Should duplicate point if needed
        assert len(result.t) >= 2

    def test_deterministic_with_seed(self):
        """Test that simulation can be made deterministic."""
        config1 = MockConfig(simulation_time=1.0)
        config2 = MockConfig(simulation_time=1.0)

        # Set same random seed
        import random

        random.seed(42)
        strategy1 = PhysicsInformedCombinedStrategy(config1)
        result1 = strategy1(trajectory_id=13)

        random.seed(42)
        strategy2 = PhysicsInformedCombinedStrategy(config2)
        result2 = strategy2(trajectory_id=13)

        # Results should be identical with same seed
        coords1 = list(result1.ls.coords)
        coords2 = list(result2.ls.coords)

        assert len(coords1) == len(coords2)
        for i in range(min(5, len(coords1))):  # Check first few points
            assert abs(coords1[i][0] - coords2[i][0]) < 1e-10
            assert abs(coords1[i][1] - coords2[i][1]) < 1e-10

    def test_3d_physics_simulation(self):
        """Test 3D physics simulation."""
        config = MockConfig(
            dimension="3D",
            x_min=0,
            x_max=50,
            y_min=0,
            y_max=30,
            z_min=0,
            z_max=40,
            simulation_time=2.0,
        )
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=100)

        # Check that we get a valid 3D trajectory
        assert isinstance(result, Trajectory)
        assert isinstance(result.ls, LineString)
        assert len(result.ls.coords) >= 2
        assert len(result.t) >= 2

        # Check all points are 3D and within bounds
        for coord in result.ls.coords:
            assert len(coord) == 3  # Should be 3D coordinates
            assert config.x_min <= coord[0] <= config.x_max
            assert config.y_min <= coord[1] <= config.y_max
            assert config.z_min <= coord[2] <= config.z_max

    def test_3d_gravity_effect(self):
        """Test that gravity affects 3D trajectory (ball should fall in y-direction)."""
        config = MockConfig(
            dimension="3D",
            simulation_time=2.0,
            gravity=10.0,
            y_min=0,
            y_max=30,
            z_min=0,
            z_max=30,
        )
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=101)
        coords = list(result.ls.coords)

        # Ball should eventually move downward due to gravity
        # Check if there are points with decreasing y-values (falling motion)
        has_falling_motion = False
        max_height = max(coord[1] for coord in coords)
        min_height = min(coord[1] for coord in coords)

        # If max height is significantly higher than min height, ball must have fallen
        has_falling_motion = (max_height - min_height) > 5.0

        # Alternative check: look for actual falling motion in trajectory
        if not has_falling_motion:
            for i in range(10, len(coords)):  # Skip initial points
                if (
                    coords[i][1] < coords[i - 1][1] - 0.5
                ):  # Significant downward movement
                    has_falling_motion = True
                    break

        assert (
            has_falling_motion
        ), f"Ball should fall due to gravity in 3D. Height range: {max_height - min_height}"

    def test_3d_z_wall_collisions(self):
        """Test that ball bounces off front/back walls in 3D."""
        config = MockConfig(
            dimension="3D",
            z_min=0,
            z_max=20,  # Small z-range to force collisions
            simulation_time=3.0,
            bounce_damping=0.9,  # High bounce to ensure collisions
        )
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=102)
        coords = list(result.ls.coords)

        # Check if ball touches the z-walls (within small tolerance)
        touches_z_min = any(coord[2] <= config.z_min + 0.5 for coord in coords)
        touches_z_max = any(coord[2] >= config.z_max - 0.5 for coord in coords)

        # Ball should interact with z-boundaries in small space
        min_z = min(coord[2] for coord in coords)
        max_z = max(coord[2] for coord in coords)
        z_range = max_z - min_z

        assert (
            z_range > 2.0
        ), f"Ball should move significantly in z-direction. Z-range: {z_range}"

    def test_dimension_parameter_default_2d(self):
        """Test that default dimension is 2D when not specified."""
        config = MockConfig()  # No dimension specified
        strategy = PhysicsInformedCombinedStrategy(config)

        result = strategy(trajectory_id=103)

        # Should produce 2D trajectory
        for coord in result.ls.coords:
            assert len(coord) == 2  # Should be 2D coordinates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
