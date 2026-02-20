import pytest
from shapely.geometry import Point
from trajgen.point_generator import (
    RandomPointGenerator2D,
    GridPointGenerator2D,
    RandomPointGenerator3D,
    GridPointGenerator3D,
)


class TestRandomPointGenerator2D:
    """Test suite for RandomPointGenerator class."""

    def test_init(self):
        """Test that RandomPointGenerator initializes correctly."""
        generator = RandomPointGenerator2D((0, 100), (0, 200), seed=42)

        assert generator.row_range == (0, 100)
        assert generator.col_range == (0, 200)

    def test_generate_correct_number_of_points(self):
        """Test that generate() returns the correct number of points."""
        num_points = 5
        generator = RandomPointGenerator2D((0, 100), (0, 200), seed=123)
        points = generator(num_points)

        assert len(points) == num_points

    def test_generate_returns_point_objects(self):
        """Test that generate() returns a list of Point objects."""
        generator = RandomPointGenerator2D((0, 100), (0, 200), seed=123)
        points = generator(3)

        assert isinstance(points, list)
        for point in points:
            assert isinstance(point, Point)

    def test_generate_points_within_range(self):
        """Test that generated points are within the specified ranges."""
        row_range = (10, 50)
        col_range = (20, 80)
        generator = RandomPointGenerator2D(row_range, col_range, seed=123)
        points = generator(10)

        for point in points:
            # Point.x corresponds to the first coordinate (row)
            # Point.y corresponds to the second coordinate (col)
            assert row_range[0] <= point.x <= row_range[1]
            assert col_range[0] <= point.y <= col_range[1]

    def test_generate_with_zero_points(self):
        """Test that generate() works correctly with zero points."""
        generator = RandomPointGenerator2D((0, 100), (0, 200), seed=123)
        points = generator(0)

        assert len(points) == 0
        assert isinstance(points, list)

    def test_generate_with_single_point(self):
        """Test that generate() works correctly with a single point."""
        generator = RandomPointGenerator2D((0, 100), (0, 200), seed=123)
        points = generator(1)

        assert len(points) == 1
        assert isinstance(points[0], Point)

    def test_generate_with_negative_ranges(self):
        """Test that generate() works with negative coordinate ranges."""
        row_range = (-50, -10)
        col_range = (-100, -20)
        generator = RandomPointGenerator2D(row_range, col_range, seed=123)
        points = generator(5)

        assert len(points) == 5
        for point in points:
            assert row_range[0] <= point.x <= row_range[1]
            assert col_range[0] <= point.y <= col_range[1]

    def test_generate_with_same_min_max_range(self):
        """Test that generate() works when min and max are the same."""
        row_range = (42, 42)
        col_range = (84, 84)
        generator = RandomPointGenerator2D(row_range, col_range, seed=123)
        points = generator(3)
        assert len(points) == 3
        for point in points:
            assert point.x == 42
            assert point.y == 84

    def test_generate_randomness(self):
        """Test that generate() produces different results on multiple calls."""
        generator = RandomPointGenerator2D((0, 100), (0, 200), seed=123)

        # Generate two sets of points
        points1 = generator(10)
        points2 = generator(10)

        # Convert to coordinate tuples for easier comparison
        coords1 = [(p.x, p.y) for p in points1]
        coords2 = [(p.x, p.y) for p in points2]

        # It's extremely unlikely (but theoretically possible) for all points to be identical
        # In practice, this test should always pass
        assert coords1 != coords2

    def test_different_seeds_produce_different_points(self):
        """Test that different seeds produce different sequences of points."""
        generator1 = RandomPointGenerator2D((0, 100), (0, 200), seed=1)
        generator2 = RandomPointGenerator2D((0, 100), (0, 200), seed=2)

        points1 = generator1(10)
        points2 = generator2(10)

        coords1 = [(p.x, p.y) for p in points1]
        coords2 = [(p.x, p.y) for p in points2]

        assert coords1 != coords2

    def test_same_seed_produces_same_points(self):
        """Test that the same seed produces the same sequence of points."""
        generator1 = RandomPointGenerator2D((0, 100), (0, 200), seed=42)
        generator2 = RandomPointGenerator2D((0, 100), (0, 200), seed=42)

        points1 = generator1(10)
        points2 = generator2(10)

        coords1 = [(p.x, p.y) for p in points1]
        coords2 = [(p.x, p.y) for p in points2]

        assert coords1 == coords2

    @pytest.mark.parametrize(
        "num_points,row_range,col_range",
        [
            (5, (0, 10), (0, 20)),
            (1, (-5, 5), (-10, 10)),
            (100, (0, 1000), (0, 2000)),
            (0, (10, 50), (20, 80)),
        ],
    )
    def test_generate_parametrized(self, num_points, row_range, col_range):
        """Parametrized test for various input combinations."""
        generator = RandomPointGenerator2D(row_range, col_range, seed=123)
        points = generator(num_points)

        assert len(points) == num_points
        for point in points:
            assert isinstance(point, Point)
            if num_points > 0:  # Only check ranges if we have points
                assert row_range[0] <= point.x <= row_range[1]
                assert col_range[0] <= point.y <= col_range[1]


class TestGridPointGenerator2D:
    """Test suite for GridPointGenerator class."""

    def test_init(self):
        """Test that GridPointGenerator initializes correctly."""

        generator = GridPointGenerator2D((0, 10), (0, 20), step=5)

        assert generator.row_range == (0, 10)
        assert generator.col_range == (0, 20)
        assert generator.step == 5

    def test_generate_correct_number_of_points(self):
        """Test that generate() returns the correct number of points."""
        num_points = 6
        generator = GridPointGenerator2D((0, 10), (0, 20), step=5)
        points = generator(num_points)

        assert len(points) == num_points

    def test_generate_returns_point_objects(self):
        """Test that generate() returns a list of Point objects."""
        generator = GridPointGenerator2D((0, 10), (0, 20), step=5)
        points = generator(4)

        assert isinstance(points, list)
        for point in points:
            assert isinstance(point, Point)

    def test_generate_points_within_range(self):
        """Test that generated points are within the specified ranges."""
        row_range = (0, 10)
        col_range = (0, 20)
        step = 5
        generator = GridPointGenerator2D(row_range, col_range, step)
        points = generator(10)

        for point in points:
            assert row_range[0] <= point.x <= row_range[1]
            assert col_range[0] <= point.y <= col_range[1]

    def test_generate_with_zero_points(self):
        """Test that generate() works correctly with zero points."""
        generator = GridPointGenerator2D((0, 10), (0, 20), step=5)
        points = generator(0)

        assert len(points) == 0
        assert isinstance(points, list)

    def test_generate_with_single_point(self):
        """Test that generate() works correctly with a single point."""
        generator = GridPointGenerator2D((0, 10), (0, 20), step=5)
        points = generator(1)

        assert len(points) == 1
        assert isinstance(points[0], Point)

    def test_generate_with_negative_ranges(self):
        """Test that generate() works with negative coordinate ranges."""
        row_range = (-10, 0)
        col_range = (-20, 0)
        step = 5
        generator = GridPointGenerator2D(row_range, col_range, step)
        points = generator(5)

        assert len(points) == 5
        for point in points:
            assert row_range[0] <= point.x <= row_range[1]
            assert col_range[0] <= point.y <= col_range[1]

    def test_generate_stepsize(self):
        """Test that generated points respect the step size."""
        row_range = (0, 10)
        col_range = (0, 20)
        step = 5
        generator = GridPointGenerator2D(row_range, col_range, step)
        points = generator(100)  # Generate a large number to ensure full grid coverage

        expected_rows = set(range(row_range[0], row_range[1] + 1, step))
        expected_cols = set(range(col_range[0], col_range[1] + 1, step))

        generated_rows = set(int(point.x) for point in points)
        generated_cols = set(int(point.y) for point in points)

        assert generated_rows.issubset(expected_rows)
        assert generated_cols.issubset(expected_cols)

    def test_point_cycling_when_more_requested_than_available(self):
        """Test that points are cycled when more are requested than available."""
        # Create a small 2x2 grid (4 total points)
        generator = GridPointGenerator2D((0, 1), (0, 1), step=1)

        # Request more points than available (10 > 4)
        points = generator(10)

        assert len(points) == 10

        # Verify that points come from the expected set
        coords = [(int(p.x), int(p.y)) for p in points]

        # All coordinates should be from the expected grid
        expected_grid_points = {(0, 0), (0, 1), (1, 0), (1, 1)}
        actual_points = set(coords)
        assert actual_points.issubset(expected_grid_points)

        # Should have all 4 unique points represented
        unique_coords = set(coords)
        assert len(unique_coords) == 4  # Should have exactly 4 unique points
        assert unique_coords == expected_grid_points

    def test_random_cycling_produces_different_orders(self):
        """Test that random cycling produces different orders on multiple calls."""
        # Create a small 2x2 grid (4 total points)
        generator = GridPointGenerator2D((0, 1), (0, 1), step=1)

        # Generate multiple sequences and check they're different
        # (This is probabilistic, but should pass with very high probability)
        sequences = []
        for _ in range(10):
            points = generator(12)  # Request 3 full cycles
            coords = [(int(p.x), int(p.y)) for p in points]
            sequences.append(coords)

        # Check that not all sequences are identical
        # With random shuffling, it's extremely unlikely they'd all be the same
        unique_sequences = set(tuple(seq) for seq in sequences)
        assert (
            len(unique_sequences) > 1
        ), "Random cycling should produce different orders"


class TestRandomPointGenerator3D:
    """Test suite for RandomPointGenerator3D class."""

    def test_init(self):
        """Test that RandomPointGenerator3D initializes correctly."""
        generator = RandomPointGenerator3D((0, 100), (0, 200), (0, 300), seed=42)

        assert generator.x_range == (0, 100)
        assert generator.y_range == (0, 200)
        assert generator.z_range == (0, 300)

    def test_generate_correct_number_of_points(self):
        """Test that generate() returns the correct number of points."""
        num_points = 5
        generator = RandomPointGenerator3D((0, 100), (0, 200), (0, 300), seed=123)
        points = generator(num_points)

        assert len(points) == num_points

    def test_generate_returns_point_objects(self):
        """Test that generate() returns a list of Point objects."""
        generator = RandomPointGenerator3D((0, 100), (0, 200), (0, 300), seed=123)
        points = generator(3)

        assert isinstance(points, list)
        for point in points:
            assert isinstance(point, Point)

    def test_generate_points_within_range(self):
        """Test that generated points are within the specified ranges."""
        x_range = (10, 50)
        y_range = (20, 80)
        z_range = (30, 90)
        generator = RandomPointGenerator3D(x_range, y_range, z_range, seed=123)
        points = generator(10)

        for point in points:
            # Check that all coordinates are within their respective ranges
            assert x_range[0] <= point.x <= x_range[1]
            assert y_range[0] <= point.y <= y_range[1]
            assert z_range[0] <= point.z <= z_range[1]

    def test_generate_with_zero_points(self):
        """Test that generate() works correctly with zero points."""
        generator = RandomPointGenerator3D((0, 100), (0, 200), (0, 300), seed=123)
        points = generator(0)

        assert len(points) == 0
        assert isinstance(points, list)

    def test_generate_with_single_point(self):
        """Test that generate() works correctly with a single point."""
        generator = RandomPointGenerator3D((0, 100), (0, 200), (0, 300), seed=123)
        points = generator(1)

        assert len(points) == 1
        assert isinstance(points[0], Point)

    def test_generate_with_negative_ranges(self):
        """Test that generate() works with negative coordinate ranges."""
        x_range = (-50, -10)
        y_range = (-100, -20)
        z_range = (-150, -30)
        generator = RandomPointGenerator3D(x_range, y_range, z_range, seed=123)
        points = generator(5)

        assert len(points) == 5
        for point in points:
            assert x_range[0] <= point.x <= x_range[1]
            assert y_range[0] <= point.y <= y_range[1]
            assert z_range[0] <= point.z <= z_range[1]

    def test_generate_with_same_min_max_range(self):
        """Test that generate() works when min and max are the same."""
        x_range = (42, 42)
        y_range = (84, 84)
        z_range = (126, 126)
        generator = RandomPointGenerator3D(x_range, y_range, z_range, seed=123)
        points = generator(3)
        assert len(points) == 3
        for point in points:
            assert point.x == 42
            assert point.y == 84
            assert point.z == 126

    def test_generate_single_point_without_num_points(self):
        """Test that calling without num_points returns a single Point."""
        generator = RandomPointGenerator3D((0, 100), (0, 200), (0, 300), seed=123)
        point = generator()

        assert isinstance(point, Point)
        assert 0 <= point.x <= 100
        assert 0 <= point.y <= 200
        assert 0 <= point.z <= 300

    def test_generate_randomness(self):
        """Test that generate() produces different results on multiple calls."""
        generator = RandomPointGenerator3D((0, 100), (0, 200), (0, 300), seed=123)

        # Generate two sets of points
        points1 = generator(10)
        points2 = generator(10)

        # Convert to coordinate tuples for easier comparison
        coords1 = [(p.x, p.y, p.z) for p in points1]
        coords2 = [(p.x, p.y, p.z) for p in points2]

        # It's extremely unlikely for all points to be identical
        assert coords1 != coords2

    def test_different_seeds_produce_different_points(self):
        """Test that different seeds produce different sequences of points."""
        generator1 = RandomPointGenerator3D((0, 100), (0, 200), (0, 300), seed=1)
        generator2 = RandomPointGenerator3D((0, 100), (0, 200), (0, 300), seed=2)

        points1 = generator1(10)
        points2 = generator2(10)

        coords1 = [(p.x, p.y, p.z) for p in points1]
        coords2 = [(p.x, p.y, p.z) for p in points2]

        assert coords1 != coords2

    def test_same_seed_produces_same_points(self):
        """Test that the same seed produces the same sequence of points."""
        generator1 = RandomPointGenerator3D((0, 100), (0, 200), (0, 300), seed=42)
        generator2 = RandomPointGenerator3D((0, 100), (0, 200), (0, 300), seed=42)

        points1 = generator1(10)
        points2 = generator2(10)

        coords1 = [(p.x, p.y, p.z) for p in points1]
        coords2 = [(p.x, p.y, p.z) for p in points2]

        assert coords1 == coords2

    @pytest.mark.parametrize(
        "num_points,x_range,y_range,z_range",
        [
            (5, (0, 10), (0, 20), (0, 30)),
            (1, (-5, 5), (-10, 10), (-15, 15)),
            (50, (0, 1000), (0, 2000), (0, 3000)),
            (0, (10, 50), (20, 80), (30, 90)),
        ],
    )
    def test_generate_parametrized(self, num_points, x_range, y_range, z_range):
        """Parametrized test for various input combinations."""
        generator = RandomPointGenerator3D(x_range, y_range, z_range, seed=123)
        points = generator(num_points)

        assert len(points) == num_points
        for point in points:
            assert isinstance(point, Point)
            if num_points > 0:  # Only check ranges if we have points
                assert x_range[0] <= point.x <= x_range[1]
                assert y_range[0] <= point.y <= y_range[1]
                assert z_range[0] <= point.z <= z_range[1]


class TestGridPointGenerator3D:
    """Test suite for GridPointGenerator3D class."""

    def test_init(self):
        """Test that GridPointGenerator3D initializes correctly."""
        generator = GridPointGenerator3D((0, 10), (0, 20), (0, 30), step=5)

        assert generator.x_range == (0, 10)
        assert generator.y_range == (0, 20)
        assert generator.z_range == (0, 30)
        assert generator.step == 5

    def test_generate_correct_number_of_points(self):
        """Test that generate() returns the correct number of points."""
        num_points = 8
        generator = GridPointGenerator3D((0, 10), (0, 20), (0, 30), step=10)
        points = generator(num_points)

        assert len(points) == num_points

    def test_generate_returns_point_objects(self):
        """Test that generate() returns a list of Point objects."""
        generator = GridPointGenerator3D((0, 10), (0, 20), (0, 30), step=10)
        points = generator(4)

        assert isinstance(points, list)
        for point in points:
            assert isinstance(point, Point)

    def test_generate_points_within_range(self):
        """Test that generated points are within the specified ranges."""
        x_range = (0, 10)
        y_range = (0, 20)
        z_range = (0, 30)
        step = 5
        generator = GridPointGenerator3D(x_range, y_range, z_range, step)
        points = generator(20)

        for point in points:
            assert x_range[0] <= point.x <= x_range[1]
            assert y_range[0] <= point.y <= y_range[1]
            assert z_range[0] <= point.z <= z_range[1]

    def test_generate_with_zero_points(self):
        """Test that generate() works correctly with zero points."""
        generator = GridPointGenerator3D((0, 10), (0, 20), (0, 30), step=5)
        points = generator(0)

        assert len(points) == 0
        assert isinstance(points, list)

    def test_generate_with_single_point(self):
        """Test that generate() works correctly with a single point."""
        generator = GridPointGenerator3D((0, 10), (0, 20), (0, 30), step=5)
        points = generator(1)

        assert len(points) == 1
        assert isinstance(points[0], Point)

    def test_generate_single_point_without_num_points(self):
        """Test that calling without num_points returns the first grid point."""
        generator = GridPointGenerator3D((0, 10), (0, 20), (0, 30), step=5)
        point = generator()

        assert isinstance(point, Point)
        # Should return the first point in the grid (0, 0, 0)
        assert point.x == 0
        assert point.y == 0
        assert point.z == 0

    def test_generate_with_negative_ranges(self):
        """Test that generate() works with negative coordinate ranges."""
        x_range = (-10, 0)
        y_range = (-20, 0)
        z_range = (-30, 0)
        step = 5
        generator = GridPointGenerator3D(x_range, y_range, z_range, step)
        points = generator(10)

        assert len(points) == 10
        for point in points:
            assert x_range[0] <= point.x <= x_range[1]
            assert y_range[0] <= point.y <= y_range[1]
            assert z_range[0] <= point.z <= z_range[1]

    def test_generate_stepsize(self):
        """Test that generated points respect the step size."""
        x_range = (0, 10)
        y_range = (0, 20)
        z_range = (0, 30)
        step = 5
        generator = GridPointGenerator3D(x_range, y_range, z_range, step)
        points = generator(100)  # Generate enough to cover the full grid

        expected_x_values = set(range(x_range[0], x_range[1] + 1, step))
        expected_y_values = set(range(y_range[0], y_range[1] + 1, step))
        expected_z_values = set(range(z_range[0], z_range[1] + 1, step))

        generated_x_values = set(int(point.x) for point in points)
        generated_y_values = set(int(point.y) for point in points)
        generated_z_values = set(int(point.z) for point in points)

        assert generated_x_values.issubset(expected_x_values)
        assert generated_y_values.issubset(expected_y_values)
        assert generated_z_values.issubset(expected_z_values)

    def test_grid_completeness_small(self):
        """Test that a small grid generates all expected points."""
        x_range = (0, 2)
        y_range = (0, 2)
        z_range = (0, 2)
        step = 1
        generator = GridPointGenerator3D(x_range, y_range, z_range, step)
        points = generator(100)  # Request more than the grid size

        # Expected grid: 3x3x3 = 27 points
        expected_coords = set()
        for x in range(0, 3, 1):
            for y in range(0, 3, 1):
                for z in range(0, 3, 1):
                    expected_coords.add((x, y, z))

        generated_coords = set((int(p.x), int(p.y), int(p.z)) for p in points)

        # All expected coordinates should be present in generated points
        assert expected_coords.issubset(generated_coords)

    def test_grid_ordering(self):
        """Test that grid points are generated from the expected set."""
        generator = GridPointGenerator3D((0, 2), (0, 2), (0, 2), step=1)
        points = generator(8)  # Get first 8 points

        # With random cycling, we can't guarantee order, but we can verify
        # that all points come from the expected grid
        expected_coords = set()
        for x in range(0, 3, 1):
            for y in range(0, 3, 1):
                for z in range(0, 3, 1):
                    expected_coords.add((x, y, z))

        actual_coords = set((int(p.x), int(p.y), int(p.z)) for p in points)

        # All generated points should be from the expected grid
        assert actual_coords.issubset(expected_coords)
        # We should have exactly 8 points
        assert len(points) == 8

    @pytest.mark.parametrize(
        "x_range,y_range,z_range,step",
        [
            ((0, 4), (0, 6), (0, 8), 2),
            ((-5, 5), (-10, 10), (-15, 15), 5),
            ((0, 1), (0, 1), (0, 1), 1),
            ((10, 20), (30, 40), (50, 60), 10),
        ],
    )
    def test_generate_parametrized(self, x_range, y_range, z_range, step):
        """Parametrized test for various input combinations."""
        generator = GridPointGenerator3D(x_range, y_range, z_range, step)
        points = generator(5)

        assert len(points) == 5
        for point in points:
            assert isinstance(point, Point)
            assert x_range[0] <= point.x <= x_range[1]
            assert y_range[0] <= point.y <= y_range[1]
            assert z_range[0] <= point.z <= z_range[1]

    def test_point_cycling_when_more_requested_than_available(self):
        """Test that points are cycled when more are requested than available."""
        # Create a small 2x2x2 grid (8 total points)
        generator = GridPointGenerator3D((0, 1), (0, 1), (0, 1), step=1)

        # Request more points than available (15 > 8)
        points = generator(15)

        assert len(points) == 15

        # Verify that points come from the expected set
        coords = [(int(p.x), int(p.y), int(p.z)) for p in points]

        # All coordinates should be from the expected grid
        expected_grid_points = set()
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    expected_grid_points.add((x, y, z))

        actual_points = set(coords)
        assert actual_points.issubset(expected_grid_points)

        # Should have all 8 unique points represented
        unique_coords = set(coords)
        assert len(unique_coords) == 8  # Should have exactly 8 unique points
        assert unique_coords == expected_grid_points

    def test_random_cycling_produces_different_orders_3d(self):
        """Test that random cycling produces different orders on multiple calls for 3D."""
        # Create a small 2x2x2 grid (8 total points)
        generator = GridPointGenerator3D((0, 1), (0, 1), (0, 1), step=1)

        # Generate multiple sequences and check they're different
        # (This is probabilistic, but should pass with very high probability)
        sequences = []
        for _ in range(10):
            points = generator(24)  # Request 3 full cycles
            coords = [(int(p.x), int(p.y), int(p.z)) for p in points]
            sequences.append(coords)

        # Check that not all sequences are identical
        # With random shuffling, it's extremely unlikely they'd all be the same
        unique_sequences = set(tuple(seq) for seq in sequences)
        assert (
            len(unique_sequences) > 1
        ), "Random cycling should produce different orders"
