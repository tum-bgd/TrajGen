import pytest
from unittest.mock import Mock, patch
from collections import defaultdict
from trajgen.config import Config
from trajgen.trajectory import Trajectory
from trajgen.spatial_strategy import EqualDistributionStrategy


def _C(seed=42, **kwargs):
    """Build a Config for testing from keyword arguments."""
    cfg = Config({}, seed=seed)
    length = kwargs.pop("length", None)
    length_min = kwargs.pop("length_min", None)
    length_max = kwargs.pop("length_max", None)
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    if length is not None:
        cfg.get_next_length = length
        cfg.get_next_length_mode = "fixed for dataset"
    elif length_min is not None or length_max is not None:
        cfg.get_next_length_mode = "fixed for trajectory"
        cfg.get_next_length_distribution = "uniform"
        cfg.get_next_length_min = length_min if length_min is not None else 1
        cfg.get_next_length_max = length_max if length_max is not None else 10
    return cfg


class TestEqualDistributionStrategy:
    config = Mock()
    config.grid_rows = 3
    config.grid_cols = 3
    config.seed = 42
    config.length_min = 2
    config.length_max = 4

    def test_init(self):
        config = _C(seed=123, grid_rows=5, grid_cols=5, length_min=3, length_max=10)
        strategy = EqualDistributionStrategy(config)

        assert strategy.config == config
        assert strategy.seed == 123
        assert len(strategy.actual_visits) == 0
        assert strategy._target_visits is None
        assert strategy._lengths is None
        assert strategy._traj_order is None
        assert strategy._current_traj_idx == 0

    def test_compute_targets_and_lengths_basic(self):

        strategy = EqualDistributionStrategy(self.config)

        with (
            patch.object(self.config, "get_next_length", side_effect=[3, 2, 4]),
            patch("random.sample", return_value=[(0, 0)]),
        ):
            strategy._compute_targets_and_lengths(3)

        assert len(strategy._lengths) == 3
        assert strategy._lengths == [3, 2, 4]
        assert len(strategy._target_visits) == 9  # 3x3 grid
        assert sum(strategy._target_visits.values()) == 9  # total points
        assert strategy._traj_order == [2, 0, 1]  # sorted by length desc

    def test_compute_targets_and_lengths_equal_distribution(self):
        config = _C(grid_rows=2, grid_cols=2, length=4)
        strategy = EqualDistributionStrategy(config)

        with (
            patch("random.randrange", return_value=4),
            patch("random.sample", return_value=[]),
        ):
            strategy._compute_targets_and_lengths(2)

        # 2 trajectories of length 4 = 8 points, 4 cells = 2 visits per cell
        for cell, visits in strategy._target_visits.items():
            assert visits == 2

    def test_reset_for_dataset(self):
        config = _C(grid_rows=3, grid_cols=3, length_min=2, length_max=5)
        strategy = EqualDistributionStrategy(config)

        # Add some data to clear
        strategy.actual_visits[(0, 0)] = 5
        strategy._current_traj_idx = 3

        with patch.object(strategy, "_compute_targets_and_lengths") as mock_compute:
            strategy.reset_for_dataset(10)

        assert len(strategy.actual_visits) == 0
        assert strategy._current_traj_idx == 0
        mock_compute.assert_called_once_with(10)

    def test_find_cell_with_highest_deficit_single_max(self):
        config = _C(grid_rows=2, grid_cols=2, length_min=1, length_max=1)
        strategy = EqualDistributionStrategy(config)

        strategy._target_visits = {(0, 0): 3, (0, 1): 2, (1, 0): 2, (1, 1): 1}
        strategy.actual_visits = defaultdict(int, {(0, 0): 1, (0, 1): 1})

        # Deficits: (0,0)=2, (0,1)=1, (1,0)=2, (1,1)=1
        # Max deficit is 2, cells (0,0) and (1,0) tie
        with patch("random.choice", return_value=(0, 0)) as mock_choice:
            result = strategy._find_cell_with_highest_deficit()

        assert result == (0, 0)
        mock_choice.assert_called_once_with([(0, 0), (1, 0)])

    def test_find_cell_with_highest_deficit_all_equal(self):
        config = _C(grid_rows=2, grid_cols=2, length_min=1, length_max=1)
        strategy = EqualDistributionStrategy(config)

        strategy._target_visits = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
        strategy.actual_visits = defaultdict(int)

        with patch("random.choice", return_value=(1, 1)) as mock_choice:
            result = strategy._find_cell_with_highest_deficit()

        assert result == (1, 1)
        # All cells have deficit of 1, so all should be in the list
        mock_choice.assert_called_once_with([(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_get_valid_neighbors_corner(self):
        config = _C(grid_rows=3, grid_cols=3, length_min=1, length_max=1)
        strategy = EqualDistributionStrategy(config)

        neighbors = strategy._get_valid_neighbors(0, 0)  # top-left corner
        expected = [(0, 1), (1, 0), (1, 1)]  # 3 neighbors

        assert len(neighbors) == 3
        assert set(neighbors) == set(expected)

    def test_get_valid_neighbors_center(self):
        config = _C(grid_rows=3, grid_cols=3, length_min=1, length_max=1)
        strategy = EqualDistributionStrategy(config)

        neighbors = strategy._get_valid_neighbors(1, 1)  # center
        expected = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]

        assert len(neighbors) == 8
        assert set(neighbors) == set(expected)

    def test_get_valid_neighbors_edge(self):
        config = _C(grid_rows=3, grid_cols=3, length_min=1, length_max=1)
        strategy = EqualDistributionStrategy(config)

        neighbors = strategy._get_valid_neighbors(0, 1)  # top edge
        expected = [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]

        assert len(neighbors) == 5
        assert set(neighbors) == set(expected)

    def test_select_next_cell_by_deficit_high_deficit(self):
        config = _C(grid_rows=3, grid_cols=3, length_min=1, length_max=1)
        strategy = EqualDistributionStrategy(config)

        strategy._target_visits = {(0, 0): 1, (0, 1): 5, (1, 0): 1, (1, 1): 1}
        strategy.actual_visits = defaultdict(int, {(0, 0): 1})

        with (
            patch.object(
                strategy, "_get_valid_neighbors", return_value=[(0, 1), (1, 0), (1, 1)]
            ),
            patch("random.uniform", return_value=25),
        ):  # Should select (0,1) with weight 40
            result = strategy._select_next_cell_by_deficit((0, 0), [(0, 0)])

        assert result == (0, 1)

    def test_select_next_cell_by_deficit_backtrack_penalty(self):
        config = _C(grid_rows=3, grid_cols=3, length_min=1, length_max=1)
        strategy = EqualDistributionStrategy(config)

        strategy._target_visits = {(0, 0): 2, (0, 1): 2, (1, 0): 2}
        strategy.actual_visits = defaultdict(int)
        trajectory = [(0, 0), (0, 1)]  # came from (0,0) to (0,1)

        with patch.object(
            strategy, "_get_valid_neighbors", return_value=[(0, 0), (1, 0)]
        ):
            # (0,0) gets penalty (0.5 weight multiplier) for backtracking
            with patch("random.uniform", return_value=15):  # Should select (1,0)
                result = strategy._select_next_cell_by_deficit((0, 1), trajectory)

        assert result == (1, 0)

    def test_select_next_cell_by_deficit_no_weighted_choices(self):
        config = _C(grid_rows=3, grid_cols=3, length_min=1, length_max=1)
        strategy = EqualDistributionStrategy(config)

        with (
            patch.object(strategy, "_get_valid_neighbors", return_value=[]),
            patch("random.choice", return_value=(1, 1)) as mock_choice,
        ):
            result = strategy._select_next_cell_by_deficit((0, 0), [(0, 0)])

        assert result == (1, 1)
        mock_choice.assert_called_once_with([])

    def test_select_next_cell_by_deficit_zero_total_weight(self):
        config = _C(grid_rows=3, grid_cols=3, length_min=1, length_max=1)
        strategy = EqualDistributionStrategy(config)

        strategy._target_visits = {(0, 1): 0, (1, 0): 0}
        strategy.actual_visits = defaultdict(
            int, {(0, 1): 1, (1, 0): 1}
        )  # over-visited (deficit -1 -> weight 0.1)

        neighbors = [(0, 1), (1, 0)]
        with (
            patch.object(strategy, "_get_valid_neighbors", return_value=neighbors),
            # Total weight 0.2. Uniform(0, 0.2).
            # If we return 0.15, it skips first (0.1) and picks second.
            patch("random.uniform", return_value=0.15),
        ):
            result = strategy._select_next_cell_by_deficit((0, 0), [(0, 0)])

        assert result == (1, 0)

    def test_call_generates_trajectory(self):
        config = _C(grid_rows=2, grid_cols=2, length_min=3, length_max=3)
        strategy = EqualDistributionStrategy(config)

        # Setup state as if reset_for_dataset was called
        strategy._lengths = [3]
        strategy._traj_order = [0]
        strategy._current_traj_idx = 0
        strategy._target_visits = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}
        strategy.actual_visits = defaultdict(int)

        with (
            patch.object(
                strategy, "_find_cell_with_highest_deficit", return_value=(0, 0)
            ),
            patch.object(
                strategy, "_select_next_cell_by_deficit", side_effect=[(0, 1), (1, 0)]
            ),
        ):
            trajectory = strategy(1)

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory.ls.coords) == 3
        # Check coordinate conversion: (row, col) -> Point(col/cols, row/rows)
        assert trajectory.ls.coords[0] == (0.25, 0.25)  # (0,0) -> (0.0, 0.0)
        assert trajectory.ls.coords[1] == (0.75, 0.25)  # (0,1) -> (0.5, 0.0)
        assert trajectory.ls.coords[2] == (0.25, 0.75)  # (1,0) -> (0.0, 0.5)

    def test_call_updates_visits(self):
        config = _C(grid_rows=2, grid_cols=2, length_min=2, length_max=2)
        strategy = EqualDistributionStrategy(config)

        strategy._lengths = [2]
        strategy._traj_order = [0]
        strategy._current_traj_idx = 0
        strategy._target_visits = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
        strategy.actual_visits = defaultdict(int)

        with (
            patch.object(
                strategy, "_find_cell_with_highest_deficit", return_value=(0, 0)
            ),
            patch.object(strategy, "_select_next_cell_by_deficit", return_value=(1, 1)),
        ):
            strategy(1)

        assert strategy.actual_visits[(0, 0)] == 1
        assert strategy.actual_visits[(1, 1)] == 1
        assert strategy._current_traj_idx == 1

    def test_call_multiple_trajectories_ordering(self):
        config = _C(grid_rows=2, grid_cols=2, length_min=1, length_max=3)
        strategy = EqualDistributionStrategy(config)

        # Setup for 3 trajectories with lengths [2, 1, 3], sorted order should be [2, 0, 1]
        strategy._lengths = [2, 1, 3]
        strategy._traj_order = [2, 0, 1]  # indices sorted by length desc
        strategy._current_traj_idx = 0
        strategy._target_visits = {(0, 0): 2, (0, 1): 2, (1, 0): 1, (1, 1): 1}
        strategy.actual_visits = defaultdict(int)

        with (
            patch.object(
                strategy, "_find_cell_with_highest_deficit", return_value=(0, 0)
            ),
            patch.object(
                strategy,
                "_select_next_cell_by_deficit",
                side_effect=[(0, 1), (1, 0), (1, 1), (0, 0)],
            ),
        ):
            # First call should get trajectory with index 2 (length 3)
            traj1 = strategy(0)
            assert len(traj1.ls.coords) == 3
            assert strategy._current_traj_idx == 1

            # Second call should get trajectory with index 0 (length 2)
            traj2 = strategy(1)
            assert len(traj2.ls.coords) == 2
            assert strategy._current_traj_idx == 2

    def test_coordinate_conversion_scaling(self):
        config = _C(grid_rows=4, grid_cols=5, length_min=1, length_max=1)
        strategy = EqualDistributionStrategy(config)

        strategy._lengths = [2]
        strategy._traj_order = [0]
        strategy._current_traj_idx = 0
        strategy._target_visits = {}
        strategy.actual_visits = defaultdict(int)

        with (
            patch.object(
                strategy,
                "_find_cell_with_highest_deficit",
                side_effect=[(2, 3), (2, 4)],
            ),
            patch.object(
                strategy, "_select_next_cell_by_deficit", side_effect=[(2, 4)]
            ),
        ):
            trajectory = strategy(0)

        # Grid cell (2, 3) should convert to Point(3/5, 2/4) = Point(0.6, 0.5)
        assert len(trajectory.ls.coords) == 2
        assert trajectory.ls.coords[0] == pytest.approx(
            (3 * 0.2 + 0.1, 2 * 0.25 + 0.125)
        )  # (2,3)
