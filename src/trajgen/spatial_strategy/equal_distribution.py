from trajgen.spatial_strategy.requirements_helpers import bbox_requirements
from ..trajectory import Trajectory
from shapely.geometry import LineString
from ..config import Config
import random
from collections import defaultdict
from shapely.geometry import Point


class EqualDistributionStrategy:
    def __init__(self, config: Config):
        self.config = config
        self.seed = config.seed
        random.seed(self.seed)
        self.actual_visits: dict[tuple[int, int], int] = defaultdict(int)
        self._target_visits: dict[tuple[int, int], int] = None
        self._lengths: list[int] = None
        self._traj_order: list[int] = None
        self._current_traj_idx = 0

    def _compute_targets_and_lengths(self, num_trajectories: int):
        """Compute targets once per dataset (called in reset)."""
        total_cells = self.config.grid_rows * self.config.grid_cols
        self._lengths = [self.config.get_next_length() for _ in range(num_trajectories)]
        total_points = sum(self._lengths)
        base_visits = total_points // total_cells
        extra_visits = total_points % total_cells
        all_cells = [
            (i, j)
            for i in range(self.config.grid_rows)
            for j in range(self.config.grid_cols)
        ]
        extra_cells = (
            []
            if extra_visits == 0
            else random.sample(all_cells, min(extra_visits, total_cells))
        )
        # print(f"extra_cells: {extra_cells}, length of all cells:
        # {len(all_cells)}, base_visits: {base_visits}, extra_visits: {extra_visits}")
        self._target_visits = {}
        for cell in all_cells:
            self._target_visits[cell] = base_visits + (1 if cell in extra_cells else 0)
        # Sort traj indices by length descending
        self._traj_order = list(range(num_trajectories))
        self._traj_order.sort(key=lambda x: self._lengths[x], reverse=True)
        self._current_traj_idx = 0

    def reset_for_dataset(self, num_trajectories: int):
        """Reset state for new dataset generation."""
        self.actual_visits.clear()
        self._current_traj_idx = 0
        self._compute_targets_and_lengths(num_trajectories)

    def __call__(self, id: int) -> Trajectory:
        """
        Generate one balanced grid trajectory.
        Expects reset_for_dataset(num_trajectories) called first on dataset gen.
        Maps integer grid (row,col) to float Point(x,y), e.g., x=col, y=row.
        """
        # Get traj index in sorted order
        if self._traj_order is None:
            n = getattr(self.config, "num_trajectories", 1)
            self.reset_for_dataset(max(n, 1))
        traj_idx = self._traj_order[self._current_traj_idx % len(self._traj_order)]
        self._current_traj_idx += 1
        target_length = self._lengths[traj_idx]

        start_cell = self._find_cell_with_highest_deficit()
        traj_grid = [start_cell]
        current_cell = start_cell
        for _ in range(target_length - 1):
            next_cell = self._select_next_cell_by_deficit(current_cell, traj_grid)
            traj_grid.append(next_cell)
            current_cell = next_cell

        # Update visits (after generation, as in original)
        for point in traj_grid:
            self.actual_visits[point] += 1

        # Convert to Points: assume x=col (horizontal), y=row (vertical), scaled to [0,1]
        scale_x = 1.0 / self.config.grid_cols
        scale_y = 1.0 / self.config.grid_rows

        return Trajectory(
            id,
            LineString(
                [
                    Point(col * scale_x + 0.5 * scale_x, row * scale_y + 0.5 * scale_y)
                    for row, col in traj_grid
                ]
            ),
        )

    def _find_cell_with_highest_deficit(self):
        max_deficit = -1
        best_cells = []
        for i in range(self.config.grid_rows):
            for j in range(self.config.grid_cols):
                cell = (i, j)
                deficit = self._target_visits[cell] - self.actual_visits[cell]
                if deficit > max_deficit:
                    max_deficit = deficit
                    best_cells = [cell]
                elif deficit == max_deficit:
                    best_cells.append(cell)
        return random.choice(best_cells)

    def _select_next_cell_by_deficit(self, current_cell, trajectory):
        neighbors = self._get_valid_neighbors(current_cell[0], current_cell[1])
        weighted_choices = []
        for neigh in neighbors:
            deficit = self._target_visits[neigh] - self.actual_visits[neigh]
            if deficit > 0:
                weight = deficit * 10
            elif deficit == 0:
                weight = 1
            else:
                weight = 0.1
            if len(trajectory) > 1 and neigh == trajectory[-2]:
                weight *= 0.5
            weighted_choices.append((neigh, weight))
        if not weighted_choices:
            print("No valid neighbors with weights, choosing randomly.")
            return random.choice(neighbors)
        total_weight = sum(w for _, w in weighted_choices)
        if total_weight == 0:
            return random.choice(neighbors)
        rand_val = random.uniform(0, total_weight)
        cum_weight = 0
        for neigh, weight in weighted_choices:
            cum_weight += weight
            if rand_val <= cum_weight:
                return neigh
        return weighted_choices[-1][0]

    def _get_valid_neighbors(self, row, col):
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.config.grid_rows and 0 <= nc < self.config.grid_cols:
                    neighbors.append((nr, nc))
        return neighbors

    @staticmethod
    def get_requirements(spatial_dim: str = "2D") -> dict:
        return {
            "get_next_grid_rows": {
                "short_name": "Grid Rows",
                "type": "get_int_function",
                "default": 10,
                "description": "Number of rows in the spatial grid.",
                "optional": False,
            },
            "get_next_grid_cols": {
                "short_name": "Grid Columns",
                "type": "get_int_function",
                "default": 10,
                "description": "Number of columns in the spatial grid.",
                "optional": False,
            },
            "get_next_length": {
                "short_name": "Trajectory Length",
                "type": "get_int_function",
                "default": 10,
                "description": "Number of points per trajectory.",
                "optional": False,
            },
            **bbox_requirements(spatial_dim),
        }
