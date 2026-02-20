import pytest  # noqa: F401

from unittest.mock import Mock
import trajgen
from trajgen.evaluation.evaluator import TrajectoryEvaluator, TrajectoryComparisonEvaluator


@pytest.fixture
def mock_trajectory():
    """Mock Trajectory with Shapely LineString."""
    traj = Mock(spec=trajgen.trajectory.Trajectory)
    mock_ls = Mock()
    mock_ls.coords = [(0, 0), (1, 1), (2, 2)]  # 3 points
    mock_ls.length = 2.828  # sqrt(2)*2
    mock_ls.envelope = Mock()
    mock_ls.envelope.area = 1.0
    traj.ls = mock_ls
    return traj


@pytest.fixture
def mock_trajectory_pair():
    """Two trajectories for comparison."""
    traj_a = Mock(spec=trajgen.trajectory.Trajectory)
    traj_b = Mock(spec=trajgen.trajectory.Trajectory)
    traj_a.ls = Mock()
    traj_b.ls = Mock()
    traj_a.ls.hausdorff_distance.return_value = 1.5
    traj_a.ls.frechet_distance.return_value = 2.0
    return traj_a, traj_b


class TestTrajectoryEvaluator:
    def test_default_measures(self):
        evaluator = TrajectoryEvaluator()
        assert evaluator.measures == ["length", "total_distance", "bbox_area"]

    def test_custom_measures(self):
        evaluator = TrajectoryEvaluator(["length"])
        assert evaluator.measures == ["length"]

    def test_evaluate_length_all_measures(self, mock_trajectory):
        evaluator = TrajectoryEvaluator()
        results = evaluator.evaluate_length(mock_trajectory)

        assert "length" in results
        assert results["length"] == 3
        assert "total_distance" in results
        assert results["total_distance"] == pytest.approx(2.828)
        assert "bbox_area" in results
        assert results["bbox_area"] == 1.0

    def test_evaluate_length_subset(self, mock_trajectory):
        evaluator = TrajectoryEvaluator(["length"])
        results = evaluator.evaluate_length(mock_trajectory)

        assert list(results.keys()) == ["length"]
        assert results["length"] == 3

    def test_evaluate_length_no_measures(self, mock_trajectory):
        evaluator = TrajectoryEvaluator([])
        results = evaluator.evaluate_length(mock_trajectory)
        assert results == {}


class TestTrajectoryComparisonEvaluator:
    def test_default_measures(self):
        evaluator = TrajectoryComparisonEvaluator()
        assert evaluator.measures == ["hausdorff_distance", "frechet_distance"]

    def test_custom_measures(self):
        evaluator = TrajectoryComparisonEvaluator(["hausdorff_distance"])
        assert evaluator.measures == ["hausdorff_distance"]

    def test_compare_all_measures(self, mock_trajectory_pair):
        traj_a, traj_b = mock_trajectory_pair
        evaluator = TrajectoryComparisonEvaluator()
        results = evaluator(traj_a, traj_b)

        assert "hausdorff_distance" in results
        assert results["hausdorff_distance"] == 1.5
        assert "frechet_distance" in results
        assert results["frechet_distance"] == 2.0

    def test_compare_subset(self, mock_trajectory_pair):
        traj_a, traj_b = mock_trajectory_pair
        evaluator = TrajectoryComparisonEvaluator(["hausdorff_distance"])
        results = evaluator(traj_a, traj_b)

        assert list(results.keys()) == ["hausdorff_distance"]
        assert results["hausdorff_distance"] == 1.5

    def test_compare_no_measures(self, mock_trajectory_pair):
        traj_a, traj_b = mock_trajectory_pair
        evaluator = TrajectoryComparisonEvaluator([])
        results = evaluator(traj_a, traj_b)
        assert results == {'frechet_distance': 2.0, 'hausdorff_distance': 1.5}
