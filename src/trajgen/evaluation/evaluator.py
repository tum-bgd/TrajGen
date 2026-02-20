import trajgen


class TrajectoryEvaluator:
    def __init__(self, measures: list[str] = None):
        self.measures = measures or []
        if measures is None:
            self.measures = ["length", "total_distance", "bbox_area"]

    def evaluate_length(self, trajectory: trajgen.trajectory.Trajectory) -> dict:
        results = dict()
        if "length" in self.measures:
            results["length"] = len(trajectory.ls.coords)
        if "total_distance" in self.measures:
            results["total_distance"] = trajectory.ls.length
        if "bbox_area" in self.measures:
            results["bbox_area"] = trajectory.ls.envelope.area
        return results


class TrajectoryComparisonEvaluator:
    def __init__(self, measures: list[str] = None):
        self.measures = measures or ["hausdorff_distance", "frechet_distance"]

    def __call__(self, traj_a: trajgen.trajectory.Trajectory, traj_b: trajgen.trajectory.Trajectory) -> dict:
        results = dict()
        if "hausdorff_distance" in self.measures:
            results["hausdorff_distance"] = traj_a.ls.hausdorff_distance(traj_b.ls)
        if "frechet_distance" in self.measures:
            results["frechet_distance"] = traj_a.ls.frechet_distance(traj_b.ls)
        return results
