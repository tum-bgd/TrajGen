from trajgen.spatial_strategy import (
    RandomWalkStrategy,
    EqualDistributionStrategy,
    FreespaceStrategy,
    OsmSamplingStrategy,
)
from trajgen.combined_strategy import PhysicsInformedCombinedStrategy
from trajgen.temporal_strategy import (
    ConstantTemporalStrategy,
    VelocityTemporalStrategy,
    AccelerationTemporalStrategy,
    VariableTimeStepsTemporalStrategy,
)
from trajgen.resampling_strategy import (
    ConstantLengthResampling,
    ConstantTemporalStepResampling,
    ConstantSpatialStepResampling,
    NoiseResampling,
)

ALL_SPATIAL_METHODS = {
    "Random Walk": RandomWalkStrategy,
    # "Constrained Random Walk",
    "Equal Distribution": EqualDistributionStrategy,
    # "Polynomial Curves",
    "Freespace": FreespaceStrategy,
    "OSM Sampling": OsmSamplingStrategy,
}

ALL_COMBINED_METHODS = {"Physics Informed": PhysicsInformedCombinedStrategy}
ALL_TEMPORAL_METHODS = {
    "None": None,
    "Constant Time": ConstantTemporalStrategy,
    "Velocity": VelocityTemporalStrategy,
    "Acceleration": AccelerationTemporalStrategy,
    "Time Steps": VariableTimeStepsTemporalStrategy,
}


ALL_RESAMPLING_METHODS = {
    "None": None,
    "Constant Length": ConstantLengthResampling,
    "Constant Temporal Step": ConstantTemporalStepResampling,
    "Constant Spatial Step": ConstantSpatialStepResampling,
    "Noise": NoiseResampling,
}

ALL_METHODS = {
    **ALL_SPATIAL_METHODS,
    **ALL_COMBINED_METHODS,
    **ALL_TEMPORAL_METHODS,
    **ALL_RESAMPLING_METHODS,
}

ALL_METHOD_DESCRIPTIONS = {
    "Random Walk": "Generates trajectories through random movement steps from point to point.",
    "Constrained Random Walk": "Random walk with constraints to prevent going outside defined boundaries.",
    "Equal Distribution": "Distributes points equally across a discrete grid, ensuring balanced spatial coverage.",
    "Polynomial Curves": "Creates smooth trajectories using polynomial mathematical functions.",
    "Freespace": "Generates trajectories in open/free space without obstacles.",
    "OSM Sampling": "Samples trajectories from real-world OpenStreetMap road networks.",
    "Physics Informed": "Uses physics principles (velocity, acceleration) to create realistic movement patterns.",
    "Constant Time": "Assigns the same time value to every point in the trajectory.",
    "Velocity": "Derives timestamps from spatial distances and a velocity value.",
    "Acceleration": "Derives timestamps using initial velocity and acceleration.",
    "Time Steps": "Assigns timestamps using a fixed or variable time interval between points.",
    "Noise": "Adds spatial noise (random or orthogonal) to trajectory points.",
}
