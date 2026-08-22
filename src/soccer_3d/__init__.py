"""Three-dimensional soccer environments."""

__all__ = [
    "AdaptiveStartCurriculum",
    "CylinderSoccerEnv",
    "G1SoccerEnv",
]


def __getattr__(name):
    """Load Gymnasium-dependent environments only when requested."""
    if name == "AdaptiveStartCurriculum":
        from .curriculum import AdaptiveStartCurriculum

        return AdaptiveStartCurriculum
    if name == "CylinderSoccerEnv":
        from .cylinder_env import CylinderSoccerEnv

        return CylinderSoccerEnv
    if name == "G1SoccerEnv":
        from .g1_soccer_env import G1SoccerEnv

        return G1SoccerEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
