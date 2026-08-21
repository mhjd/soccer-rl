"""Three-dimensional soccer environments."""

__all__ = ["AdaptiveStartCurriculum", "CylinderSoccerEnv"]


def __getattr__(name):
    """Load Gymnasium-dependent environments only when requested."""
    if name == "AdaptiveStartCurriculum":
        from .curriculum import AdaptiveStartCurriculum

        return AdaptiveStartCurriculum
    if name == "CylinderSoccerEnv":
        from .cylinder_env import CylinderSoccerEnv

        return CylinderSoccerEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
