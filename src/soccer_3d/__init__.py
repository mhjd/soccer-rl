"""Three-dimensional soccer environments."""

__all__ = [
    "AdaptiveStartCurriculum",
    "CylinderSoccerEnv",
    "G1AdaptiveStartCurriculum",
    "G1HighLevelKickResidualEnv",
    "G1GeometricResidualEnv",
    "G1KickResidualEnv",
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
    if name == "G1KickResidualEnv":
        from .g1_kick_residual_env import G1KickResidualEnv

        return G1KickResidualEnv
    if name == "G1HighLevelKickResidualEnv":
        from .g1_high_level_kick_residual_env import (
            G1HighLevelKickResidualEnv,
        )

        return G1HighLevelKickResidualEnv
    if name == "G1GeometricResidualEnv":
        from .g1_geometric_residual_env import G1GeometricResidualEnv

        return G1GeometricResidualEnv
    if name == "G1AdaptiveStartCurriculum":
        from .g1_curriculum import G1AdaptiveStartCurriculum

        return G1AdaptiveStartCurriculum
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
