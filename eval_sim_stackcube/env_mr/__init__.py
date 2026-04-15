"""StackCube-specific environment metamorphic relations.

StackCube task-specific environment mutations are registered here.
"""

from .env import *  # noqa: F401,F403
from eval_sim.env_mr.registry import get_env, register_env
