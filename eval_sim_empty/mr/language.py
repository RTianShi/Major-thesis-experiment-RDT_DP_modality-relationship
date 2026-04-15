"""PushCube language mutations.

The generic PushCube MR implementations live in `eval_sim.mr.language`.
Importing them here registers those mutations against the shared registry while
keeping a task-local package entrypoint.
"""

from eval_sim.mr.language import *  # noqa: F401,F403
