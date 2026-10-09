"""Public action guidance; never selects or filters a ScienceWorld action."""

FOCUS_V3_GUIDANCE = (
    "Inspect an object with `look at <object>`; inspect a container's contents with "
    "`look in <container>` (open it first if closed). `focus on <object>` commits a "
    "task-target selection and can irreversibly end the episode if wrong. Before "
    "focusing, check the task's current requested target against observed evidence. "
    "A container holding a target is not the target itself. If the target is absent "
    "or an action is unrecognized, explore or inspect; do not substitute another "
    "visible object. Open a closed exit before moving through it. An available "
    "action is not necessarily correct for the task."
)

FOCUS_V3_REMINDER = (
    "[Action guidance, not simulator observation] Inspect with `look at <object>` "
    "or `look in <container>`. Use `focus on` only for the current task target "
    "supported by observations, never its container or an unrelated visible object. "
    "Wrong focus can end the task. If the target is absent, find it first."
)
