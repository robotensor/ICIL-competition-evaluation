"""`bpp_robotwin_v1`: the BPP network over a bimanual robot's arrays.

The network and the observation contract are the ones `bpp_libero_v1` already carries - the two
templates are byte-identical apart from their name - and everything that differs is the
conversion in `conversion.py`, which turns `frames_<camera>`, `qpos`, `endpose` and
`gripper_joints` into `agentview_rgb`, `eye_in_hand_rgb`, `ee_pos`, `ee_ori` and
`gripper_states`, and turns the network's 10-dimensional delta back into an end-effector command.
That is why it is a separate architecture id rather than a flag: in this competition an
architecture denotes the input contract as well as the network, and a duel's fingerprint should
record which conversion it ran.

Nothing here imports the benchmark, and nothing outside `policy.py` imports torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..architectures import register

#: The architecture in `arch/` this package implements.
ARCHITECTURE = "bpp_robotwin_v1"


def make_policy(
    model_dir: str | Path, arch_dir: str | Path, spec: Any, skill: str, device: str = "cuda"
) -> Any:
    from .policy import BPPRoboTwinPolicy

    return BPPRoboTwinPolicy(model_dir, arch_dir, spec, skill, device=device)


register(ARCHITECTURE, make_policy)
