"""The architecture side of a field's demonstration view.

`icilval.demoview` keeps the withheld arrays out of the dict a policy is handed. That is the
procedural half. This is the architectural half: an architecture used by a field that withholds a
channel must have **no input for it at all**, so the guarantee does not rest on the loader alone.

The Behavior Prompting architectures already carry the switches - `ignore_prompt_obs`,
`ignore_prompt_proprio`, `ignore_prompt_action` on the prompt tokenizer - and
`model/fingerprint.py` pins every value of the template that is not in `model.mutable_keys`. So a
submission cannot flip one: the weights would be checked against a template that says the channel
is ignored, and a model built to read it would not fit.

That makes the claim a property of the graph rather than of a code path, which is the only part
of the video-only rule that survives someone rewriting the loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: demonstration channel -> the template flag that must be true when a field withholds it.
PROMPT_SWITCHES = {
    "video": "ignore_prompt_obs",
    "proprio": "ignore_prompt_proprio",
    "actions": "ignore_prompt_action",
}


def template_path(arch_dir: str | Path, name: str, kind: str = "cfg") -> Path:
    return Path(arch_dir) / f"{name}.{kind}.json"


def exists(arch_dir: str | Path, name: str) -> bool:
    return (
        template_path(arch_dir, name).exists() and template_path(arch_dir, name, "tensors").exists()
    )


def switches(cfg: Any) -> dict[str, bool]:
    """Every `ignore_prompt_*` flag in a template, wherever it sits in the config tree.

    Walked rather than looked up at a fixed path: the flags live on whatever prompt tokenizer an
    architecture uses, and a new architecture may nest it differently.
    """
    found: dict[str, bool] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in PROMPT_SWITCHES.values() and isinstance(value, bool):
                    # Any one of them left on is enough to read the channel, so a repeated flag
                    # is only ignored when every copy of it ignores the channel.
                    found[key] = found.get(key, True) and value
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(cfg)
    return found


def check_view(cfg: Any, withheld: tuple[str, ...] | list[str]) -> list[str]:
    """Everything an architecture would still read that its field withholds."""
    flags = switches(cfg)
    problems: list[str] = []
    for channel in withheld:
        switch = PROMPT_SWITCHES.get(channel)
        if switch is None:
            continue  # a channel with no architectural switch: the loader is the only guard
        if switch not in flags:
            problems.append(f"{channel}: the template has no {switch}")
        elif not flags[switch]:
            problems.append(f"{channel}: withheld by the field, but {switch} is false")
    return problems


def check_spec(spec: Any, arch_dir: str | Path) -> list[str]:
    """Every field's architectures against what that field withholds.

    Separate from `validate_spec`, which must pass with no `arch/` directory at all so CI and the
    dashboard can check the contract. This is the validator host's check, behind
    `icilval spec validate --strict`.
    """
    problems: list[str] = []
    for track in spec.tracks:
        withheld = tuple(spec.withheld(track))
        for name in spec.architectures(track):
            if not exists(arch_dir, name):
                problems.append(f"{track}: architecture {name} has no template in {arch_dir}")
                continue
            if not withheld:
                continue
            cfg = json.loads(template_path(arch_dir, name).read_text())
            problems.extend(f"{track}/{name}: {p}" for p in check_view(cfg, withheld))
    return problems
