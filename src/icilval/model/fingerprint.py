"""Is this submission a set of BPP checkpoints of the allow-listed architectures? Decided
without torch.

A submission holds one directory per skill (`spec.model.layout`). For each skill:
- `<skill>/config.yaml` is parsed with yaml.safe_load and diffed against arch/<name>.cfg.json.
- `<skill>/model.safetensors`' header (JSON, no tensor data) is compared with
  arch/<name>.tensors.json.
Nothing here executes anything from the submission.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..spec import Spec


@dataclass
class SkillReport:
    skill: str
    architecture: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    param_count: int = 0
    config_sha256: str = ""
    model_sha256: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repo_bytes: int = 0
    skills: dict[str, SkillReport] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def param_count(self) -> int:
        return sum(r.param_count for r in self.skills.values())


def load_arch(arch_dir: str | Path, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    arch_dir = Path(arch_dir)
    cfg = json.loads((arch_dir / f"{name}.cfg.json").read_text())
    tensors = json.loads((arch_dir / f"{name}.tensors.json").read_text())
    return cfg, tensors


def read_safetensors_header(path: str | Path) -> dict[str, dict[str, Any]]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        if n > 100 * 1024 * 1024:
            raise ValueError("safetensors header is implausibly large")
        header = json.loads(fh.read(n).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


def _mutable(path: str, mutable_keys: list[str]) -> bool:
    return any(path == m or path.startswith(m + ".") for m in mutable_keys)


def diff_model_cfg(submitted: Any, template: Any, mutable_keys: list[str]) -> list[str]:
    a, b = flatten(submitted), flatten(template)
    diffs: list[str] = []
    for k in sorted(set(a) | set(b)):
        if _mutable(k, mutable_keys):
            continue
        if k not in a:
            diffs.append(f"{k}: missing (template has {b[k]!r})")
        elif k not in b:
            diffs.append(f"{k}: unexpected key")
        elif a[k] != b[k] and not (
            isinstance(a[k], (int, float))
            and isinstance(b[k], (int, float))
            and abs(float(a[k]) - float(b[k])) < 1e-9
        ):
            diffs.append(f"{k}: expected {b[k]!r}, got {a[k]!r}")
    return diffs


def targets_in(cfg: Any) -> set[str]:
    return {v for k, v in flatten(cfg).items() if k.endswith("_target_") and isinstance(v, str)}


def _numel(shape: list[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def check_skill(skill_dir: str | Path, spec: Spec, arch_dir: str | Path, skill: str) -> SkillReport:
    """One skill's directory against its architecture template."""
    from ..canon import sha256_file

    skill_dir = Path(skill_dir)
    model_spec = spec.model
    name = spec.architecture(skill)
    report = SkillReport(skill=skill, architecture=name)
    try:
        template_cfg, template_tensors = load_arch(arch_dir, name)
    except OSError as exc:
        report.errors.append(f"architecture template unavailable: {exc}")
        return report
    for required in model_spec["required_files"]:
        if not (skill_dir / required).exists():
            report.errors.append(f"{required} missing")
    if report.errors:
        return report

    # config
    cfg_path = skill_dir / "config.yaml"
    try:
        submitted = yaml.safe_load(cfg_path.read_text())
    except yaml.YAMLError as exc:
        report.errors.append(f"config.yaml: not valid YAML ({exc})")
        return report
    if not isinstance(submitted, dict):
        report.errors.append("config.yaml: expected a mapping")
        return report
    if submitted.get("architecture") != name:
        report.errors.append(
            f"config.yaml: architecture must be {name!r}, got {submitted.get('architecture')!r}"
        )
    template_targets = targets_in(template_cfg.get("model", {}))
    extra_targets = targets_in(submitted.get("model", {})) - template_targets
    for t in sorted(extra_targets):
        report.errors.append(f"config.yaml: _target_ {t!r} is not allow-listed")
    for d in diff_model_cfg(
        submitted.get("model"), template_cfg.get("model"), list(model_spec.get("mutable_keys", []))
    ):
        report.errors.append(f"config.yaml: model.{d}")
    report.config_sha256 = sha256_file(cfg_path)

    # tensors
    st_path = skill_dir / "model.safetensors"
    try:
        header = read_safetensors_header(st_path)
    except (OSError, ValueError, struct.error) as exc:
        report.errors.append(f"model.safetensors: unreadable header ({exc})")
        return report
    allowed_dtypes = set(model_spec["allowed_dtypes"])
    missing = sorted(set(template_tensors) - set(header))
    unexpected = sorted(set(header) - set(template_tensors))
    for k in missing[:20]:
        report.errors.append(f"model.safetensors: tensor {k} missing")
    for k in unexpected[:20]:
        report.errors.append(f"model.safetensors: unexpected tensor {k}")
    if len(missing) > 20 or len(unexpected) > 20:
        report.errors.append(
            f"model.safetensors: {len(missing)} missing and {len(unexpected)} unexpected tensors in total"
        )
    for k, info in header.items():
        if k not in template_tensors:
            continue
        if list(info.get("shape", [])) != list(template_tensors[k]["shape"]):
            report.errors.append(
                f"model.safetensors: {k} shape {info.get('shape')} != {template_tensors[k]['shape']}"
            )
        if info.get("dtype") not in allowed_dtypes:
            report.errors.append(f"model.safetensors: {k} dtype {info.get('dtype')} not allowed")
        report.param_count += _numel(info.get("shape", []))
    if report.param_count > int(model_spec["max_params"]):
        report.errors.append(
            f"model has {report.param_count} parameters; limit {model_spec['max_params']}"
        )
    report.model_sha256 = sha256_file(st_path)
    return report


def check_submission(
    model_dir: str | Path, spec: Spec, arch_dir: str | Path, skills: tuple[str, ...] | None = None
) -> CheckReport:
    """The whole repository: file allow-list and size, then every skill's directory."""
    report = CheckReport()
    model_dir = Path(model_dir)
    model_spec = spec.model
    allowed_ext = set(model_spec["allowed_extensions"])
    for p in sorted(model_dir.rglob("*")):
        if p.is_dir() or any(part.startswith(".") for part in p.relative_to(model_dir).parts):
            continue
        report.repo_bytes += p.stat().st_size
        if p.suffix.lower() not in allowed_ext:
            report.errors.append(f"{p.relative_to(model_dir)}: extension not allowed")
    if report.repo_bytes > int(model_spec["max_repo_bytes"]):
        report.errors.append(
            f"repository is {report.repo_bytes} bytes; limit {model_spec['max_repo_bytes']}"
        )
    for skill in skills or spec.all_skills:
        sub = model_dir / skill
        if not sub.is_dir():
            report.skills[skill] = SkillReport(
                skill, spec.architecture(skill), [f"{skill}/ missing"]
            )
            report.errors.append(f"{skill}: directory missing ({spec.model['layout']})")
            continue
        r = check_skill(sub, spec, arch_dir, skill)
        report.skills[skill] = r
        report.errors.extend(f"{skill}: {e}" for e in r.errors)
        report.warnings.extend(f"{skill}: {w}" for w in r.warnings)
    return report
