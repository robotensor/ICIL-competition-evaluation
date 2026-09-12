import json
import struct

import yaml

from icilval.model.fingerprint import (
    check_skill,
    check_submission,
    diff_model_cfg,
    read_safetensors_header,
    targets_in,
)

TEMPLATE_MODEL = {
    "_target_": "pkg.Policy",
    "name": "prompt_diffusion_unet",
    "num_inference_steps": 16,
    "obs_encoder": {
        "_target_": "pkg.Encoder",
        "n_emb": 768,
        "obs_encoder": {"_target_": "pkg.Tokenizer", "pretrained": False},
    },
}
TENSORS = {
    "bpp_libero_v1": {"a.weight": ([4, 3], "F32"), "b": ([2], "F32")},
    "bpp_draw_v1": {"a.weight": ([4, 3], "F32"), "c": ([3], "F32")},
}


def write_safetensors(path, tensors):
    header = {k: {"dtype": d, "shape": s, "data_offsets": [0, 0]} for k, (s, d) in tensors.items()}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def make_arch(tmp_path):
    arch = tmp_path / "arch"
    arch.mkdir(exist_ok=True)
    for name, tensors in TENSORS.items():
        (arch / f"{name}.cfg.json").write_text(
            json.dumps({"architecture": name, "model": TEMPLATE_MODEL})
        )
        (arch / f"{name}.tensors.json").write_text(
            json.dumps({k: {"shape": s, "dtype": d} for k, (s, d) in tensors.items()})
        )
    return arch


def make_submission(spec, tmp_path, model=None, tensors=None, extra=None, skills=None):
    """One directory per spec skill (or per `skills`: skill -> architecture)."""
    root = tmp_path / "sub"
    root.mkdir(exist_ok=True)
    # A submission is to one field, not to the whole contract: its directories are that
    # field's skills. These tests use the sensorimotor field, whose templates are on disk.
    skills = skills or {s: spec.architecture(s) for s in spec.skills("sensorimotor")}
    for skill, arch in skills.items():
        d = root / skill
        d.mkdir(exist_ok=True)
        (d / "config.yaml").write_text(
            yaml.safe_dump({"architecture": arch, "model": model or TEMPLATE_MODEL})
        )
        write_safetensors(d / "model.safetensors", tensors or TENSORS[arch])
    for name, content in (extra or {}).items():
        (root / name).write_bytes(content)
    return root


def test_diff_and_targets():
    changed = json.loads(json.dumps(TEMPLATE_MODEL))
    changed["name"] = "mine"
    changed["obs_encoder"]["n_emb"] = 512
    changed["obs_encoder"]["obs_encoder"]["_target_"] = "evil.Loader"
    diffs = diff_model_cfg(changed, TEMPLATE_MODEL, ["name"])
    assert any(d.startswith("obs_encoder.n_emb") for d in diffs)
    assert any("_target_" in d for d in diffs)
    assert not any(d.startswith("name") for d in diffs)
    assert "evil.Loader" in targets_in(changed)


def test_header_and_accept(spec, tmp_path):
    arch = make_arch(tmp_path)
    sub = make_submission(spec, tmp_path)
    assert read_safetensors_header(sub / "pick_and_place" / "model.safetensors")["a.weight"][
        "shape"
    ] == [4, 3]
    report = check_submission(sub, spec, arch, spec.skills("sensorimotor"))
    assert report.ok, report.errors
    assert set(report.skills) == set(spec.skills("sensorimotor"))
    assert report.skills["pick_and_place"].param_count == 14
    assert report.skills["draw_anything"].param_count == 15
    assert report.param_count == sum(r.param_count for r in report.skills.values())
    assert all(r.model_sha256 and r.config_sha256 for r in report.skills.values())
    one = check_skill(sub / "draw_anything", spec, arch, "draw_anything")
    assert one.ok and one.architecture == "bpp_draw_v1"


def test_rejections(spec, tmp_path):
    arch = make_arch(tmp_path)
    m = json.loads(json.dumps(TEMPLATE_MODEL))
    m["obs_encoder"]["obs_encoder"]["_target_"] = "evil.Loader"
    r = check_submission(make_submission(spec, tmp_path, model=m), spec, arch)
    assert any("allow-listed" in e for e in r.errors)
    assert all(e.startswith(tuple(f"{s}:" for s in spec.all_skills)) for e in r.errors)

    r = check_submission(
        make_submission(
            spec,
            tmp_path,
            tensors={"a.weight": ([4, 4], "F32"), "b": ([2], "F32")},
            skills={"pick_and_place": "bpp_libero_v1"},
        ),
        spec,
        arch,
    )
    assert any("shape" in e for e in r.errors)
    r = check_submission(
        make_submission(spec, tmp_path, tensors={"a.weight": ([4, 3], "F64"), "b": ([2], "F32")}),
        spec,
        arch,
    )
    assert any("dtype" in e for e in r.errors)
    r = check_submission(
        make_submission(spec, tmp_path, tensors={"a.weight": ([4, 3], "F32")}), spec, arch
    )
    assert any("missing" in e for e in r.errors)
    r = check_submission(
        make_submission(spec, tmp_path, extra={"weights.ckpt": b"\x80\x04"}), spec, arch
    )
    assert any("extension" in e for e in r.errors)
    (tmp_path / "sub" / "weights.ckpt").unlink()
    (tmp_path / "sub" / "draw_anything" / "config.yaml").write_text(
        "architecture: other\nmodel: {}\n"
    )
    r = check_submission(tmp_path / "sub", spec, arch, spec.skills("sensorimotor"))
    assert any("draw_anything: config.yaml: architecture" in e for e in r.errors)
    assert r.skills["pick_and_place"].ok
    (tmp_path / "sub" / "draw_anything" / "config.yaml").unlink()
    r = check_submission(tmp_path / "sub", spec, arch, spec.skills("sensorimotor"))
    assert any("draw_anything: config.yaml missing" in e for e in r.errors)


def test_missing_skill_directory(spec, tmp_path):
    arch = make_arch(tmp_path)
    sub = make_submission(spec, tmp_path, skills={"pick_and_place": "bpp_libero_v1"})
    r = check_submission(sub, spec, arch, spec.skills("sensorimotor"))
    assert not r.ok and any("draw_anything: directory missing" in e for e in r.errors)
    assert r.skills["pick_and_place"].ok
