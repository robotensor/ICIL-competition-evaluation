"""Independent verification of a store: signatures, sequence, events, media, schema."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..canon import canonical_json, verify_signature
from ..spec import Spec, load_schema
from .records import media_shas, prompt_shas
from .writer import Store


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    records: int = 0
    events: int = 0
    media: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _schema_validator(schema: dict[str, Any] | None):
    if schema is None:
        return None
    try:
        import jsonschema
    except ImportError:
        return None
    return jsonschema.Draft202012Validator(schema)


def _validate(validator, ref: str, obj: Any, where: str, report: Report) -> None:
    if validator is None:
        return
    sub = {"$ref": f"#/$defs/{ref}", "$defs": validator.schema["$defs"]}
    import jsonschema

    v = jsonschema.Draft202012Validator(sub)
    for err in v.iter_errors(obj):
        report.errors.append(
            f"{where}: schema {ref}: {err.message} at {'/'.join(str(p) for p in err.path)}"
        )


def verify_store(
    root: str | Path, spec: Spec, schema: dict[str, Any] | None = None, check_schema: bool = True
) -> Report:
    report = Report()
    store = Store(root, spec)
    schema_doc = schema if schema is not None else (load_schema() if check_schema else None)
    validator = _schema_validator(schema_doc) if check_schema else None

    manifest = store.manifest()
    if manifest is None:
        report.errors.append("manifest.json missing or unreadable")
        return report
    _validate(validator, "Manifest", manifest, "manifest.json", report)
    key = str(manifest.get("validator_key", ""))
    if manifest.get("spec_fingerprint") != spec.fingerprint:
        report.warnings.append("manifest spec_fingerprint differs from the loaded spec.json")

    video_ext = str(spec.media["video"]["format"])
    prompt_ext = str(spec.media["prompt"]["format"])
    for track in manifest.get("tracks", []):
        expected_seq = 1
        last: dict[str, Any] | None = None
        part = 0
        while True:
            path = store.index_part_path(track, part)
            if not path.exists():
                break
            lines = path.read_text(encoding="utf-8").split("\n")
            for n, raw in enumerate(lines, start=1):
                if not raw.strip():
                    continue
                where = f"{path.relative_to(store.root)}:{n}"
                if "\t" not in raw:
                    report.errors.append(f"{where}: no signature")
                    continue
                body, sig = raw.split("\t", 1)
                try:
                    record = json.loads(body)
                except ValueError:
                    report.errors.append(f"{where}: unparsable record")
                    continue
                if canonical_json(record) != body:
                    report.errors.append(f"{where}: record is not canonical JSON")
                if not verify_signature(key, body, sig.strip()):
                    report.errors.append(f"{where}: bad signature")
                if record.get("seq") != expected_seq:
                    report.errors.append(
                        f"{where}: seq {record.get('seq')} expected {expected_seq}"
                    )
                expected_seq = int(record.get("seq", expected_seq)) + 1
                if store.part_of(int(record.get("seq", 1))) != part:
                    report.errors.append(f"{where}: record filed in the wrong part")
                _validate(validator, "IndexRecord", record, where, report)
                report.records += 1
                last = record

                event = store.event(track, str(record.get("event_id", "")))
                if event is None:
                    report.errors.append(f"{where}: event file missing")
                    continue
                report.events += 1
                _validate(
                    validator,
                    "DuelEvent",
                    event,
                    f"events/{track}/{record.get('event_id')}.json",
                    report,
                )
                for k in ("event_id", "kind", "block", "dethroned", "king", "challenger"):
                    if event.get(k) != record.get(k):
                        report.errors.append(f"{where}: event.{k} differs from the index record")
                for sha in media_shas(event.get("units", [])):
                    if store.has_media(sha, video_ext):
                        report.media += 1
                    else:
                        report.errors.append(f"{where}: media {sha[:12]} missing")
                for sha in prompt_shas(event.get("units", [])):
                    if store.has_media(sha, prompt_ext):
                        report.media += 1
                    else:
                        report.errors.append(f"{where}: prompt {sha[:12]} missing")
            part += 1

        head = store.head(track)
        if head is None:
            report.errors.append(f"tracks/{track}/head.json missing")
        else:
            _validate(validator, "Head", head, f"tracks/{track}/head.json", report)
            if last is not None and (
                head.get("seq") != last.get("seq") or head.get("event_id") != last.get("event_id")
            ):
                report.errors.append(f"tracks/{track}/head.json does not point at the last record")
            if last is None and head.get("seq") not in (0, None):
                report.errors.append(f"tracks/{track}/head.json claims records that do not exist")
        queue = store.queue_path(track)
        if queue.exists():
            try:
                _validate(
                    validator,
                    "QueueSnapshot",
                    json.loads(queue.read_text()),
                    f"tracks/{track}/queue.json",
                    report,
                )
            except ValueError:
                report.warnings.append(
                    f"tracks/{track}/queue.json unreadable (rewritten every cycle; may be mid-write)"
                )
    return report
