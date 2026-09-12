"""A measurement published beside the competition, deliberately outside every ladder.

Some things are worth publishing and are not contests: a benchmark's own oracle, a released
checkpoint measured on a new benchmark, a baseline run to see whether a field is reachable at all.
The benchmark repository already draws this line - `robotwin-icil report --reference` prints other
runs beside a score as "context, never a second score".

**Where a record lives is itself a claim.** `tracks/<field>/index-NNNN.jsonl` is that field's
signed, append-only ladder: a line in it says the validator ran this, under that field's contract,
for its crown. A benchmark run produced elsewhere is none of those things, and filing it there
would lend it exactly the provenance it has not earned - invisibly, because the JSON would look
like every other line. Worse, the orchestrator stamps `prompt.view` and the event's
`demonstration` block *from the field*, so a run handed the actions would be published under a
field that says it withholds them: machine-readably false.

So an exhibit goes to `references/<id>.json` instead - signed, content-addressed clips in the same
`media/` tree, and in no index at all. Nothing that reads a ladder can see it, by construction
rather than by a filter somebody has to remember.

What every exhibit must carry: what the policy was **shown**, whether that is what the field shows,
and whether the model would even be admissible. A number is most likely to be misread exactly when
it was measured under conditions the field forbids, so it is stated rather than left to be assumed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .canon import canonical_json

SCHEMA = 1
KIND = "benchmark_reference"

#: Where exhibits live in the store. Deliberately a sibling of `tracks/`, never inside one.
ROOT = "references"

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


class ReferenceError(ValueError):
    """An exhibit that would misrepresent what it measured."""


def exhibit(
    *,
    reference_id: str,
    headline: str,
    not_a_competition_score: str,
    benchmark: dict[str, Any],
    protocol: dict[str, Any],
    demonstration_shown: dict[str, Any],
    subject: dict[str, Any],
    results: dict[str, Any],
    ceiling: dict[str, Any] | None = None,
    published_at: str,
    media: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One exhibit, as it is published.

    `ladder` and `track` are literals, not parameters: an exhibit is on no ladder and belongs to
    no field. Making them arguments would be inviting the one mistake this module exists to stop.
    """
    if not ID_RE.match(reference_id):
        raise ReferenceError(f"reference_id {reference_id!r} must be a short lower-case slug")
    for field, value in (
        ("headline", headline),
        ("not_a_competition_score", not_a_competition_score),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ReferenceError(f"{field} must say something; it is what a reader sees first")
    if "view" not in demonstration_shown:
        raise ReferenceError(
            "demonstration_shown.view is required: an exhibit that does not say what the policy "
            "was shown is the thing this format exists to prevent"
        )
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "reference_id": reference_id,
        # Not parameters. An exhibit is on no ladder and under no field.
        "ladder": False,
        "track": None,
        "published_at": published_at,
        "headline": headline,
        "not_a_competition_score": not_a_competition_score,
        "benchmark": benchmark,
        "protocol": protocol,
        "demonstration_shown": demonstration_shown,
        "subject": subject,
        "results": results,
        **({"ceiling": ceiling} if ceiling else {}),
        "media": media or [],
    }


def write(store_root: str | Path, doc: dict[str, Any], signer: Any) -> Path:
    """Sign and write an exhibit. Same line format as an index record: canonical JSON, TAB, sig."""
    root = Path(store_root)
    out = root / ROOT / f"{doc['reference_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json(doc)
    out.write_text(f"{body}\t{signer.sign(body)}\n", encoding="utf-8")
    return out


def read(path: str | Path) -> dict[str, Any]:
    line = Path(path).read_text(encoding="utf-8").strip()
    body = line.split("\t", 1)[0]
    return json.loads(body)
