"""Identifiers derived from the contract. Deterministic, reproducible by third parties."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .canon import sha256_hex

REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
HEX_RE = re.compile(r"^[0-9a-f]+$")


@dataclass(frozen=True)
class ModelRef:
    key: str
    repo: str
    revision: str

    @classmethod
    def make(cls, repo: str, revision: str) -> ModelRef:
        return cls(key=model_key(repo, revision), repo=repo, revision=revision)

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "repo": self.repo, "revision": self.revision}

    @classmethod
    def from_dict(cls, d: dict | None) -> ModelRef | None:
        if not d:
            return None
        return cls(key=str(d["key"]), repo=str(d["repo"]), revision=str(d["revision"]))

    @property
    def entry(self) -> str:
        return f"{self.repo}@{self.revision}"


def model_key(repo: str, revision: str) -> str:
    return sha256_hex(f"{repo}@{revision}")[:16]


def duel_id(spec_version: int, track: str, challenger: ModelRef, king: ModelRef | None) -> str:
    king_key = king.key if king else "none"
    king_rev = king.revision if king else "none"
    return sha256_hex(
        "|".join(
            [str(spec_version), track, challenger.key, challenger.revision, king_key, king_rev]
        )
    )


def event_id(kind: str, track: str, block: int, subject: str) -> str:
    """Unique per published event: the same pair may duel again in a later block."""
    return sha256_hex("|".join(["event", kind, track, str(block), subject]))


def unit_seed(duel: str, skill: str, index: int) -> int:
    return int(sha256_hex(f"{duel}|{skill}|{index}")[:8], 16)


def prompt_seed(duel: str, skill: str, index: int, attempt: int) -> int:
    """The seed of a unit's `attempt`-th prompt generation (`generation.prompt_seed`)."""
    return int(sha256_hex(f"{duel}|{skill}|{index}|{attempt}")[:8], 16)


def unit_id(code: str, index: int) -> str:
    """`<skill code>-<index>`: the code is the two-letter `skills.<skill>.code`."""
    return f"{code}-{index:03d}"


def is_repo(value: str) -> bool:
    return bool(REPO_RE.match(value)) and len(value) <= 200


def is_sha_revision(value: str) -> bool:
    return bool(HEX_RE.match(value)) and 7 <= len(value) <= 64
