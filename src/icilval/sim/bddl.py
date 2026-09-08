"""BDDL as data: parse and query a task file. Pure Python, no simulator.

A BDDL file is one s-expression, kept as nested lists of string atoms. The
pool builder reads a task's language and goal predicates from it; LIBERO's own
parser does the rest at simulation time.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

Node = list  # nested lists of str atoms

_TOKEN = re.compile(r"\(|\)|[^\s()]+")


def parse(text: str) -> Node:
    tokens = _TOKEN.findall(text)
    pos = 0

    def read() -> Any:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            out: list = []
            while tokens[pos] != ")":
                out.append(read())
            pos += 1
            return out
        if tok == ")":
            raise ValueError("unexpected ')'")
        return tok

    tree = read()
    if pos != len(tokens):
        raise ValueError("trailing tokens after the problem definition")
    return tree


def load(path: str | Path) -> Node:
    return parse(Path(path).read_text())


def section(tree: Node, name: str) -> Node | None:
    for item in tree:
        if isinstance(item, list) and item and item[0] == name:
            return item
    return None


def language(tree: Node) -> str:
    sec = section(tree, ":language")
    return " ".join(sec[1:]) if sec else ""


def predicates(tree: Node, name: str) -> list[list[str]]:
    sec = section(tree, name)
    if not sec:
        return []
    body = sec[1:]
    if name == ":goal" and body and isinstance(body[0], list) and body[0] and body[0][0] == "And":
        body = body[0][1:]
    return [list(p) for p in body if isinstance(p, list)]


def goal_predicates(tree: Node) -> list[list[str]]:
    return predicates(tree, ":goal")
