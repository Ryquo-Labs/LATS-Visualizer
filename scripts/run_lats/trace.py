"""
The trace format, and the recorder that writes it.

A trace is a complete, replayable recording of one search: the tree that was
built, and the ordered list of operations that built it. It is the contract
between the Python demo and the browser demo, and it is meant to be readable by
a person as well as by the viewer - open one in an editor and you can follow
the whole run.

Shape
-----

::

    {
      "schema":  "lats-trace/1",
      "method":  lats | react | reflexion | tot - which algorithm ran
      "task":    what was being solved, and where the reward comes from
      "config":  n, w, lambda, depth limits, seed, when LM(s) was taken
      "policy":  which sampler produced the actions, how, and what it cost
      "result":  solved / best reward / best path / why it stopped
      "nodes":   [ every node, with the fields that never change ]
      "steps":   [ every operation, each carrying a snapshot of what changes ]
    }

``method``, ``config.evaluation`` and ``policy.sampling`` arrived in version
1.1 of the package. A trace without them was written by 1.0, which scored every
candidate before executing it and, under a real model, asked for ``n``
distinct actions in one reply; the viewer labels such traces.

The split between ``nodes`` and ``steps`` is the important part. A node's
action, observation and parent are written once. Its ``visits``, ``value`` and
``reward`` change constantly, so every step carries a full snapshot of them for
the nodes that exist at that point. That costs a few kilobytes and buys a
viewer that can jump to any step without replaying the ones before it.

Step ``op`` is one of ``init``, ``selection``, ``expansion``, ``evaluation``,
``simulation``, ``backpropagation``, ``reflection``, ``result`` - the six
operations of the paper, plus a bookend at each end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import Config, Node
from .tasks.base import Task

SCHEMA = "lats-trace/1"


class TraceRecorder:
    """Collects a trace while :class:`~run_lats.search.LATS` runs."""

    def __init__(
        self, task: Task, config: Config, policy: Any, method: str = "lats"
    ) -> None:
        self.task = task
        self.config = config
        self.policy = policy
        self.method = method
        self.nodes: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self._by_id: dict[int, Node] = {}

    # -- called by the search ----------------------------------------------

    def register(self, node: Node) -> None:
        """Record a node's immutable half, the moment it is created."""
        self._by_id[node.id] = node
        self.nodes.append(
            {
                "id": node.id,
                "parent": node.parent.id if node.parent else None,
                "depth": node.depth,
                "label": node.action.label if node.action else "root",
                "action": node.action.text if node.action else None,
                "observation": node.observation.text if node.observation else None,
                "detail": node.observation.detail if node.observation else {},
                "terminal": node.terminal,
                # The step index this node first appears at, so the viewer can
                # grow the tree in time with the narration.
                "created_at": len(self.steps),
            }
        )

    def emit(
        self,
        op: str,
        *,
        iteration: int,
        title: str,
        summary: str,
        detail: dict[str, Any],
        focus: list[int],
        path: list[int],
    ) -> None:
        self.steps.append(
            {
                "index": len(self.steps),
                "op": op,
                "iteration": iteration,
                "title": title,
                "summary": summary,
                "detail": detail,
                # Nodes this step is about, and the root-to-leaf path it walked.
                "focus": focus,
                "path": path,
                "tokens": self.policy.info()["tokens"],
                "state": self._snapshot(),
            }
        )

    def _snapshot(self) -> dict[str, dict[str, Any]]:
        """Everything about the tree that changes as the search runs.

        ``value`` is ``null`` for a node that has not been graded yet - which
        is every new child between its expansion and its evaluation - and for
        every node of a method that keeps no value at all.
        """
        out: dict[str, dict[str, Any]] = {}
        for node_id, node in self._by_id.items():
            state: dict[str, Any] = {
                "visits": node.visits,
                "value": _r(node.value),
                "reward": _r(node.reward),
                "lm": _r(node.lm_score),
                "sc": _r(node.sc_score),
                "prior": _r(node.prior),
                "terminal": node.terminal,
                "reflected": node.reflection is not None,
            }
            if self.method == "tot":
                state["pruned"] = node.pruned
            out[str(node_id)] = state
        return out

    # -- output -------------------------------------------------------------

    def to_dict(self, result: dict[str, Any], *, name: str) -> dict[str, Any]:
        from . import __version__

        return {
            "schema": SCHEMA,
            "name": name,
            "method": self.method,
            "generated_by": {"package": "run_lats", "version": __version__},
            "task": self.task.brief(),
            "config": self.config.to_dict(self.method),
            "policy": self.policy.info(),
            "result": result,
            "nodes": self.nodes,
            "steps": self.steps,
        }

    def write(self, path: Path, result: dict[str, Any], *, name: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(result, name=name)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 4)
