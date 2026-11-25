"""
The data types shared by the algorithm, the environments and the trace writer.

Nothing in here knows how to search. ``search.py`` owns the six LATS
operations; the environments in ``tasks/`` own the world; this module is the
vocabulary they use to talk to each other.

The split that matters
----------------------
An :class:`Action` is what the *agent* decided to do. An :class:`Observation`
is what the *environment* said back. LATS's central claim is that the value of
a node is computed **after** the observation exists - so keeping the two apart
in the type system is not pedantry, it is the point.

The same split runs through the scores. A :class:`Proposal` carries the
model's ``prior``: its grade of its own idea, given before anything ran. A
:class:`Grade` is the value function's verdict on the state that came back,
written with the observation in hand. Only the second is ``LM(s)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    """One move by the agent, in the task's own vocabulary.

    ``label`` is what the tree visualiser prints inside a node, so keep it
    short. ``text`` is the full thing the agent "said". ``payload`` is the
    machine-readable part the environment actually consumes - the code string,
    the arithmetic step, the search query.
    """

    label: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """What the environment returned after an action was executed.

    ``reward`` is the *objective* signal - tests passed, expression evaluated,
    answer matched. It is ``None`` for every non-terminal state, which is
    exactly why LATS needs a learned value function in the first place.
    """

    text: str
    detail: dict[str, Any] = field(default_factory=dict)
    terminal: bool = False
    reward: float | None = None


@dataclass(frozen=True)
class Proposal:
    """One candidate action sampled from the policy.

    ``prior`` is the model grading its own idea *before* the environment has
    had any say. It is **not** ``LM(s)``: LATS computes the value after the
    observation exists (see :class:`Grade`). The prior is kept for two reasons
    only - the viewer shows it beside the real grade, so the effect of the
    observation is visible, and it is the whole value function of the
    Tree-of-Thoughts baseline and of the ``--evaluate-before`` ablation.

    ``signature`` is what two samples must share to count as *agreeing* - it
    is what the self-consistency term ``SC(s)`` is computed over.
    """

    action: Action
    prior: float
    rationale: str = ""
    signature: str = ""


@dataclass(frozen=True)
class Grade:
    """``LM(s)``: the value function's verdict on a state it has seen.

    Produced by ``policy.evaluate`` *after* the environment has executed the
    action, from the trajectory with the observation in it. ``why`` is the
    grader's one-line reason, which the viewer prints under the score.
    """

    score: float
    why: str = ""


@dataclass
class Node:
    """One state in the search tree.

    ``visits`` starts at 1, not 0. That is the paper's initialisation, and it
    keeps the exploration term ``sqrt(ln N(p) / N(s))`` finite for a node that
    has never been backed up through.
    """

    id: int
    depth: int
    parent: "Node | None" = None
    action: Action | None = None
    observation: Observation | None = None

    #: Environment state after ``action`` was applied. Owned by the task.
    data: dict[str, Any] = field(default_factory=dict)

    visits: int = 1
    #: ``None`` until the node has been evaluated, and for good under a method
    #: with no value function at all (ReAct, Reflexion).
    value: float | None = 0.0

    #: The two halves of V(s), kept separately so the demo can show the blend.
    lm_score: float | None = None
    sc_score: float | None = None
    #: The model's grade of the action before it ran. Recorded, not used -
    #: except as the value itself in the Tree-of-Thoughts baseline.
    prior: float | None = None
    #: The grader's reason for ``lm_score``.
    why: str = ""

    children: list["Node"] = field(default_factory=list)
    terminal: bool = False
    reward: float | None = None

    #: Reflection text written after this node's trajectory failed.
    reflection: str | None = None
    #: Dropped by a beam (Tree of Thoughts) and never reconsidered.
    pruned: bool = False

    @property
    def path(self) -> list["Node"]:
        """Root-to-here, inclusive."""
        out: list[Node] = []
        node: Node | None = self
        while node is not None:
            out.append(node)
            node = node.parent
        return list(reversed(out))

    def descendants(self):
        yield self
        for child in self.children:
            yield from child.descendants()


@dataclass
class Config:
    """Search hyper-parameters.

    The defaults are tuned for a tree a human can read, not for a benchmark
    number. Zhou et al. use ``n = 5`` and 30-50 trajectories; ``n = 3`` and
    eight iterations fit on a screen and still show every operation. Per-task
    overrides for ``lam`` and ``simulate`` come from the task itself - see
    :meth:`run_lats.tasks.base.Task.defaults`.
    """

    #: Children sampled per expansion. Paper: 5.
    n: int = 3
    #: Exploration weight in UCT. Paper: 1.
    w: float = 1.0
    #: Blend between LLM self-evaluation and self-consistency. Paper: 0.5 for
    #: HotPotQA and Game of 24, 0.8 for programming and WebShop.
    lam: float = 0.5
    #: Search iterations, one selection-to-backpropagation cycle each.
    iterations: int = 8
    #: Hard cap on tree depth, so a runaway policy still terminates.
    max_depth: int = 4
    #: Run the simulation step. False reproduces the paper's programming
    #: setting, where every node is already a complete program.
    simulate: bool = True
    #: Run the reflection step at all. Turn it off for the ablation.
    reflect: bool = True
    #: Backpropagated rewards strictly below this trigger a reflection.
    reflect_threshold: float = 1.0
    #: Stop as soon as a trajectory reaches this reward.
    solved_at: float = 1.0
    #: Seed for the mock policy. Same seed, same trace, byte for byte.
    seed: int = 7
    #: When ``LM(s)`` is taken. ``"after"`` is LATS: a separate grading call,
    #: made once the environment has replied, that reads the reply.
    #: ``"before"`` uses the model's own grade of the proposal instead - the
    #: Tree-of-Thoughts ordering, kept as an ablation. It is also exactly what
    #: version 1.0 of this package did, so it reproduces those traces.
    evaluation: str = "after"
    #: Tree of Thoughts only: how many states survive each depth.
    beam: int = 2

    def to_dict(self, method: str = "lats") -> dict[str, Any]:
        """The trace's ``config`` block. ``beam`` appears only where it is used."""
        out: dict[str, Any] = {
            "n": self.n,
            "w": self.w,
            "lambda": self.lam,
            "iterations": self.iterations,
            "max_depth": self.max_depth,
            "simulate": self.simulate,
            "reflect": self.reflect,
            "reflect_threshold": self.reflect_threshold,
            "solved_at": self.solved_at,
            "seed": self.seed,
            "evaluation": self.evaluation,
        }
        if method == "tot":
            out["beam"] = self.beam
        return out
