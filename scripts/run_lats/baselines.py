"""
Three methods LATS is measured against, run on the same environments.

    ReAct              Yao et al., ICLR 2023 (arXiv:2210.03629)
    Reflexion          Shinn et al., NeurIPS 2023 (arXiv:2303.11366)
    Tree of Thoughts   Yao et al., NeurIPS 2023 (arXiv:2305.10601)

Each one is the smallest faithful reading of its paper that runs here, and each
writes the same trace format as :class:`~run_lats.search.LATS`, with LATS's
names for the operations it shares. That is the point of them. Open any two
side by side in the viewer and each method is LATS with some of its six
operations taken away:

    method      selection   expansion   evaluation       simulation  backprop  reflection
    ReAct          -          n = 1        -                -           -          -
    Reflexion      -          n = 1        -                -           -         yes
    ToT          beam           n       before the reply    -           -          -
    LATS          UCT           n       after the reply    yes         yes        yes

They share the task, the policy and the depth limit with the LATS run they are
compared against, so a difference between two traces is a difference between
the algorithms. Like LATS, none of them is a re-implementation of the authors'
code; all three are written from the papers.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .search import NullRecorder
from .tasks.base import Task
from .types import Config, Node, Proposal


class _Method:
    """What the three share: a root, a way to take one step, and the bookends."""

    method = "?"
    title = "?"
    #: How the method starts, for the init step. ``{reward}`` is filled in.
    blurb = ""

    def __init__(
        self, task: Task, config: Config, policy: Any, recorder: Any | None = None
    ) -> None:
        self.task = task
        self.cfg = self.configure(config)
        self.policy = policy
        self.rec = recorder or NullRecorder()
        self._next_id = 0
        self.reflections: list[str] = []
        # No value function means no value at all, not a value of zero.
        self.root = self._node(depth=0, data=task.root_data(), value=None)
        self.rec.register(self.root)

    @classmethod
    def configure(cls, config: Config) -> Config:
        """The config this method actually runs with, which is what the trace
        records - ReAct is not run with five samples and then reported so."""
        return config

    # -- plumbing -----------------------------------------------------------

    def _node(self, **kw: Any) -> Node:
        node = Node(id=self._next_id, **kw)
        self._next_id += 1
        return node

    def _act(self, node: Node, proposal: Proposal) -> tuple[Node | None, dict]:
        """Execute one proposal from ``node`` in the real environment."""
        try:
            data, obs = self.task.step(node.data, proposal.action)
        except Exception as exc:  # a policy may propose something illegal
            return None, {"label": proposal.action.label, "rejected": str(exc), "id": None}
        child = self._node(
            depth=node.depth + 1,
            parent=node,
            action=proposal.action,
            observation=obs,
            data=data,
            terminal=obs.terminal or node.depth + 1 >= self.cfg.max_depth,
            reward=obs.reward,
            value=None,
        )
        child.prior = proposal.prior
        node.children.append(child)
        self.rec.register(child)
        return child, {
            "id": child.id,
            "label": proposal.action.label,
            "text": proposal.action.text,
            "observation": obs.text,
            "detail": obs.detail,
            "terminal": child.terminal,
            "reward": obs.reward,
            "samples": 1,
            "prior": round(proposal.prior, 4),
            "rejected": None,
        }

    def _earned(self, node: Node) -> float:
        """The reward a trajectory ending here has earned: the environment's,
        if it said the episode was over or every state is a whole answer anyway
        (a complete program), and zero if only the depth limit stopped it."""
        if node.observation is None:
            return 0.0
        finished = node.observation.terminal or self.task.complete_states
        return (node.reward or 0.0) if finished else 0.0

    def _start(self) -> None:
        self.rec.emit(
            "init",
            iteration=0,
            title="Root",
            summary=f"{self.task.title}. {self.blurb.format(reward=self.task.reward_desc)}",
            detail={"task": self.task.brief(), "config": self.cfg.to_dict(self.method)},
            focus=[self.root.id],
            path=[self.root.id],
        )

    def _finish(
        self, best: Node | None, iterations: int, stop: str, **extra: Any
    ) -> dict[str, Any]:
        nodes = list(self.root.descendants())
        reward = (best.reward or 0.0) if best else None
        result = {
            "solved": bool(best and (best.reward or 0.0) >= self.cfg.solved_at),
            "best_reward": None if reward is None else round(reward, 4),
            "best_node": best.id if best else None,
            "best_path": [n.id for n in best.path] if best else [],
            "nodes": len(nodes),
            "iterations_run": iterations,
            "stopped_because": stop,
            "reflections": list(self.reflections),
            **extra,
        }
        self.rec.emit(
            "result",
            iteration=iterations,
            title="Search over",
            summary=(
                f"Solved: reward {reward:g} at node {best.id}."
                if result["solved"] and best and reward is not None
                else f"Stopped without solving - {stop}. Reward "
                + ("none" if reward is None else f"{reward:g}")
                + "."
            ),
            detail=result,
            focus=result["best_path"],
            path=result["best_path"],
        )
        return result


# ---------------------------------------------------------------------------
# ReAct, and Reflexion on top of it
# ---------------------------------------------------------------------------


class ReAct(_Method):
    """One trajectory, start to finish, with no second opinion anywhere.

    At every step the policy is sampled once, the action is executed, and the
    observation is appended to the trajectory. Nothing is compared, nothing is
    graded, nothing is undone. It is LATS with ``n = 1``, one iteration, and
    no value function - the baseline the whole family of search methods is
    built to beat.
    """

    method = "react"
    title = "ReAct"
    blurb = (
        "ReAct samples one action at a time and commits to each; reward is "
        "{reward}."
    )

    @classmethod
    def configure(cls, config: Config) -> Config:
        return replace(config, n=1, iterations=1, evaluation="none")

    def run(self) -> dict[str, Any]:
        self._start()
        leaf = self._episode(iteration=1)
        leaf.reward = self._earned(leaf)
        stop = (
            "the episode ended"
            if leaf.observation is not None and leaf.observation.terminal
            else "the depth limit ended the episode"
        )
        return self._finish(leaf, 1, stop)

    def _episode(self, iteration: int) -> Node:
        """Act until the environment says stop or the depth limit does."""
        node = self.root
        while not node.terminal and node.depth < self.cfg.max_depth:
            proposals = self.policy.propose(self.task, node.data, 1, self.reflections)
            child, record = (
                self._act(node, proposals[0]) if proposals else (None, None)
            )
            self.rec.emit(
                "expansion",
                iteration=iteration,
                title="Act",
                summary=(
                    f"One sample, and it is taken: {child.action.label}."
                    if child is not None and child.action
                    else "The policy returned nothing usable, so the episode ends here."
                ),
                detail={
                    "parent": node.id,
                    "n": 1,
                    "children": [record] if record else [],
                    "reflections_in_context": list(self.reflections),
                },
                focus=[(child or node).id],
                path=[n.id for n in (child or node).path],
            )
            if child is None:
                break
            node = child
        return node


class Reflexion(ReAct):
    """ReAct, repeated, with a written note carried from each failure.

    Every trial starts again from the root. When one fails, the agent writes a
    reflection on why, and the note goes into the context of every later
    trial - the same mechanism, and in this package the same note, as LATS's
    sixth operation. What Reflexion does not have is the tree: one trajectory
    per trial, no comparison between candidates, and no value to carry what was
    learned about a state from one trial to the next.
    """

    method = "reflexion"
    title = "Reflexion"
    blurb = (
        "Reflexion runs ReAct trials from the root, with a note written after "
        "each failure; reward is {reward}."
    )

    @classmethod
    def configure(cls, config: Config) -> Config:
        # ``iterations`` is the number of trials; LATS's budget, unchanged.
        return replace(config, n=1, evaluation="none")

    def run(self) -> dict[str, Any]:
        self._start()
        stop = "the trial budget ran out"
        best: Node | None = None
        trial = 0
        for trial in range(1, self.cfg.iterations + 1):
            leaf = self._episode(iteration=trial)
            leaf.reward = reward = self._earned(leaf)
            if best is None or reward >= (best.reward or 0.0):
                best = leaf
            if reward >= self.cfg.solved_at:
                stop = "a trial reached the target reward"
                break
            if not self.cfg.reflect or reward >= self.cfg.reflect_threshold:
                continue
            text = self.task.reflection_for(leaf.data, leaf.observation)
            leaf.reflection = text
            self.reflections.append(text)
            self.rec.emit(
                "reflection",
                iteration=trial,
                title="Reflection",
                summary=(
                    f"Trial {trial} failed, so the agent writes down why. The "
                    "note joins the context of every later trial."
                ),
                detail={
                    "node": leaf.id,
                    "text": text.split("|")[0].strip(),
                    "trajectory": [n.id for n in leaf.path],
                    "reward": reward,
                    "total_notes": len(self.reflections),
                },
                focus=[leaf.id],
                path=[n.id for n in leaf.path],
            )
        return self._finish(best, trial, stop)


# ---------------------------------------------------------------------------
# Tree of Thoughts
# ---------------------------------------------------------------------------


class TreeOfThoughts(_Method):
    """Breadth-first search, pruned to a beam by a value taken before the reply.

    Depth by depth, every state in the beam is expanded with ``n`` samples, and
    every new state is valued by the policy's own grade of the step that made
    it - the only kind of value available to a method that has no
    environment. The ``b`` best survive; the rest are dropped and never looked
    at again. At the depth limit the best-valued answer kept along the way is
    returned, and only then does the environment's verdict on it count.

    The original searches over thoughts and never acts. Here the steps still
    have to be executed for the next state to exist - there is nothing to read
    after a search until the search has happened - but nothing Tree of
    Thoughts scores or keeps is decided by what the environment replied, and
    the verdicts stay hidden in the trace until the end. There is no
    exploration bonus, no backpropagation and no reflection.
    """

    method = "tot"
    title = "Tree of Thoughts"
    blurb = (
        "Tree of Thoughts searches breadth-first, keeping the best few states "
        "at each depth by the policy's own grade; reward is {reward}."
    )

    @classmethod
    def configure(cls, config: Config) -> Config:
        return replace(config, evaluation="before")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: The environment's verdicts, withheld from the tree until the end.
        self._verdicts: dict[int, float | None] = {}

    def _answer(self, node: Node) -> bool:
        """Can this state be returned as the answer?"""
        if self.task.complete_states:
            return True
        return node.observation is not None and node.observation.terminal

    def run(self) -> dict[str, Any]:
        self._start()
        cfg = self.cfg
        frontier = [self.root]
        answers: list[Node] = []
        keep: list[Node] = []
        stop = "the depth limit was reached"
        depth = 0
        for depth in range(1, cfg.max_depth + 1):
            fresh: list[Node] = []
            for parent in frontier:
                fresh += self._expand(parent, depth)
            if not fresh:
                stop = "nothing in the beam could be expanded"
                break

            # Evaluation: the policy's grade, from before anything ran.
            for child in fresh:
                child.lm_score = child.value = child.prior
            self.rec.emit(
                "evaluation",
                iteration=depth,
                title="Evaluation",
                summary=(
                    "Each state is valued by the policy's own grade of the step "
                    "that made it, given before the step ran. Tree of Thoughts "
                    "has no environment to ask."
                ),
                detail={
                    "lam": 1.0,
                    "timing": "before",
                    "scores": [
                        {
                            "id": c.id,
                            "label": c.action.label if c.action else "root",
                            "prior": round(c.prior or 0.0, 4),
                            "lm": round(c.prior or 0.0, 4),
                            "sc": 0.0,
                            "value": round(c.prior or 0.0, 4),
                            "reward": None,
                            "why": "The policy's grade of the step, from before it ran.",
                        }
                        for c in fresh
                    ],
                },
                focus=[c.id for c in fresh],
                path=[self.root.id],
            )

            # Selection: keep the b best, drop the rest for good. Sorted on
            # (value, -id) so a tie goes to the state proposed first.
            ranked = sorted(fresh, key=lambda c: (c.value or 0.0, -c.id), reverse=True)
            keep, drop = ranked[: cfg.beam], ranked[cfg.beam :]
            for child in drop:
                child.pruned = True
            self.rec.emit(
                "selection",
                iteration=depth,
                title="Beam",
                summary=(
                    f"Keep the {len(keep)} best of {len(fresh)} by value; the "
                    f"other {len(drop)} are dropped and never reconsidered."
                    if drop
                    else f"All {len(keep)} fit in a beam of {cfg.beam}; nothing is dropped."
                ),
                detail={
                    "kind": "beam",
                    "beam": cfg.beam,
                    "depth": depth,
                    "ranked": [
                        {
                            "id": c.id,
                            "label": c.action.label if c.action else "root",
                            "value": round(c.value or 0.0, 4),
                            "kept": c in keep,
                            "answer": self._answer(c),
                        }
                        for c in ranked
                    ],
                    "kept": [c.id for c in keep],
                    "dropped": [c.id for c in drop],
                },
                focus=[c.id for c in keep],
                path=[self.root.id],
            )
            answers += [c for c in keep if self._answer(c)]
            frontier = [c for c in keep if not c.terminal]
            if not frontier:
                if depth < cfg.max_depth:
                    stop = "every state in the beam was finished"
                break

        # The answer is the best-valued answer the beam ever kept. With none,
        # the best of the last beam is returned, and it has answered nothing.
        pool = answers or keep
        best = max(pool, key=lambda c: (c.value or 0.0, -c.id)) if pool else None
        # Only now is anything checked. Every withheld verdict goes back on its
        # node, so the finished tree shows what the beam threw away.
        for node in self.root.descendants():
            if node.id in self._verdicts:
                node.reward = self._verdicts[node.id]
        if best is not None and not self._answer(best):
            best.reward = 0.0  # no answer was ever kept: nothing to check
        missed = [
            n.id
            for n in self.root.descendants()
            if n is not best and (n.reward or 0.0) >= cfg.solved_at
        ]
        return self._finish(best, depth, stop, missed=missed)

    def _expand(self, parent: Node, depth: int) -> list[Node]:
        """``n`` samples from one state; agreeing samples are one child."""
        proposals = self.policy.propose(self.task, parent.data, self.cfg.n, [])
        groups: dict[str, list[Proposal]] = {}
        for p in proposals:
            groups.setdefault(p.signature or p.action.text, []).append(p)

        children: list[Node] = []
        records: list[dict] = []
        for group in groups.values():
            child, record = self._act(parent, group[0])
            if child is not None:
                child.prior = sum(p.prior for p in group) / len(group)
                record["samples"] = len(group)
                record["prior"] = round(child.prior, 4)
                # Tree of Thoughts never sees this. Hide it until the end.
                record["reward"] = None
                self._verdicts[child.id] = child.reward
                child.reward = None
                children.append(child)
            records.append(record)

        self.rec.emit(
            "expansion",
            iteration=depth,
            title="Expansion",
            summary=(
                f"{self.cfg.n} samples from #{parent.id} reached "
                f"{len(children)} distinct state{'' if len(children) == 1 else 's'}."
            ),
            detail={"parent": parent.id, "n": self.cfg.n, "children": records},
            focus=[c.id for c in children] or [parent.id],
            path=[n.id for n in parent.path],
        )
        return children


#: Everything ``--method`` accepts besides ``lats``, by the name used for it.
METHODS: dict[str, type[_Method]] = {
    ReAct.method: ReAct,
    Reflexion.method: Reflexion,
    TreeOfThoughts.method: TreeOfThoughts,
}
