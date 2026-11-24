"""
What a task has to provide for LATS to search it.

A task is an *environment*, not a prompt. It owns the state, it executes
actions for real, and it is the only thing allowed to say what a reward is.
The search loop never inspects ``data``; it only ever asks the task.

Some of the methods exist for the two policies:

``mock_propose`` / ``mock_evaluate``
                   the offline stand-in policy draws candidates from a bank the
                   task defines, and grades the states that come back with a
                   heuristic the task defines, so the demo runs with no API key
                   and produces the same trace every time.
``render`` / ``render_observation`` / ``parse_action``
                   the OpenAI-backed policy needs the state and the
                   environment's reply as text, and needs to turn a JSON reply
                   back into an :class:`Action`.
"""

from __future__ import annotations

import random
import re
from typing import Any

from ..types import Action, Grade, Observation, Proposal


class Task:
    """Base class. Subclasses override everything that raises."""

    #: Stable identifier, used as the trace filename.
    id: str = "task"
    #: One of ``programming`` | ``game`` | ``qa``. Only used for display and
    #: for picking the paper's per-family defaults.
    family: str = "programming"
    #: True when every state is already a whole answer - a complete program -
    #: rather than a step on the way to one. Tree of Thoughts reads it to know
    #: which states it may return.
    complete_states: bool = False
    title: str = ""
    #: The problem statement shown to the agent and to the viewer.
    prompt: str = ""
    #: One line describing where the objective reward comes from.
    reward_desc: str = ""
    #: Anything the viewer should show alongside the task - the test suite,
    #: the starting numbers, the corpus size.
    context: dict[str, Any] = {}

    def defaults(self) -> dict[str, Any]:
        """Per-task overrides applied on top of :class:`~run_lats.types.Config`."""
        return {}

    # -- the environment ----------------------------------------------------

    def root_data(self) -> dict[str, Any]:
        """Environment state before the agent has done anything."""
        raise NotImplementedError

    def step(self, data: dict, action: Action) -> tuple[dict, Observation]:
        """Execute ``action`` against ``data``. Returns the new state and what
        the environment said back. This is where real work happens: code is
        run, arithmetic is evaluated, documents are retrieved."""
        raise NotImplementedError

    # -- the offline policy -------------------------------------------------

    def mock_propose(
        self, data: dict, rng: random.Random, n: int, reflections: list[str]
    ) -> list[Proposal]:
        """Sample ``n`` candidate actions, with replacement, without calling a
        model. Duplicates are the point: they are what ``SC(s)`` counts."""
        raise NotImplementedError

    def mock_evaluate(
        self,
        data: dict,
        observation: Observation,
        action: Action,
        prior: float,
        reflections: list[str],
    ) -> Grade:
        """Grade the state ``action`` led to, *after* the environment replied.

        The offline stand-in for ``LM(s)``. It sees what a real model's value
        prompt would see - the new state, the observation, the notes in
        context, and the policy's own earlier opinion of the move - and must be
        deterministic. It should also be fallible: a grader that could not be
        fooled would make the search pointless.
        """
        raise NotImplementedError

    # -- the online policy --------------------------------------------------

    def render(self, data: dict) -> str:
        """The current state as text, for a real model's prompt."""
        raise NotImplementedError

    def render_observation(self, observation: Observation) -> str:
        """The environment's reply as the *agent* sees it.

        Usually the observation text itself. A task whose text is written for
        the viewer as well - one that names the gold answer, say - overrides
        this so the value function is not handed the oracle.
        """
        return observation.text

    def action_schema(self) -> str:
        """One paragraph telling a real model what JSON to return for *one*
        action. The policy asks for ``n`` independent replies, not ``n``
        actions in one reply - see :class:`~run_lats.llm.OpenAILLM`."""
        raise NotImplementedError

    def signature(self, action: Action) -> str:
        """What two independently sampled actions share when they *agree*.

        The default is the action text with case and spacing normalised. A
        task where two spellings reach the same state - ``2 * 11`` and
        ``11 * 2`` - overrides it, so that ``SC(s)`` counts agreement on the
        move rather than on its wording.
        """
        return re.sub(r"\s+", " ", action.text.strip().lower())

    def parse_action(self, obj: dict, data: dict) -> Action:
        """Turn one JSON object from a real model into an :class:`Action`.

        ``data`` is the state the action would be applied to. Most tasks do not
        need it - a program or a tool call means the same thing wherever it
        appears - but a task whose actions only make sense against the current
        state does, and it is also the only chance to reject a move the model
        invented. Raise :class:`ValueError` to drop the candidate; the search
        treats an expansion that yields nothing as a dead end.
        """
        raise NotImplementedError

    # -- reflection ---------------------------------------------------------

    def reflection_for(self, node_data: dict, observation: Observation | None) -> str:
        """A short note on why this trajectory failed, written from the
        environment's own output rather than invented."""
        if observation is None:
            return "This trajectory ended without any environment feedback."
        return f"That attempt scored {observation.reward}. {observation.text}"

    # -- display ------------------------------------------------------------

    def brief(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "title": self.title,
            "prompt": self.prompt,
            "reward": self.reward_desc,
            "context": self.context,
        }
