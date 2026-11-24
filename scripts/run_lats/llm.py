"""
The two policies: a deterministic offline stand-in, and a real OpenAI call.

The design rule for the whole submission is that the default path must run with
**no API key and no network**. So :class:`MockLLM` is the default everywhere,
and it is honest about what it is: a seeded sampler over a bank of candidate
actions that each task defines for itself, and a per-task heuristic for grading
the states that come back.

What is *not* mocked is the environment. The code really runs, the arithmetic
really evaluates, the retrieval really searches. Every reward in every trace
this package writes came out of a real execution - which is the part that
matters, because the reward is what LATS backpropagates.

:class:`OpenAILLM` swaps in a real model. It is selected with ``--llm openai``
and is never used unless you ask for it.

A policy does two jobs, and the order between them is the point of LATS:

``propose``    sample ``n`` candidate actions from a state, each one an
               independent draw, so that two samples agreeing means something.
               Each comes with the model's ``prior`` - its opinion of the idea
               before anything has run.
``evaluate``   grade one state *after* the environment has executed the action
               that reached it, from a trajectory that includes the reply. This
               is ``LM(s)``. It is a separate call made later, not a number
               carried over from ``propose``.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any

from .tasks.base import Task
from .types import Action, Grade, Observation, Proposal

#: How the ``n`` samples of one expansion are drawn, recorded in every trace.
#: ``SC(s)`` is a count of agreements, so it is only informative when the
#: samples are independent draws - version 1.0 asked a real model for ``n``
#: *distinct* actions in one reply, under which every count is 1.
SAMPLING = "independent"


class MockLLM:
    """A deterministic policy. Same seed, same trace, byte for byte.

    Sampling happens *with replacement* from the task's candidate bank, which
    is deliberate: the self-consistency term ``SC(s)`` is defined as the share
    of the ``n`` samples that agree, so duplicates are not waste, they are the
    measurement.

    Grading is the task's own heuristic (:meth:`Task.mock_evaluate`), run on
    the state the environment returned. It draws nothing from the random
    stream, so turning evaluation on or off cannot change which actions are
    sampled.
    """

    kind = "mock"
    name = "MockLLM"

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.calls = 0
        self.tokens = 0

    def propose(
        self, task: Task, data: dict, n: int, reflections: list[str]
    ) -> list[Proposal]:
        proposals = task.mock_propose(data, self.rng, n, reflections)
        self.calls += 1
        self.tokens += _estimate_tokens(
            task.render(data) + task.action_schema() + " ".join(reflections),
            " ".join(p.action.text + p.rationale for p in proposals),
        )
        return proposals

    def evaluate(
        self,
        task: Task,
        data: dict,
        observation: Observation,
        action: Action,
        prior: float,
        reflections: list[str],
    ) -> Grade:
        grade = task.mock_evaluate(data, observation, action, prior, reflections)
        self.calls += 1
        self.tokens += _estimate_tokens(
            task.render(data)
            + task.render_observation(observation)
            + " ".join(reflections),
            grade.why,
        )
        return grade

    def info(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "model": None,
            "seed": self.seed,
            "calls": self.calls,
            "tokens": self.tokens,
            "tokens_are_estimated": True,
            "sampling": SAMPLING,
        }


def _estimate_tokens(prompt: str, completion: str) -> int:
    """A four-characters-per-token estimate of what this call would have cost.

    Rough on purpose, and flagged as an estimate in the trace. It exists so the
    cost axis of the demo is not simply missing when the mock policy is used;
    the shape of the curve is right even though the constant is not.
    """
    return (len(prompt) + len(completion)) // 4


# ---------------------------------------------------------------------------
# The real model
# ---------------------------------------------------------------------------


PROPOSE_SYSTEM = (
    "You are the policy inside a Language Agent Tree Search. You are given the "
    "current state of a partially solved task. Propose the single next action "
    "you judge best, and grade it yourself on how likely it is to lead to a "
    "correct solution. Your grade is a prior only: the environment will execute "
    "your action, and a separate evaluation will judge what happened. Reply with "
    "JSON and nothing else."
)

VALUE_SYSTEM = (
    "You are the value function inside a Language Agent Tree Search. You are "
    "shown a partial trajectory: the task, the actions taken so far, and what "
    "the environment replied to the most recent one. Judge, from all of it, how "
    "likely this trajectory is to end in a correct solution. Reply with JSON and "
    "nothing else."
)


class OpenAILLM:
    """The same interface as :class:`MockLLM`, backed by a real model.

    Needs ``OPENAI_API_KEY``. ``OPENAI_BASE_URL`` points the SDK at a different
    endpoint - a compatible local server, or a gateway - in which case the key
    may not be needed at all.

    **Proposals are independent samples.** One prompt asks for one action, and
    it is sampled ``n`` times: as ``n`` choices of a single request where the
    endpoint allows it, so the prompt is paid for once; in batches where it caps
    ``n`` (gpt-5 takes 8); and as ``n`` separate requests where it takes no
    ``n`` at all. Asking for ``n`` *distinct* actions in one reply
    - which version 1.0 did - would make every sample disagree with every other
    by construction and leave ``SC(s)`` pinned at ``1/n``.

    **Grading is a second call, made after execution.** Every state the
    environment returns is graded on its own, from a prompt that shows the
    trajectory and the environment's reply to its last action. That is one
    request per state, which is most of what a real-model search costs.

    Reply parsing is deliberately lenient rather than schema-enforced, so
    ``--model`` is free to name anything the endpoint serves.

    Traces produced this way are *not* reproducible, and the search will run
    whatever code the model writes. See the warning in the README.
    """

    kind = "openai"
    #: Used when ``--model`` is not given.
    default_model = "gpt-5"

    def __init__(self, model: str | None = None, effort: str | None = "low") -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SystemExit(
                "The openai policy needs the OpenAI SDK: pip install openai"
            ) from exc

        self.model = model or self.default_model
        self.name = f"OpenAI ({self.model})"
        self.effort = effort
        #: Whether this endpoint has accepted ``n > 1`` so far. Cleared, for
        #: good, the first time it refuses.
        self.multi_choice = True
        #: The most choices one request may ask for, once the endpoint has
        #: named a limit; ``None`` until then.
        self.max_choices: int | None = None
        self.calls = 0
        self.tokens = 0
        self._openai = openai
        # Resolves OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
        try:
            self.client = openai.OpenAI()
        except openai.OpenAIError as exc:
            raise SystemExit(
                f"Could not construct the OpenAI client: {exc}\n"
                "Export OPENAI_API_KEY, or set OPENAI_BASE_URL to an endpoint "
                "that does not need one."
            ) from exc

    # -- the interface the search sees --------------------------------------

    def propose(
        self, task: Task, data: dict, n: int, reflections: list[str]
    ) -> list[Proposal]:
        prompt = self._propose_prompt(task, data, reflections)
        out: list[Proposal] = []
        for text in self._sample(PROPOSE_SYSTEM, prompt, n):
            proposal = self._parse_proposal(task, text, data)
            if proposal is not None:
                out.append(proposal)
        return out

    def evaluate(
        self,
        task: Task,
        data: dict,
        observation: Observation,
        action: Action,
        prior: float,
        reflections: list[str],
    ) -> Grade:
        # ``prior`` is deliberately not in the prompt: the grade should come
        # from what happened, not from what the policy hoped would happen.
        replies = self._complete(
            VALUE_SYSTEM, self._value_prompt(task, data, observation, reflections)
        )
        payload = _first_json_object(replies[0] if replies else "") or {}
        if "score" not in payload:
            return Grade(0.5, "No parseable grade came back, so 0.5 stands in.")
        return Grade(
            _unit(payload.get("score"), 0.5), str(payload.get("why", "")).strip()[:240]
        )

    def info(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "model": self.model,
            "effort": self.effort,
            "seed": None,
            "calls": self.calls,
            "tokens": self.tokens,
            "tokens_are_estimated": False,
            "sampling": SAMPLING,
        }

    # -- the requests -------------------------------------------------------

    def _sample(self, system: str, prompt: str, n: int) -> list[str]:
        """``n`` independent completions of one prompt."""
        replies: list[str] = []
        # As many choices per request as the endpoint allows: gpt-5 takes at
        # most 8, so twelve samples are a request for 8 and one for 4.
        while self.multi_choice and n - len(replies) > 1:
            want = n - len(replies)
            got = self._complete(system, prompt, min(want, self.max_choices or want))
            if not got:
                break
            replies += got
        # Top up one request at a time: an endpoint that refused ``n`` returns
        # nothing above, and one that ignored it returns a single choice.
        while len(replies) < n:
            replies += self._complete(system, prompt)[:1] or [""]
        return replies[:n]

    def _complete(self, system: str, prompt: str, n: int = 1) -> list[str]:
        """Send one request for ``n`` choices. Returns their texts."""
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            # A reasoning model spends most of this thinking, and a reply cut
            # off by the cap comes back empty - a sample paid for and lost.
            # One gpt-5 sample on the hard Game of 24 thinks for ~4,000 tokens.
            "max_completion_tokens": 16000,
        }
        if n > 1:
            request["n"] = n
        # Only the reasoning models accept reasoning_effort. Rather than keep a
        # list of which ones do - it changes every few months - send it, and
        # drop it for good the first time an endpoint says no.
        if self.effort:
            request["reasoning_effort"] = self.effort

        try:
            response = self._create(request)
        except self._openai.APIStatusError as exc:
            if self.effort and "reasoning_effort" in str(exc):
                self.effort = None
                return self._complete(system, prompt, n)
            if n > 1:
                cap = _choice_cap(exc)
                if cap is not None and 1 < cap < n:
                    # "Expected a value <= 8": keep batching, in smaller batches.
                    self.max_choices = cap
                    return self._complete(system, prompt, cap)
                # Not every model takes n > 1. If the refusal was about
                # something else, the single request that follows fails too
                # and reports it.
                self.multi_choice = False
                return []
            raise SystemExit(f"OpenAI request failed: {exc}") from exc

        self.calls += 1
        usage = response.usage
        self.tokens += usage.total_tokens if usage else 0
        return [
            "" if getattr(choice.message, "refusal", None) else choice.message.content or ""
            for choice in response.choices or []
        ]

    def _create(self, request: dict[str, Any]):
        try:
            return self.client.chat.completions.create(**request)
        except self._openai.APIConnectionError as exc:
            raise SystemExit(f"Could not reach the OpenAI API: {exc}") from exc

    # -- prompts, and turning replies into moves ----------------------------

    @staticmethod
    def _propose_prompt(task: Task, data: dict, reflections: list[str]) -> str:
        return (
            f"{task.render(data)}{_notes(reflections)}\n\n"
            f"Propose the next action.\n{task.action_schema()}"
        )

    @staticmethod
    def _value_prompt(
        task: Task, data: dict, observation: Observation, reflections: list[str]
    ) -> str:
        return (
            f"{task.render(data)}\n\n"
            "The environment's reply to the last action:\n"
            f"{task.render_observation(observation)}{_notes(reflections)}\n\n"
            'Reply with JSON: {"why": "one sentence", "score": 0.0-1.0}, where '
            "score is how likely this trajectory is to end in a correct solution."
        )

    @staticmethod
    def _parse_proposal(task: Task, text: str, data: dict) -> Proposal | None:
        """Turn one reply into a proposal, or ``None`` if it is malformed.

        A policy that occasionally returns nothing usable is normal; the search
        loop treats an expansion that yields nothing as a dead end rather than
        a crash.
        """
        payload = _first_json_object(text)
        if not payload:
            return None
        # Lenient about shape: one object, as asked, or the first of a list a
        # model sent anyway.
        item: Any = payload
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates:
            item = candidates[0]
        if not isinstance(item, dict):
            return None
        try:
            action = task.parse_action(item, data)
        except (KeyError, ValueError, TypeError):
            return None
        return Proposal(
            action=action,
            prior=_unit(item.get("prior", item.get("self_eval")), 0.5),
            rationale=str(item.get("text", "")),
            signature=task.signature(action),
        )


def _notes(reflections: list[str]) -> str:
    """The reflections in context, prose only - the tail is for the mock."""
    if not reflections:
        return ""
    joined = "\n".join(f"- {r.split('|')[0].strip()}" for r in reflections)
    return (
        "\n\nNotes written after earlier attempts failed. Do not repeat those "
        f"mistakes:\n{joined}"
    )


def _choice_cap(exc: Exception) -> int | None:
    """The largest ``n`` an endpoint says it accepts, if that is the refusal.

    OpenAI's reply names it: ``Invalid 'n': integer above maximum value.
    Expected a value <= 8``. Anything else - ``n`` not supported at all, or a
    refusal about something other than ``n`` - is ``None``.
    """
    if getattr(exc, "param", None) != "n":
        return None
    match = re.search(r"<=\s*(\d+)", str(exc))
    return int(match.group(1)) if match else None


def _unit(value: Any, default: float) -> float:
    """A model's number, clamped into [0, 1], or ``default`` if it is not one."""
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return default


def _first_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of a reply, fenced or not."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


#: Everything ``--llm`` accepts, in the order it is offered on the command line.
POLICIES = ("mock", "openai")


def build_policy(
    kind: str, seed: int, model: str | None = None
) -> MockLLM | OpenAILLM:
    """Construct the policy named on the command line."""
    if kind == "mock":
        return MockLLM(seed=seed)

    if kind == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            if os.environ.get("OPENAI_BASE_URL"):
                print(
                    "note: no OPENAI_API_KEY in the environment; using "
                    "OPENAI_BASE_URL, which is assumed not to need one."
                )
            else:
                raise SystemExit(
                    "The openai policy needs an API key. Export OPENAI_API_KEY:\n"
                    "    export OPENAI_API_KEY=sk-...       # macOS / Linux\n"
                    "    $env:OPENAI_API_KEY = 'sk-...'     # PowerShell\n"
                    "Or set OPENAI_BASE_URL to a compatible endpoint that does "
                    "not need one."
                )
        return OpenAILLM(model=model)

    raise SystemExit(
        f"unknown policy {kind!r}: expected one of {', '.join(POLICIES)}"
    )
