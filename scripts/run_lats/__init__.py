"""
A from-scratch Language Agent Tree Search, small enough to read in one sitting.

Zhou, Yan, Shlapentokh-Rothman, Wang and Wang, *Language Agent Tree Search
Unifies Reasoning, Acting and Planning in Language Models*, ICML 2024
(arXiv:2310.04406).

Written for the NeurIPS 2026 Education Track. Nothing here is vendored from the
authors' repository or from any agent framework; :mod:`run_lats.search` is a
direct reading of Algorithm 1, and the rest is the scaffolding it needs to run
against a real environment and to record what it did.

    from run_lats import LATS, Config, TASKS
    from run_lats.llm import MockLLM

    task = TASKS["game-of-24"]()
    search = LATS(task, Config(**task.defaults()), MockLLM(seed=7))
    print(search.run())

The module layout:

``types``      the vocabulary - actions, observations, grades, nodes, config
``tasks/``     the environments, which own the state and the reward
``llm``        the policies: a deterministic offline mock, and a real OpenAI call
``search``     the six operations
``baselines``  ReAct, Reflexion and Tree of Thoughts, for comparison
``trace``      the replayable recording the browser demo reads

Version 1.1 moved ``LM(s)`` to a separate grading call made after the
environment replies, and made a real model's ``n`` samples independent draws.
Traces written by 1.0 carry neither change; the viewer says so when it opens
one.
"""

from .baselines import METHODS, ReAct, Reflexion, TreeOfThoughts
from .search import LATS, backpropagate, uct
from .tasks import TASKS, Task
from .trace import SCHEMA, TraceRecorder
from .types import Action, Config, Grade, Node, Observation, Proposal

__version__ = "1.1.0"

__all__ = [
    "LATS",
    "METHODS",
    "Action",
    "Config",
    "Grade",
    "Node",
    "Observation",
    "Proposal",
    "ReAct",
    "Reflexion",
    "SCHEMA",
    "TASKS",
    "Task",
    "TraceRecorder",
    "TreeOfThoughts",
    "backpropagate",
    "uct",
    "__version__",
]
