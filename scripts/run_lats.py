#!/usr/bin/env python3
"""
Run Language Agent Tree Search and write a trace the viewer can replay.

    python scripts/run_lats.py                      # every bundled preset
    python scripts/run_lats.py --list               # what can be run
    python scripts/run_lats.py --task game-of-24   # one task, its own defaults
    python scripts/run_lats.py --task game-of-24 --w 0 --seed 3 --name game-of-24_greedy
    python scripts/run_lats.py --task game-of-24 --method react   # a baseline
    python scripts/run_lats.py --publish            # refresh public/traces/

Traces land in a timestamped directory under ``results/lats-traces/``, which is
gitignored::

    results/lats-traces/20260901-041530/
        traces-manifest.json    an index of what this run produced
        traces/
            mock_game-of-24.json ...    one file per trace

A name is ``<policy>_<task>_<variant>``: ``-`` joins the words of one phrase,
``_`` joins the phrases. So ``mock_game-of-24_no-value`` reads as the mock
policy, on Game of 24, with the value function ablated.

Drop any of those onto the viewer to step through it. ``--publish`` writes into
``public/traces/`` instead - that is the committed set the viewer loads on
startup, so use it when you mean to change what ships.

With the default ``--llm mock`` policy this needs no API key and no network,
and the same seed always produces the same trace, byte for byte. ``--llm openai``
swaps in a real model, reading ``OPENAI_API_KEY`` from the environment or from
a ``.env`` file at the repository root; see the README before using it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_lats import LATS, METHODS, SCHEMA, Config, TASKS  # noqa: E402
from run_lats.env import load_env  # noqa: E402
from run_lats.llm import POLICIES, OpenAILLM, build_policy  # noqa: E402
from run_lats.trace import TraceRecorder  # noqa: E402

#: Everything ``--method`` accepts, LATS first.
METHOD_NAMES = ("lats", *METHODS)

ROOT = SCRIPTS.parent
#: Generated runs. Gitignored, one timestamped directory each.
TRACE_RESULTS = ROOT / "results" / "lats-traces"
#: The committed set the viewer serves as static files.
PUBLIC_TRACES = ROOT / "public" / "traces"

MANIFEST_SCHEMA = "lats-trace-manifest/1"


def manifest_path(trace_dir: Path) -> Path:
    """Where the index of ``trace_dir`` lives: beside the folder, not inside it.

    ``public/traces/`` is indexed by ``public/traces-manifest.json``. The index
    is not a trace, and keeping it out means every file in the folder is one -
    the viewer can list the directory, a script can glob it, and nothing has to
    special-case a name.
    """
    return trace_dir.parent / f"{trace_dir.name}-manifest.json"


def trace_name(base: str, llm: str) -> str:
    """``game-of-24`` + ``openai`` -> ``openai_game-of-24``.

    Which policy produced a trace is the first thing you want to know about it
    and the easiest thing to lose track of, so it goes in the name rather than
    only inside the file. Deriving it from ``--llm`` means a trace cannot be
    mislabelled by hand.
    """
    return base if base.startswith(f"{llm}_") else f"{llm}_{base}"


#: The offline traces that ship with the viewer, in the order the picker
#: groups them: one environment at a time, the plain run before its ablations,
#: and LATS before the methods it is compared with. Four of these are
#: ablations, which is the point - the interesting thing about a knob is what
#: happens when you turn it - and the baselines share every setting with the
#: LATS run beside them that their method has a use for.
PRESETS: list[dict] = [
    {
        "name": "game-of-24",
        "task": "game-of-24",
        "note": (
            "All six operations, including simulation: the tree descends greedily "
            "by value, in the real environment, until one number is left. Solved "
            "on the fifth iteration, after four dead ends."
        ),
        "overrides": {},
    },
    {
        "name": "game-of-24_no-value",
        "task": "game-of-24",
        "note": (
            "Ablation: lambda = 0, so V(s) is self-consistency alone and the "
            "value function's grade is thrown away. A solution does turn up - "
            "built during a rollout on the first iteration - but selection never "
            "walks back into it, so all twelve iterations back up a reward of 0."
        ),
        "overrides": {"lam": 0.0},
    },
    {
        "name": "game-of-24_greedy",
        "task": "game-of-24",
        "note": (
            "Ablation: w = 0, so UCT collapses to pure exploitation. It starves "
            "the branch that wins: the answer opens on a fraction, which the "
            "policy scores badly, and without the exploration bonus that branch "
            "stays the least visited of the five. Never finds 24."
        ),
        "overrides": {"w": 0.0},
    },
    {
        "name": "game-of-24_react",
        "task": "game-of-24",
        "method": "react",
        "note": (
            "ReAct on the same puzzle: one sample per step, taken without "
            "comparison. It opens with 2 / 8 - the first move LATS needed five "
            "iterations to settle on - then takes 5 - 11 and ends at -24. One "
            "trajectory, and no way back into it."
        ),
        "overrides": {},
    },
    {
        "name": "game-of-24_reflexion",
        "task": "game-of-24",
        "method": "reflexion",
        "note": (
            "Reflexion: twelve ReAct trials, a note after each failure. Trial 1 "
            "opens with 2 / 8, goes wrong a step later, and its note says not to "
            "open with that step again - blaming the one move that was right. No "
            "later trial opens with it, and all twelve fail. A note blames a "
            "whole trajectory; in LATS the value of 2 / 8 is an average over "
            "every rollout through it, so one bad continuation does not condemn "
            "the opening."
        ),
        "overrides": {},
    },
    {
        "name": "game-of-24_tot",
        "task": "game-of-24",
        "method": "tot",
        "note": (
            "Tree of Thoughts with a beam of two. Of the five openings it draws, "
            "only 2 / 8 can still reach 24, and the policy grades that fraction "
            "0.15 - so the first cut keeps 5 - 2 (0.9) and 2 * 5 (0.7) and drops "
            "it for good. The beam ends three tidy steps later at 1. LATS gives "
            "2 / 8 the same low first value; the exploration bonus is what "
            "brings it back."
        ),
        "overrides": {},
    },
    {
        "name": "merge-intervals",
        "task": "merge-intervals",
        "note": (
            "The programming setting: every node is a complete program, its "
            "reward is the fraction of tests it passes, and simulation is "
            "skipped. Before anything runs the model prefers the one-pass sweep, "
            "0.70 to 0.62; once the tests have run the grader reads 3 of 5 "
            "against 4 of 5 and the order flips. Exploration still spends "
            "iteration 2 on the one-pass branch, where every patch caps at three "
            "of five, and iteration 3 solves it from sort-then-sweep."
        ),
        "overrides": {},
    },
    {
        "name": "merge-intervals_value-before",
        "task": "merge-intervals",
        "note": (
            "Ablation, and the old behaviour: LM(s) is the model's grade from "
            "before the tests ran - the ordering Tree of Thoughts uses, and what "
            "version 1.0 of this generator did on every trace. The one-pass "
            "sweep wins the first evaluation on opinion alone and is backed up "
            "at 0.6, and exploration rather than the value carries iteration 2 "
            "to the right branch. Both runs reach 5 of 5; what differs is what "
            "V(s) means - here what the model expected, beside it what the "
            "tests said."
        ),
        "overrides": {"evaluation": "before"},
    },
    {
        "name": "merge-intervals_react",
        "task": "merge-intervals",
        "method": "react",
        "note": (
            "ReAct: one program, then one patch of it. It writes the model's "
            "favourite, the one-pass sweep, patches it with the model's favourite "
            "fix, and stops at 3 of 5. The approach was the bug, and ReAct never "
            "sees an alternative to it."
        ),
        "overrides": {},
    },
    {
        "name": "merge-intervals_reflexion",
        "task": "merge-intervals",
        "method": "reflexion",
        "note": (
            "Reflexion: trial 1 is the ReAct run, 3 of 5. Its note reads the "
            "failures - every failing case arrives out of order - and says sort "
            "first. Trial 2 opens with sort-then-sweep and patches it to 5 of 5. "
            "On this task one well-aimed note does the work of the whole tree."
        ),
        "overrides": {},
    },
    {
        "name": "merge-intervals_tot",
        "task": "merge-intervals",
        "method": "tot",
        "note": (
            "Tree of Thoughts, beam of two, valued by the model's opinion alone. "
            "At depth 2 it writes the correct program - merge on touch too, 5 of "
            "5 - grades it 0.64, and drops it for two patches it likes more, "
            "0.78 and 0.71. It returns a program that passes 3 of 5. The tests "
            "were run on every program and read for none."
        ),
        "overrides": {},
    },
    {
        "name": "multihop-qa",
        "task": "multihop-qa",
        "note": (
            "Two-hop retrieval with a distractor. A grader that counts words "
            "cannot tell the UCT page from the page that holds the answer - 0.59 "
            "each - and the first rollout opens with the distractor. The answer "
            "it then tries twice, ECML 2006, comes straight back rejected both "
            "times, so the rollout runs out of depth rather than committing to "
            "it. Iteration 2 starts from the right page, and the policy's "
            "recency bias now points at the right venue."
        ),
        "overrides": {},
    },
    {
        "name": "multihop-qa_no-reflection",
        "task": "multihop-qa",
        "note": (
            "Ablation: reflection off. Identical outcome, one step shorter - which "
            "is the paper's own finding. Reflection is the smallest of its three "
            "ablations; the value function and the search structure matter more."
        ),
        "overrides": {"reflect": False},
    },
    {
        "name": "multihop-qa_react",
        "task": "multihop-qa",
        "method": "react",
        "note": (
            "ReAct: it reads the distractor, then the page that holds the "
            "answer, and answers from the distractor - ECML 2006. One sample per "
            "step, so a single draw of the policy's bias decides it, and nothing "
            "gets a second look."
        ),
        "overrides": {},
    },
    {
        "name": "multihop-qa_reflexion",
        "task": "multihop-qa",
        "method": "reflexion",
        "note": (
            "Reflexion: eight trials, eight failures. Every note is accurate - "
            "each names the wrong answer and the page it came from - and none "
            "says where to look, so each trial learns one more answer not to "
            "give and wanders the corpus again. With no value function, nothing "
            "carries over about which states were promising."
        ),
        "overrides": {},
    },
    {
        "name": "multihop-qa_tot",
        "task": "multihop-qa",
        "method": "tot",
        "note": (
            "Tree of Thoughts: by depth four it has written the right answer and "
            "two wrong ones, and the model is exactly as sure of Computers and "
            "Games 2006 as of EMNLP 2023 - 0.85 each. With nothing from the "
            "environment to separate them, it returns the one it found first, "
            "which is wrong."
        ),
        "overrides": {},
    },
    {
        "name": "game-of-24-hard",
        "task": "game-of-24-hard",
        "note": (
            "6, 9, 9, 10, where the twelve best-looking first moves are all dead "
            "ends and every solution ends in 9 + 15. On this seed the offline "
            "policy, which rates a fraction badly, never proposes a move that "
            "reaches 15, so sixteen iterations of correct search find nothing. "
            "Search cannot choose what the policy never suggests - compare the "
            "Reflexion run, and the OpenAI runs, on the same puzzle."
        ),
        "overrides": {},
    },
    {
        "name": "game-of-24-hard_reflexion",
        "task": "game-of-24-hard",
        "method": "reflexion",
        "note": (
            "A seed on which starting over beats searching: Reflexion solves the "
            "puzzle on its fifth trial, and the LATS run on the same seed never "
            "does. LATS expanded the root once and searched the five moves it "
            "drew there. Reflexion starts again every trial, each note forbids "
            "the opening that just failed, and trial 5 opens with 6 / 10 - a "
            "fraction no tidy-minded policy picks first - then 9 / 3/5 = 15 and "
            "9 + 15 = 24. Over sixty seeds LATS still solves this puzzle more "
            "often, 17 to 10; the README has the table."
        ),
        "overrides": {},
    },
]


def build_config(task, args: argparse.Namespace) -> Config:
    """Task defaults first, then anything given explicitly on the command line."""
    cfg = replace(Config(), **task.defaults())
    explicit = {
        "n": args.n,
        "w": args.w,
        "lam": getattr(args, "lambda"),
        "iterations": args.iterations,
        "max_depth": args.max_depth,
        "seed": args.seed,
        "beam": args.beam,
    }
    cfg = replace(cfg, **{k: v for k, v in explicit.items() if v is not None})
    if args.no_simulate:
        cfg = replace(cfg, simulate=False)
    if args.no_reflect:
        cfg = replace(cfg, reflect=False)
    if args.evaluate_before:
        cfg = replace(cfg, evaluation="before")
    return cfg


def run_one(
    task_id: str,
    cfg: Config,
    *,
    name: str,
    out: Path,
    llm: str,
    model: str | None,
    method: str = "lats",
    quiet: bool = False,
) -> dict:
    """Search one task, write its trace, and return its manifest entry."""
    if task_id not in TASKS:
        raise SystemExit(
            f"unknown task {task_id!r}. Known: {', '.join(sorted(TASKS))}"
        )
    task = TASKS[task_id]()
    algorithm = LATS if method == "lats" else METHODS[method]
    if algorithm is not LATS:
        # The baselines fix some knobs - ReAct takes one sample - and the trace
        # records the config that actually ran, not the one that was asked for.
        cfg = algorithm.configure(cfg)
    policy = build_policy(llm, cfg.seed, model)
    recorder = TraceRecorder(task, cfg, policy, method=method)
    result = algorithm(task, cfg, policy, recorder).run()
    doc = recorder.to_dict(result, name=name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    if not quiet:
        status = "solved" if result["solved"] else "not solved"
        print(
            f"  {name:<36} {status:<10} "
            f"reward {result['best_reward']}  "
            f"{result['nodes']} nodes  {len(recorder.steps)} steps  "
            f"{policy.info()['calls']} calls"
        )
    return manifest_entry(doc, out.name)


def manifest_entry(doc: dict, file: str) -> dict:
    """What the picker needs to know about a trace without opening it.

    A trace written before version 1.1 is marked ``legacy``: it took ``LM(s)``
    from the policy's grade before execution and, under a real model, asked for
    distinct actions, so the viewer has something to say about it before a
    single step is shown.
    """
    entry = {
        "file": file,
        "name": doc["name"],
        "task": doc["task"]["id"],
        "title": doc["task"]["title"],
        "family": doc["task"]["family"],
        "method": doc.get("method", "lats"),
        "solved": doc["result"]["solved"],
        "best_reward": doc["result"]["best_reward"],
        "nodes": doc["result"]["nodes"],
        "steps": len(doc["steps"]),
        "policy": doc["policy"]["kind"],
    }
    if "evaluation" not in doc.get("config", {}):
        entry["legacy"] = True
    return entry


def refresh(entry: dict, folder: Path) -> dict:
    """Re-derive a kept manifest entry from its file, keeping its note.

    So an entry written by an older version of this script picks up the fields
    a newer one adds - ``method``, ``legacy`` - without the trace being re-run.
    """
    path = folder / entry["file"]
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return entry
    fresh = manifest_entry(doc, entry["file"])
    if entry.get("note"):
        fresh["note"] = entry["note"]
    return fresh


def display(path: Path) -> str:
    """Repo-relative when it can be, absolute when the target is elsewhere."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_manifest(folder: Path) -> list[dict]:
    path = manifest_path(folder)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("traces", [])
    except (json.JSONDecodeError, OSError):
        return []


def write_manifest(folder: Path, entries: list[dict]) -> Path:
    """An index of a trace folder, so the viewer knows what is there.

    Deliberately free of timestamps: regenerating ``public/traces/`` with the
    same seeds should produce no diff at all.
    """
    path = manifest_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": MANIFEST_SCHEMA, "traces": entries}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def preset_order(entries: list[dict]) -> list[dict]:
    """The reading order: one environment at a time, working example first.

    Environments come in the order :data:`PRESETS` first mentions them; within
    an environment the offline policy comes before a real model, LATS before
    the baselines it is compared with, and a run that solved its task before
    one that did not. Someone opening the viewer should land on a search that
    works, not on an ablation that fails.

    The viewer sorts by the same rule, so a hand-promoted trace lands in the
    right group whether or not this function put it there.
    """
    tasks: dict[str, int] = {}
    for preset in PRESETS:
        tasks.setdefault(preset["task"], len(tasks))
    order = {p["name"]: i for i, p in enumerate(PRESETS)}

    def rank(base: str) -> int:
        """The curated index, matching on the longest preset name that fits.

        A variant like ``game-of-24-hard_wide`` is not a preset itself, but it
        belongs beside ``game-of-24-hard`` rather than at the end of the list.
        """
        if base in order:
            return order[base]
        prefixes = [n for n in order if base.startswith(f"{n}_")]
        return order[max(prefixes, key=len)] if prefixes else len(PRESETS)

    def key(entry: dict) -> tuple:
        base = entry["name"].split("_", 1)[-1]
        method = entry.get("method", "lats")
        return (tasks.get(entry.get("task", ""), len(tasks)),
                0 if entry.get("policy") == "mock" else 1,
                METHOD_NAMES.index(method) if method in METHOD_NAMES else len(METHOD_NAMES),
                0 if entry.get("solved") else 1,
                rank(base),
                entry["name"])

    return sorted(entries, key=key)


def sweep(seeds: int, task_ids: list[str]) -> int:
    """Solve rates of every method on every task, over seeds 1..``seeds``.

    One bundled trace is one draw of a seeded policy, and a single draw can
    flatter any method. This is the table the README quotes beside them: the
    offline policy, each task's own defaults, ``seeds`` seeds, and the number of
    actions each method executed on average. Nothing is written to disk.
    """
    variants = [
        ("LATS", "lats", {}),
        ("LATS, value before", "lats", {"evaluation": "before"}),
        ("ReAct", "react", {}),
        ("Reflexion", "reflexion", {}),
        ("ToT", "tot", {}),
    ]
    print(f"solved / {seeds} seeds of the offline policy (mean actions executed)\n")
    print("| task | " + " | ".join(label for label, _, _ in variants) + " |")
    print("|" + " --- |" * (len(variants) + 1))
    for task_id in task_ids:
        cells = []
        for _, method, overrides in variants:
            solved = actions = 0
            for seed in range(1, seeds + 1):
                task = TASKS[task_id]()
                cfg = replace(Config(), **{**task.defaults(), "seed": seed, **overrides})
                algorithm = LATS if method == "lats" else METHODS[method]
                if algorithm is not LATS:
                    cfg = algorithm.configure(cfg)
                result = algorithm(task, cfg, build_policy("mock", seed)).run()
                solved += bool(result["solved"])
                actions += result["nodes"] - 1
            cells.append(f"{solved} ({actions / seeds:.0f})")
        print(f"| `{task_id}` | " + " | ".join(cells) + " |")
    return 0


def destination(args: argparse.Namespace) -> Path:
    """Where this invocation's traces go.

    ``--publish`` targets the committed folder the viewer serves; everything
    else gets a fresh timestamped directory, so two runs never clobber each
    other and you can diff one against the next.
    """
    if args.publish:
        PUBLIC_TRACES.mkdir(parents=True, exist_ok=True)
        return PUBLIC_TRACES
    folder = TRACE_RESULTS / datetime.now().strftime("%Y%m%d-%H%M%S") / "traces"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def report(folder: Path, count: int, args: argparse.Namespace) -> None:
    """Say where the traces went, and what to do with them."""
    print(f"\nwrote {count} trace(s) to {display(folder)}/")
    print(f"       indexed in {display(manifest_path(folder))}")
    if args.publish:
        print("that is the set the viewer loads on startup - "
              "`npm run dev` to see it.")
    else:
        print("drop one onto the viewer (`npm run dev`) to step through it, "
              "or re-run with --publish to make it part of the bundled set.")


def promote(source: Path, name: str | None, note: str | None) -> int:
    """Copy a trace that already exists into ``public/traces/`` and index it.

    A run against a real model costs money and minutes and is not reproducible,
    so deciding after the fact that one is worth shipping should not mean
    running it again. The trace keeps the name it was written with - which
    already carries its policy prefix - unless ``--name`` overrides it.
    """
    if not source.is_file():
        raise SystemExit(f"no such trace: {source}")
    try:
        doc = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"{source} is not readable as a trace: {exc}") from exc
    if doc.get("schema") != SCHEMA:
        raise SystemExit(
            f"{source} says schema {doc.get('schema')!r}; expected {SCHEMA!r}.")

    final = name or doc.get("name") or source.stem
    doc["name"] = final
    PUBLIC_TRACES.mkdir(parents=True, exist_ok=True)
    target = PUBLIC_TRACES / f"{final}.json"
    target.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    entry = manifest_entry(doc, target.name)
    if note:
        entry["note"] = note
    elif (old := next((e for e in read_manifest(PUBLIC_TRACES)
                       if e["file"] == target.name), None)) and old.get("note"):
        entry["note"] = old["note"]        # keep the description on a re-promote

    kept = [refresh(e, PUBLIC_TRACES) for e in read_manifest(PUBLIC_TRACES)
            if e["file"] != target.name]
    write_manifest(PUBLIC_TRACES, preset_order(kept + [entry]))
    print(f"promoted {display(source)}\n      -> {display(target)}"
          f"  ({target.stat().st_size // 1024} kB, "
          f"{'solved' if entry['solved'] else 'not solved'})")
    if not note and "note" not in entry:
        print("      no --note given, so the picker will show this one bare.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", help="run one task instead of every bundled preset")
    p.add_argument("--list", action="store_true",
                   help="list tasks and presets, then exit")
    p.add_argument("--publish", action="store_true",
                   help="write into public/traces/, the set the viewer ships with")
    p.add_argument("--promote", type=Path, metavar="TRACE",
                   help="add an existing trace file to public/traces/ instead of "
                        "running a search (use --note to describe it)")
    p.add_argument("--out", type=Path,
                   help="write the trace here instead (single-task runs)")
    p.add_argument("--name",
                   help="name recorded inside the trace, and its filename")
    p.add_argument("--note",
                   help="one line on what this trace is for; the viewer's "
                        "picker shows it under the name")
    p.add_argument("--sweep", type=int, metavar="SEEDS",
                   help="print every method's solve rate on every task (or "
                        "--task) over this many seeds of the offline policy, "
                        "and write nothing")

    g = p.add_argument_group("search")
    g.add_argument("--n", type=int, help="samples per expansion (paper: 5)")
    g.add_argument("--w", type=float, help="exploration weight in UCT (paper: 1)")
    g.add_argument("--lambda", type=float, dest="lambda",
                   help="weight on LLM self-evaluation")
    g.add_argument("--iterations", type=int, help="search iterations")
    g.add_argument("--max-depth", type=int, dest="max_depth", help="depth limit")
    g.add_argument("--seed", type=int, help="seed for the mock policy")
    g.add_argument("--no-simulate", action="store_true",
                   help="skip the simulation step")
    g.add_argument("--no-reflect", action="store_true",
                   help="skip the reflection step")
    g.add_argument("--evaluate-before", action="store_true",
                   help="ablation: take LM(s) from the policy's grade before "
                        "the action ran, as Tree of Thoughts does")
    g.add_argument("--method", default="lats", choices=list(METHOD_NAMES),
                   help="lats (default), or a baseline to compare it with: "
                        "react, reflexion, tot")
    g.add_argument("--beam", type=int,
                   help="states kept per depth by --method tot (default 2)")

    m = p.add_argument_group("policy")
    m.add_argument("--llm", default="mock", choices=list(POLICIES),
                   help="mock is deterministic and offline (default); "
                        "openai calls a real model")
    m.add_argument("--model", default=None,
                   help="model id for --llm openai "
                        f"(default: {OpenAILLM.default_model})")

    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    # A key in .env beats having to export one every session; anything already
    # exported still wins. Names only in the message - never the value.
    from_env_file = load_env(ROOT / ".env")
    if from_env_file and args.llm != "mock" and not args.quiet:
        print(f"read {', '.join(from_env_file)} from .env")

    if args.list:
        print("tasks:")
        for task_id, cls in sorted(TASKS.items()):
            print(f"  {task_id:<20} {cls.title}")
        print("\nmethods: " + ", ".join(METHOD_NAMES))
        print("\npresets (written by a bare `python scripts/run_lats.py`):")
        for preset in PRESETS:
            extra = (
                ", ".join(f"{k}={v}" for k, v in preset["overrides"].items())
                or "task defaults"
            )
            print(f"  {preset['name']:<32} {preset['task']}  "
                  f"{preset.get('method', 'lats')}  [{extra}]")
        return 0

    if args.promote:
        return promote(args.promote, args.name, args.note)

    if args.sweep:
        order = list(dict.fromkeys(p["task"] for p in PRESETS))
        return sweep(args.sweep, [args.task] if args.task else order)

    if args.out and not args.task:
        raise SystemExit("--out applies to a single-task run; add --task.")

    if args.task:
        task = TASKS.get(args.task)
        if task is None:
            raise SystemExit(
                f"unknown task {args.task!r}. Known: {', '.join(sorted(TASKS))}"
            )
        # A run that is not plain LATS says what it is in its name, unless
        # it was given one: `--method react` writes <policy>_<task>_react.
        variant = (args.method if args.method != "lats"
                   else "value-before" if args.evaluate_before else None)
        base = args.name or (f"{args.task}_{variant}" if variant else args.task)
        name = trace_name(base, args.llm)
        out = args.out or destination(args) / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not args.quiet:
            print(f"running {args.task} with {args.method} and the {args.llm} policy")
        entry = run_one(
            args.task,
            build_config(task(), args),
            name=name,
            out=out,
            llm=args.llm,
            model=args.model,
            method=args.method,
            quiet=args.quiet,
        )
        # Keep the folder's index in step with what is on disk, so a hand-run
        # trace shows up in the picker next to the bundled ones.
        if args.note:
            entry["note"] = args.note
        kept = [refresh(e, out.parent) for e in read_manifest(out.parent)
                if e["file"] != entry["file"]]
        write_manifest(out.parent, preset_order(kept + [entry]))
        if not args.quiet:
            report(out.parent, 1, args)
        return 0

    folder = destination(args)
    if not args.quiet:
        print(f"running {len(PRESETS)} presets with the {args.llm} policy\n")
    entries = []
    for preset in PRESETS:
        task = TASKS[preset["task"]]()
        cfg = replace(build_config(task, args), **preset["overrides"])
        entry = run_one(
            preset["task"],
            cfg,
            name=trace_name(preset["name"], args.llm),
            out=folder / f"{trace_name(preset['name'], args.llm)}.json",
            llm=args.llm,
            model=args.model,
            method=preset.get("method", "lats"),
            quiet=args.quiet,
        )
        entry["note"] = preset["note"]
        entries.append(entry)

    # Keep any trace this run did not write - a published OpenAI trace, say -
    # so regenerating the offline set does not evict it from the picker. Stale
    # entries whose file is gone are dropped on the way past, and the rest are
    # re-read so their entries carry whatever this version of the index adds.
    written = {e["file"] for e in entries}
    kept = [refresh(e, folder) for e in read_manifest(folder)
            if e["file"] not in written and (folder / e["file"]).is_file()]
    write_manifest(folder, preset_order(kept + entries))
    if not args.quiet:
        report(folder, len(entries), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
