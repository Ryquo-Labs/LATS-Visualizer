# Language Agent Tree Search (LATS) Visualizer

A trace generator and visualizer for *Language Agent Tree Search* (LATS; Zhou et al., ICML 2024 [arXiv:2310.04406](https://arxiv.org/abs/2310.04406)). `scripts/run_lats.py` searches a real
environment and records what it did as an **execution trace**; the web app at the root of this
repository replays that trace one operation at a time, the tree growing as the
search grows, with the logic and arithmetic behind each operation visualized as well.
The generator also runs ReAct, Reflexion and Tree of Thoughts for side-by-side
comparison, and the viewer's play mode lets a student make LATS's judgements.

> **The default path runs offline with no API key.** The default policy is a
> seeded stand-in, so the same command always writes the same trace, byte for
> byte. `--llm openai` swaps in a real model when you want one.
> What is *not* mocked is the environment: the candidate programs are executed,
> the arithmetic is evaluated, the documents are retrieved. Every reward in
> every bundled trace came out of a real execution.

---

## Quick start

```
# The viewer - this alone needs no Python
npm install
npm run dev                       # open the URL specified

# The search
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt   # only the real model and the video need these

python scripts/run_lats.py        # search run, saved in results/lats-traces/
```

The viewer opens on the traces committed in `public/traces/`, so it has something
to show before you have run anything. The default `--llm mock` policy imports
nothing beyond the standard library, so a search runs before you install anything.

---

## The search

```
python scripts/run_lats.py                     # every bundled preset
python scripts/run_lats.py --list              # tasks, methods and presets
python scripts/run_lats.py --task game-of-24   # one task, its own defaults
python scripts/run_lats.py --task game-of-24 --w 0 --seed 3 --name game-of-24_greedy
python scripts/run_lats.py --task merge-intervals --method tot   # a baseline
python scripts/run_lats.py --sweep 60          # every method, sixty seeds, a table
python scripts/run_lats.py --publish           # refresh public/traces/
```

| flag                | meaning                                                         | paper's value                                           |
| ------------------- | --------------------------------------------------------------- | ------------------------------------------------------- |
| `--n`               | independent samples per expansion                               | 5                                                       |
| `--w`               | exploration weight in UCT                                       | 1                                                       |
| `--lambda`          | weight on the value function's grade, against self-consistency | 0.5 (HotpotQA, Game of 24) · 0.8 (programming, WebShop) |
| `--iterations`      | search iterations                                               | 30–50 trajectories                                      |
| `--max-depth`       | hard depth limit                                                | 6–7                                                     |
| `--seed`            | seed for the offline policy                                     | —                                                       |
| `--no-simulate`     | skip simulation                                                 | the paper does this for programming                     |
| `--no-reflect`      | skip reflection                                                 | ablation                                                |
| `--evaluate-before` | take LM(s) from the policy's grade *before* the action ran      | ablation: the Tree-of-Thoughts ordering                 |
| `--method`          | `lats` (default), `react`, `reflexion` or `tot`                 | —                                                       |
| `--beam`            | states Tree of Thoughts keeps per depth (default 2)             | —                                                       |
| `--llm`             | `mock` (default) or `openai`                                    | —                                                       |

Task defaults come first and flags override them, so `--task merge-intervals` picks
up λ = 0.8 and no simulation. Each run writes a timestamped directory under
`results/lats-traces/`; drop any file in it onto the viewer to step through it. A
trace name is `<policy>_<task>_<variant>` — `mock_game-of-24_no-value` is the offline
policy on Game of 24 with the value function ablated — and the prefix comes from
`--llm`, so a trace cannot be mislabelled.

`--publish` regenerates the offline set in `public/traces/` and leaves anything else
there alone. A real-model trace can be published as it runs, or promoted afterwards
without paying for it twice:

```
python scripts/run_lats.py --task game-of-24-hard --llm openai --publish \
    --note "what this trace is for; the picker shows it under the name"
python scripts/run_lats.py --promote results/lats-traces/<run>/traces/<trace>.json
```

### How a node is valued

`V(s) = λ·LM(s) + (1 − λ)·SC(s)`, and the two halves come from different calls:

- **SC(s) is counted at expansion.** `n` independent samples are drawn from the same
  state and grouped by the state they reach; a child's `SC(s)` is its share of the
  usable samples (self-consistency, Wang et al., ICLR 2023).
- **LM(s) is graded after the environment replies.** Expansion executes every child
  and grades none; the evaluation step then calls the value function once per child,
  with the reply — the test report, the retrieved page, the numbers left — in the
  prompt. That ordering is the paper's stated difference from Tree of Thoughts.

The policy's own grade of a move from before it ran, its **prior**, is recorded so
the viewer can show what the observation changed (`prior 0.70 → LM 0.63` for a
program that passed three tests of five). It enters `V(s)` only under
`--evaluate-before` and in Tree of Thoughts. Offline, `LM(s)` is a deliberately
fallible per-task heuristic that reads the same reply.

### The four environments

**`merge-intervals`** — write `merge(intervals)` against five visible tests. Reward
is the fraction that pass, measured by running them in a child process. As in the
paper's programming setting, every node is a complete program, so simulation is
skipped. The example is a trap: the obvious one-pass sweep looks best before anything
runs and caps at three of five, as does every refinement of it. The fix is a
different branch.

**`game-of-24`** — make 24 from `2, 5, 8, 11`, each number used once. Reward is 1 or
0 and needs no oracle. The offline policy scores a move by how *tidy* the result
looks, which is plausible and frequently wrong; that gap is what backpropagation
exists to close.

**`game-of-24-hard`** — the same on `6, 9, 9, 10`, the four-number puzzle that
punishes a value function hardest. Every solution ends `9 + 15`, and 15 is reachable
only through a fraction or `9 * 10 = 90`. The heuristic's **twelve best-looking
first moves are all dead ends**; at rank five, `6 + 9 = 15` makes the number every
solution needs and still loses, because it spends the 9 the last step requires.
Five first moves do work.

**`multihop-qa`** — a ReAct-shaped loop (`search[term]`, `finish[answer]`) over a
small corpus. The question asks which venue published the paper that *introduced*
the algorithm LATS adapts; the corpus also holds a 2006 paper about that algorithm's
selection *rule*, published elsewhere, and the policy's recency bias walks straight
into it.

### The baselines

`--method` runs ReAct, Reflexion or Tree of Thoughts on the same task, policy and
depth limit. Each is written from its paper (`scripts/run_lats/baselines.py`) and
records the same trace format, so the viewer shows it as LATS with operations taken
away:

| method                                           | selection | expansion | evaluation       | simulation | backprop | reflection |
| ------------------------------------------------ | --------- | --------- | ---------------- | ---------- | -------- | ---------- |
| ReAct (Yao et al., ICLR 2023)                    | —         | n = 1     | —                | —          | —        | —          |
| Reflexion (Shinn et al., NeurIPS 2023)           | —         | n = 1     | —                | —          | —        | yes        |
| Tree of Thoughts, BFS (Yao et al., NeurIPS 2023) | beam      | n         | before the reply | —          | —        | —          |
| LATS                                             | UCT       | n         | after the reply  | yes        | yes      | yes        |

Reflexion reruns ReAct from the root with a note from each failed trial. Tree of
Thoughts keeps the `b` states its prior rates best at each depth; each step is still
executed so the next state exists, but nothing it keeps depends on the reply.

One trace is one draw. `--sweep 60` prints solved out of sixty seeds of the offline
policy, mean actions executed in brackets:

| task              | LATS    | LATS, value before | ReAct  | Reflexion | ToT    |
| ----------------- | ------- | ------------------ | ------ | --------- | ------ |
| `game-of-24`      | 14 (52) | 14 (53)            | 0 (3)  | 7 (35)    | 8 (18) |
| `merge-intervals` | 23 (6)  | 16 (5)             | 7 (2)  | 25 (9)    | 3 (6)  |
| `multihop-qa`     | 51 (16) | 50 (16)            | 9 (3)  | 41 (15)   | 9 (9)  |
| `game-of-24-hard` | 17 (59) | 17 (59)            | 0 (3)  | 10 (45)   | 0 (18) |

The baselines are not straw men: Reflexion edges LATS on `merge-intervals`, where one
note ("sort first") is most of the answer, and LATS's lead elsewhere is bought with
more actions. This is the offline policy on four small tasks, not a benchmark.

### The traces the viewer ships with

The picker groups them by environment, the offline policy before a real model and
LATS before the baselines. `--publish` regenerates the `mock_` set byte for byte; the
`openai_` set came from a real model and is not reproducible, which is why it is
checked in.

| trace                               | method    | policy | solved | nodes | steps | what it is for                                              |
| ----------------------------------- | --------- | ------ | ------ | ----- | ----- | ----------------------------------------------------------- |
| `mock_game-of-24`                   | LATS      | mock   | yes    | 45    | 31    | all six operations, including a real rollout                |
| `mock_game-of-24_no-value`          | LATS      | mock   | yes\*  | 68    | 74    | ablation: λ = 0, self-consistency only                      |
| `mock_game-of-24_greedy`            | LATS      | mock   | **no** | 64    | 74    | ablation: w = 0, exploitation only                          |
| `mock_game-of-24_react`             | ReAct     | mock   | **no** | 4     | 5     | the right first move, then one wrong one, and no way back   |
| `mock_game-of-24_reflexion`         | Reflexion | mock   | **no** | 37    | 50    | a note that blames the one move that was right              |
| `mock_game-of-24_tot`               | ToT       | mock   | **no** | 21    | 13    | the winning branch dropped at the first cut                 |
| `openai_game-of-24`                 | LATS      | gpt-5  | yes    | 5     | 7     | a strong policy needs no search; SC(s) breaks a tie         |
| `mock_merge-intervals`              | LATS      | mock   | yes    | 7     | 19    | the programming setting; the tests flip the model's ranking |
| `mock_merge-intervals_value-before` | LATS      | mock   | yes    | 5     | 13    | ablation: LM(s) from before the tests ran - version 1.0     |
| `mock_merge-intervals_react`        | ReAct     | mock   | **no** | 3     | 4     | the favourite approach, patched, at three of five           |
| `mock_merge-intervals_reflexion`    | Reflexion | mock   | yes    | 5     | 7     | one note - "sort first" - and the second trial passes       |
| `mock_merge-intervals_tot`          | ToT       | mock   | **no** | 7     | 9     | writes the correct program, grades it 0.64, and drops it    |
| `mock_multihop-qa`                  | LATS      | mock   | yes    | 14    | 13    | two-hop retrieval with a distractor                         |
| `mock_multihop-qa_no-reflection`    | LATS      | mock   | yes    | 14    | 12    | ablation: reflection off                                    |
| `mock_multihop-qa_react`            | ReAct     | mock   | **no** | 4     | 5     | reads the right page, answers from the wrong one            |
| `mock_multihop-qa_reflexion`        | Reflexion | mock   | **no** | 28    | 37    | eight accurate notes, none of which says where to look      |
| `mock_multihop-qa_tot`              | ToT       | mock   | **no** | 17    | 16    | right and wrong answer graded alike; returns the wrong one  |
| `openai_multihop-qa`                | LATS      | gpt-5  | yes    | 4     | 7     | eight wordings, three actions: SC(s) counts what they do    |
| `mock_game-of-24-hard`              | LATS      | mock   | **no** | 75    | 98    | the hard puzzle, and why more search does not help          |
| `mock_game-of-24-hard_reflexion`    | Reflexion | mock   | yes    | 16    | 21    | a seed where starting over beats a root expanded once       |
| `openai_game-of-24-hard`            | LATS      | gpt-5  | yes    | 7     | 7     | the grade after the reply picks the one opening that works  |
| `openai_game-of-24-hard_wide`       | LATS      | gpt-5  | yes    | 7     | 7     | `--n 12`: half the samples pick a dead end; LM(s) does not  |

> \* Solved only in the bookkeeping sense: a rollout *builds* a winning node on
> iteration 1, but selection never returns to it and every iteration backpropagates
> 0. A search can contain an answer it never noticed.
>
> The `openai_` traces are gpt-5 at low reasoning effort on each task's defaults
> (`--n 12` for `_wide`), one draw each. All four are solved by the first rollout: on
> tasks this small gpt-5 needs almost no search, so the dead ends and recoveries are
> in the offline set.

Four things the set shows.

**Grading after the reply matters where the reply carries news.** At the first
evaluation of `mock_merge-intervals` the model prefers the one-pass sweep, 0.70 to
0.62, until the tests run: three of five against four of five flips the order.
Across sixty seeds that is worth 23 solves to 16 (`mock_merge-intervals_value-before`
is the same seed graded from opinion alone), and nothing on Game of 24, where the
reply says nothing the move did not. A real model can still gain from the second
look: in `openai_game-of-24-hard` the grading call marks down two openings gpt-5
proposed at 0.95 and ranks first the one of four that can reach 24.

**Reflection matters least.** Turning it off changes nothing on `multihop-qa`,
matching the paper's ablation (−0.05 exact match, against −0.26 for the value
function and −0.21 for the search).

**Which term dominates depends on the task.** On Game of 24 the ordering is the
reverse of the paper's HotpotQA result: without exploration (`w = 0`) the search
fails, and without the value function's grade (`λ = 0`) it builds the winning node
and walks past it.

**Search cannot repair the policy; sampling and grading can.** `mock_game-of-24-hard`
expands its root into five tidy moves, none of the five that reach 24, and every
later operation runs flawlessly over a tree with no solution in it. gpt-5's
independent samples include a winning fraction at both widths, and the grade after
the reply picks it out: in `openai_game-of-24-hard_wide` half the samples and the
highest prior back the dead end `10 - 9`, and `V(s)` still ranks `10 / 6` first.
`mock_game-of-24-hard_reflexion` gets there by starting over, though across sixty
seeds LATS solves this puzzle more often, 17 to 10.

### The real-model policy

```
cp .env.example .env                    # then fill in OPENAI_API_KEY

export OPENAI_API_KEY=sk-...            # or macOS / Linux
$env:OPENAI_API_KEY = 'sk-...'          # or PowerShell

python scripts/run_lats.py --task game-of-24-hard --llm openai
```

The SDK is in `requirements.txt`; the key comes from the environment or, failing
that, `.env` (gitignored), and only variable *names* are printed. `--model` overrides
`gpt-5`; `OPENAI_BASE_URL` points at a compatible endpoint.

**Proposals are independent samples** of a prompt asking for one action — `n`
choices per request where the endpoint allows it (gpt-5 takes up to 8 at a time),
single requests otherwise. Samples agree when they reach the same state, not when
they are worded alike: the nine samples in `openai_multihop-qa` are eight strings but
three actions. Unusable samples, such as a reply cut off by the token cap, are left
out, and the trace records how many were usable.

**Grading is a second call after execution**, shown the trajectory and the
environment's reply (a `multihop-qa` answer only as accepted or rejected). That adds
a request for every new state, where version 1.0 made one per expansion; the four
bundled gpt-5 traces took 34 requests and about 110,000 tokens, roughly a dollar when
they were made. The environment keeps the last word: an invented step is rejected,
and every result is recomputed rather than taken from the reply.

> **Two warnings.** Traces produced this way are not reproducible. And on
> `merge-intervals` the environment runs whatever program the model wrote, in a
> separate process with a timeout: that bounds a runaway loop, but it is **not** a
> sandbox. The offline policy only proposes programs from `code_task.py`, so this
> applies to `--llm openai` alone.

### Adding your own task

Subclass `run_lats.tasks.base.Task`, implement `root_data`, `step`, `mock_propose`
and `mock_evaluate` (the offline value function, given the state *after* the step),
and register it in `scripts/run_lats/tasks/__init__.py`; nothing else changes.
`render`, `action_schema` and `parse_action` are needed only for `--llm openai`.
Override `signature` when two spellings of an action reach the same state and
`render_observation` to hide something the agent should not see, and set
`complete_states` when every state is a whole answer, as a program is.

---

## The viewer

```
npm run dev              # dev server with hot reload
npm run build            # static bundle into dist/
npm run preview          # serve that bundle
npm run lint             # oxlint
npx tsc -b               # typecheck only
```

React 19 · TypeScript · Vite · MUI. Nothing is loaded from a remote host. Three
modes, from the app bar:

- **Replay** a trace one operation at a time: play, scrub, `←`/`→` to step, space to
pause, `Home`/`End` to jump. Node fill is the value ramp; the ring is the objective
reward.
- **Compare** two traces side by side — LATS against a baseline, or an ablation
against its run. The arrow keys step both, and each pane counts the actions it has
executed, the fair axis for cost.
- **Play** Game of 24 as LATS's judgement: choose the leaf to expand, grade each new
state, write the reflection after a failure. Each choice is shown beside the
algorithm's, a note naming a move (*don't open with 8 / 2*) stops the policy
proposing it, and the end reports how well your grades separated solvable states
from dead ends. Any judgement can be handed back to the algorithm, and the session
is a trace: download it, or compare it with LATS's own.

In every mode the viewer:

- **Frames the step, not the whole tree.** The camera follows the operation; pan or
zoom to take over, the crosshair to hand back. Zooming out drops detail in stages.
- **Shows the arithmetic.** Selection draws UCT as stacked bars, exploitation against
exploration, with a live `w` slider that can flip the winner. Evaluation draws
`V(s)` the same way, each child's prior beside its post-reply grade and the grader's
reason. Backpropagation lists every ancestor's value before and after. A baseline's
missing operations are struck through.
- **Labels a simplified run.** A version 1.0 trace, whose `LM(s)` was taken before
execution, carries a banner, a `v1.0 scoring` chip and a warning at every evaluation.
- **Opens any node** for the action, the reply, the grade and its reason, the
program's test results, or the retrieved document.
- **Takes your own traces**, dropped on the window or uploaded. Nothing leaves the
browser, and a rejected file gets every problem found, each with a path into the
document.
- **Reads in the dark**, with hues re-picked for a dark card so the 4.5:1 contrast
holds; the choice is remembered, never guessed.
- **Fits a phone**: side by side above about 1100px, two tabs over a shared transport
on a handset.

**Changing it.** `src/theme.ts` holds both palettes, the type scale and the value
ramp; the light one mirrors `scripts/create_video/theme.py`, so change them together.
New colours go in the `Token` union and both palettes — never as a literal in a
component, and through `style` rather than `fill` in SVG. A new operation needs `Op`
and `OPS` in `types.ts`, `OP_COLOR`, a `case` in `OperationPanel` and `VALID_OPS` in
`lib/validate.ts`; a new method, an entry in `METHODS` in `lib/methods.ts`. Play mode
lives in `src/play/`.

---

## The video

`scripts/create_video.py` renders a 17-minute Manim explainer of the algorithm in
six parts, from source.

```
python scripts/create_video.py                  # draft (854x480), for iterating
python scripts/create_video.py --quality final  # delivery (1920x1080, 30 fps)
python scripts/create_video.py --quality 4k     # 4K master (3840x2160, 30 fps)
python scripts/create_video.py --parts 3 5      # re-render two parts, then re-join
python scripts/create_video.py --check          # report the toolchain
```

Rendering needs `requirements.txt`, plus **ffmpeg** and a **LaTeX** distribution on
`PATH`; `--check` reports what it can find. Every render gets its own timestamped
directory under `results/video/`, holding one file per part, the concatenated
`full.mp4`, per-beat timings, and `SCRIPT.md`.

**The video carries no sound.** The narration lives in a `NARRATION` dict at the top
of each part module, next to the animation it describes, with an `ON_SCREEN` dict
beside it. `SCRIPT.md` is generated from those two dicts plus the timings the render
actually measured, so the script always describes the mp4 sitting beside it.
`--timing` compares words against screen time, beat by beat. The holds and run times
are matched, to the frame, to the recorded narration, so the voice-over drops
straight onto a fresh `full.mp4` - change one and every later cue moves with it.

**The type is bundled, not installed.** `create_video/fonts/` holds Inter and
JetBrains Mono, and `create_video/fontpath.py` registers them with Pango on import -
privately, for the rendering process only. Build text with `txt()` and `mono()` from
`theme.py`, not Manim's `Text`: Manim has Pango lay text out at a few points, where
Pango rounds every letter to a whole unit and the spacing comes out visibly uneven,
so those two lay it out 32x larger and scale it back. On Windows, MiKTeX is usually
installed per-user and is missing from the PATH a virtualenv sees;
`create_video/texpath.py` finds it and prepends it automatically, and `LATS_TEX_BIN`
overrides the search.

---

## The trace format

`lats-trace/1`. Written by `scripts/run_lats/trace.py` and play mode, typed in
`src/types.ts`, validated in `src/lib/validate.ts`.

```
{
  "schema": "lats-trace/1",
  "name":   "mock_game-of-24",
  "method": "lats",                     // or react | reflexion | tot
  "generated_by": { "package", "version" },
  "task":   { "id", "family", "title", "prompt", "reward", "context" },
  "config": { "n", "w", "lambda", "iterations", "max_depth", "simulate",
              "reflect", "reflect_threshold", "solved_at", "seed",
              "evaluation", "beam" },  // evaluation: after | before | none
  "policy": { "kind", "name", "model", "seed", "calls", "tokens",
              "tokens_are_estimated", "sampling" },
  "result": { "solved", "best_reward", "best_node", "best_path", "nodes",
              "iterations_run", "stopped_because", "reflections", "missed" },

  "nodes": [{ "id", "parent", "depth", "label", "action", "observation",
              "detail", "terminal", "created_at" }],

  "steps": [{ "index", "op", "iteration", "title", "summary", "detail",
              "focus", "path", "tokens", "state" }]
}
```

The split is the design. A node's parent, action and observation never change, so
they are written once; its visits, value, reward, `lm`, `sc` and `prior` change
constantly, so **every step carries a full snapshot** in `state` and the viewer can
jump to any step. A `null` value means not graded yet, or a method with no value
function. `detail` is per-operation: the UCT table or beam cut, each child's sample
count (and `usable`, when fewer than `n` held an action), each grade and its reason,
every before/after backup. A trace without `method`, `config.evaluation` and
`policy.sampling` was written by version 1.0.

---

## Where this follows the paper, and where it does not

**Follows.** The six operations and their order.
`UCT(s) = V(s) + w·√(ln N(p) / N(s))` with `N` initialised to 1.
`V(s) = λ·LM(s) + (1 − λ)·SC(s)`, with `LM(s)` from a separate call *after* the
environment has replied and `SC(s)` counted over `n` independent samples.
Reflection notes in the context of both the policy and the value function.
Simulation as a greedy value-guided descent in the real environment, not a random
playout. Per-family λ. Skipping simulation in the programming setting and
backpropagating the fraction of tests passed.

**The backup rule.** Section 4.2 prints it with subscripts that do not match the
paper's own pseudocode. `scripts/run_lats/search.py` follows Algorithm 1 and
Section 3.2:

```
N(s) ← N(s) + 1
V(s) ← ( V_old(s)·(N(s) − 1) + r ) / N(s)
```

Appendix C also reports depth `d = 7` while Appendix D.1 says a maximum depth limit
of 6.

**Deliberate departures.** Smaller — `n = 3` or `5` and 6–16 iterations against the
paper's 30–50 trajectories, so traces fit on a screen. A two-level programming tree
whose children refine their parent's program, so every node stays a complete
program. One backup per iteration, short-circuiting to a candidate that has already
solved the task. A stand-in policy, `MockLLM`, labelled as such in every trace, with
token counts flagged as estimates. And **template reflections** under both policies:
`reflection_for` writes the note from the environment's output, where the paper asks
a model; play mode is where a person writes it.

**Version 1.0, and what changed.** Version 1.0 took `LM(s)` from the policy's grade
while proposing, before anything ran, and asked a real model for `n` *distinct*
actions, so samples could not agree: 176 of 180 children in its gpt-5 traces had one
sample, against 8 of 11 with two or more in the re-recorded ones. `--evaluate-before`
reproduces the old ordering — `mock_merge-intervals_value-before` is the 1.0
`mock_merge-intervals`, node for node — and 1.0 traces still load, labelled.

**Honest caveats.** The QA reward is an oracle, a stored gold string, as it is in the
paper's HotpotQA setting and the baselines it compares against. The offline graders
are heuristics. And `tasks/sandbox.py` bounds runtime; it does not contain hostile
code.

---

## Reproducing

`python scripts/run_lats.py --publish` is deterministic: same seeds, same traces,
byte for byte, and no timestamp in the manifest. `--sweep 60` prints the seed table.
Every number in these tables comes from one of the two; none is typed by hand. Play
mode is seeded with its own sampler, so it does not reproduce `run_lats.py`'s traces.

## Licensing and originality

All material here is original work. `scripts/run_lats/` is a from-scratch reading of
the paper's Algorithm 1, and `baselines.py` of the ReAct, Reflexion and Tree of
Thoughts papers — it vendors nothing from the authors' repositories, from
LangGraph, or from any agent framework. The video imports no external image, icon or
font. The viewer depends on React, MUI and Vite, all MIT licensed, and loads no
font, script or image from a remote host. See [LICENSE](LICENSE).
