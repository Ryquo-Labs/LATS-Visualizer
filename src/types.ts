/**
 * The `lats-trace/1` schema, in TypeScript.
 *
 * This mirrors `scripts/run_lats/trace.py`. If you change the schema on the Python
 * side, change it here and in `lib/validate.ts` - those three files are the
 * whole contract between the two halves of the demo.
 */

/** The six operations of the paper, plus a bookend at each end. */
export type Op =
  | 'init'
  | 'selection'
  | 'expansion'
  | 'evaluation'
  | 'simulation'
  | 'backpropagation'
  | 'reflection'
  | 'result'

export const OPS: Op[] = [
  'selection',
  'expansion',
  'evaluation',
  'simulation',
  'backpropagation',
  'reflection',
]

/**
 * Which algorithm wrote the trace. The baselines reuse LATS's operation names
 * for the parts they share, so any of them replays in the same viewer - see
 * `scripts/run_lats/baselines.py` and `lib/methods.ts`. A trace with no
 * `method` predates them and is LATS.
 */
export type Method = 'lats' | 'react' | 'reflexion' | 'tot'

/** A node's immutable half. Written once, when the node is created. */
export interface TraceNode {
  id: number
  parent: number | null
  depth: number
  label: string
  action: string | null
  observation: string | null
  detail: Record<string, unknown>
  terminal: boolean
  /** Index of the step at which this node first exists. */
  created_at: number
}

/** A node's mutable half. Snapshotted on every step. */
export interface NodeState {
  visits: number
  /**
   * V(s). `null` before the node has been evaluated - every new child between
   * its expansion and its evaluation - and throughout a method that keeps no
   * value at all (ReAct, Reflexion).
   */
  value: number | null
  reward: number | null
  /** LM(s): the value function's grade of the state, after the environment replied. */
  lm: number | null
  /** SC(s): the share of independent samples that reached this state. */
  sc: number | null
  /** The policy's grade of its own move, before it ran. Not part of V(s). */
  prior?: number | null
  terminal: boolean
  reflected: boolean
  /** Tree of Thoughts: dropped by the beam, never reconsidered. */
  pruned?: boolean
}

export interface Step {
  index: number
  op: Op
  iteration: number
  title: string
  summary: string
  detail: Record<string, unknown>
  /** Nodes this step is about. */
  focus: number[]
  /** The root-to-leaf path this step walked. */
  path: number[]
  tokens: number
  state: Record<string, NodeState>
}

export interface TaskBrief {
  id: string
  family: string
  title: string
  prompt: string
  reward: string
  context: Record<string, string[]>
}

export interface TraceConfig {
  n: number
  w: number
  lambda: number
  iterations: number
  max_depth: number
  simulate: boolean
  reflect: boolean
  reflect_threshold: number
  solved_at: number
  seed: number | null
  /**
   * When LM(s) was taken: `after` the environment replied (LATS), `before` it
   * (Tree of Thoughts, and the ablation), or `none` for a method with no value
   * function. Absent from a trace written by version 1.0, which scored before.
   */
  evaluation?: 'after' | 'before' | 'none'
  /** Tree of Thoughts: states kept per depth. */
  beam?: number
}

export interface PolicyInfo {
  kind: string
  name: string
  model: string | null
  seed: number | null
  calls: number
  tokens: number
  tokens_are_estimated: boolean
  /** How the n samples of one expansion were drawn. Absent before version 1.1. */
  sampling?: string
}

export interface TraceResult {
  solved: boolean
  best_reward: number | null
  best_node: number | null
  best_path: number[]
  nodes: number
  iterations_run: number
  stopped_because: string
  reflections: string[]
  /** Tree of Thoughts: correct answers it generated and did not return. */
  missed?: number[]
}

export interface Trace {
  schema: string
  name: string
  method?: Method
  generated_by: { package: string; version: string }
  task: TaskBrief
  config: TraceConfig
  policy: PolicyInfo
  result: TraceResult
  nodes: TraceNode[]
  steps: Step[]
}

// -- per-operation `detail` payloads ---------------------------------------

export interface UctEntry {
  id: number
  label: string
  visits: number
  /** V(s), the exploitation half. */
  exploit: number
  /** w * sqrt(ln N(p) / N(s)), the exploration half, at the trace's own w. */
  explore: number
  uct: number
  /** False for a branch that is terminal or at the depth limit. */
  available: boolean
  chosen: boolean
}

export interface UctLevel {
  parent: number
  parent_visits: number
  w: number
  children: UctEntry[]
}

export interface SelectionDetail {
  /** `beam` is Tree of Thoughts keeping its best states; absent means UCT. */
  kind?: 'uct' | 'beam'
  levels: UctLevel[]
  exhausted: boolean
  target?: number
}

/** Tree of Thoughts: the cut at one depth. */
export interface BeamDetail {
  kind: 'beam'
  beam: number
  depth: number
  ranked: { id: number; label: string; value: number; kept: boolean; answer: boolean }[]
  kept: number[]
  dropped: number[]
}

export interface ExpansionChild {
  id: number | null
  label: string
  text?: string
  observation?: string
  detail?: Record<string, unknown>
  terminal?: boolean
  reward?: number | null
  /** How many of the n samples reached this state. */
  samples?: number
  /** The policy's grade of the move, before it ran. */
  prior?: number
  /** Version 1.0 graded at expansion, and recorded these here. */
  lm?: number
  sc?: number
  value?: number
  rejected: string | null
}

export interface ExpansionDetail {
  parent: number
  n: number
  /**
   * How many of the n samples held a usable action, when fewer than n did -
   * a real model's reply can be cut off or unparseable. SC(s) is a share of
   * these, not of n. Absent when every sample was usable.
   */
  usable?: number
  children: ExpansionChild[]
  reflections_in_context?: string[]
}

/** One state's grade. Shared by the evaluation step and a rollout's. */
export interface ScoreEntry {
  id: number
  label: string
  /** The policy's grade from before the move ran; absent before version 1.1. */
  prior?: number
  lm: number
  sc: number
  value: number
  reward: number | null
  /** The grader's one-line reason. */
  why?: string
}

export interface EvaluationDetail {
  lam: number
  /** When LM(s) was taken. Absent before version 1.1, which took it before. */
  timing?: 'after' | 'before'
  scores: ScoreEntry[]
}

export interface SimulationDetail {
  rollout: number[]
  terminal: number
  truncated: boolean
  skipped?: boolean
  observation: string
  /** Every state the rollout created, graded after its reply. */
  graded?: ScoreEntry[]
}

export interface BackpropDetail {
  reward: number
  leaf: number
  updates: {
    id: number
    before: { visits: number; value: number }
    after: { visits: number; value: number }
  }[]
}

export interface ReflectionDetail {
  node: number
  text: string
  trajectory: number[]
  reward: number
  total_notes: number
}

export interface InitDetail {
  task: TaskBrief
  config: TraceConfig
}

// -- the traces folder index ------------------------------------------------

export interface ManifestEntry {
  file: string
  name: string
  task: string
  title: string
  family: string
  method?: Method
  solved: boolean
  best_reward: number | null
  nodes: number
  steps: number
  policy: string
  /** Written by version 1.0: scored before execution. */
  legacy?: boolean
  note?: string
}

export interface Manifest {
  schema: string
  traces: ManifestEntry[]
}
