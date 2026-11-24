/**
 * A LATS search you take part in.
 *
 * The same loop as `scripts/run_lats/search.py`, cut at every point where the
 * algorithm makes a judgement so a student can make it instead:
 *
 *     selection        you pick the leaf to expand - or let UCT
 *     expansion        the offline policy samples n moves; the arithmetic runs
 *     evaluation       you grade each new state - or let the model
 *     simulation       a greedy rollout, graded by the model as it goes
 *     backpropagation  the reward travels up; nothing to decide
 *     reflection       you write the note - or take the template
 *
 * Every step is recorded exactly as the Python recorder records it, with a full
 * snapshot of the tree, so the session *is* a `lats-trace/1` trace: the viewer's
 * own tree, panels and timeline render it without knowing it was played, and it
 * can be downloaded and compared side by side with the search LATS ran alone.
 * Where a student made the call, the step says so (`by: 'you'`), along with
 * what the algorithm would have done instead.
 */

import type { NodeState, Step, Trace, UctEntry, UctLevel } from '../types'
import {
  Frac,
  TARGET,
  type Move,
  modelGrade,
  movesNamed,
  propose,
  round,
  seeded,
  solvable,
} from './game24'

export type Phase = 'select' | 'expand' | 'evaluate' | 'simulate' | 'backprop' | 'reflect' | 'done'

export interface PlayConfig {
  numbers: number[]
  n: number
  w: number
  lambda: number
  iterations: number
  seed: number
}

export const PUZZLES: { label: string; numbers: number[]; bundled: string }[] = [
  { label: '2 5 8 11', numbers: [2, 5, 8, 11], bundled: 'mock_game-of-24.json' },
  { label: '6 9 9 10 (hard)', numbers: [6, 9, 9, 10], bundled: 'mock_game-of-24-hard.json' },
]

interface PNode {
  id: number
  parent: number | null
  depth: number
  label: string
  action: string | null
  observation: string | null
  detail: Record<string, unknown>
  terminal: boolean
  created_at: number
  numbers: Frac[]
  steps: string[]
  visits: number
  value: number | null
  reward: number | null
  lm: number | null
  sc: number | null
  prior: number | null
  reflected: boolean
  children: number[]
  /** The student's grade, if they gave one: kept for the calibration report. */
  yours?: number
}

/** One grade the student gave, and what the state turned out to be. */
export interface Calibration {
  id: number
  label: string
  yours: number
  model: number
  reachable: boolean
}

const MAX_DEPTH = 3

export class Session {
  readonly cfg: PlayConfig
  private rng: () => number
  private nodes: PNode[] = []
  private steps: Step[] = []
  private notes: string[] = []
  private blocked = new Set<string>()
  private calls = 0
  phase: Phase = 'select'
  iteration = 1

  /** Set between phases: what the next one acts on. */
  target: number | null = null
  fresh: number[] = []
  private leaf: number | null = null
  private truncated = false
  /** What UCT would have picked at the last selection, for feedback. */
  uctPick: number | null = null

  constructor(cfg: PlayConfig) {
    this.cfg = cfg
    this.rng = seeded(cfg.seed)
    const numbers = cfg.numbers.map((v) => new Frac(v))
    this.add({
      parent: null,
      depth: 0,
      label: 'root',
      action: null,
      observation: null,
      detail: {},
      terminal: false,
      numbers,
      steps: [],
      value: 0,
    })
    this.emit('init', 0, 'Root', `${this.title}. You make the calls; the arithmetic is real.`, {
      task: this.brief(),
      config: this.configDict(),
    }, [0], [0])
  }

  // -- reading the tree ---------------------------------------------------

  get title() {
    return `Game of 24 on ${this.cfg.numbers.join(', ')}`
  }

  node(id: number) {
    return this.nodes[id]
  }

  /** Can the search still do useful work at or under this node? */
  expandable(id: number): boolean {
    const n = this.nodes[id]
    if (n.terminal || n.depth >= MAX_DEPTH) return false
    return n.children.length === 0 || n.children.some((c) => this.expandable(c))
  }

  /** A leaf the student may choose to expand. */
  choosable(id: number): boolean {
    const n = this.nodes[id]
    return !!n && n.children.length === 0 && !n.terminal && n.depth < MAX_DEPTH
  }

  /** Why a node cannot be chosen, in words, or null if it can. */
  whyNot(id: number): string | null {
    const n = this.nodes[id]
    if (!n) return 'There is no such node.'
    if (n.terminal) return `#${id} is finished: ${n.observation ?? 'no numbers are left to combine.'}`
    if (n.children.length) {
      return `#${id} has already been expanded. LATS expands a node once — pick one of its unexpanded descendants.`
    }
    return null
  }

  private uct(child: PNode, parent: PNode) {
    const explore = this.cfg.w * Math.sqrt(Math.log(parent.visits) / child.visits)
    return { exploit: child.value ?? 0, explore }
  }

  /** The leaf UCT would select, descending from the root. */
  uctTarget(): number | null {
    let node = this.nodes[0]
    while (node.children.length) {
      const options = node.children.filter((c) => this.expandable(c)).map((c) => this.nodes[c])
      if (!options.length) return null
      let best = options[0]
      let bestScore = -Infinity
      for (const c of options) {
        const { exploit, explore } = this.uct(c, node)
        if (exploit + explore > bestScore) {
          best = c
          bestScore = exploit + explore
        }
      }
      node = best
    }
    return this.choosable(node.id) ? node.id : null
  }

  private pathTo(id: number): number[] {
    const out: number[] = []
    let n: PNode | undefined = this.nodes[id]
    while (n) {
      out.push(n.id)
      n = n.parent === null ? undefined : this.nodes[n.parent]
    }
    return out.reverse()
  }

  // -- 1. selection -------------------------------------------------------

  select(id: number, by: 'you' | 'uct'): void {
    if (this.phase !== 'select') throw new Error('not selecting')
    if (!this.choosable(id)) throw new Error(this.whyNot(id) ?? 'not choosable')
    const uct = this.uctTarget()
    this.uctPick = uct
    const path = this.pathTo(id)
    const levels: UctLevel[] = []
    for (let i = 0; i + 1 < path.length; i++) {
      const parent = this.nodes[path[i]]
      levels.push({
        parent: parent.id,
        parent_visits: parent.visits,
        w: this.cfg.w,
        children: parent.children.map((cid): UctEntry => {
          const c = this.nodes[cid]
          const { exploit, explore } = this.uct(c, parent)
          return {
            id: c.id,
            label: c.label,
            visits: c.visits,
            exploit: round(exploit),
            explore: round(explore),
            uct: round(exploit + explore),
            available: this.expandable(c.id),
            chosen: c.id === path[i + 1],
          }
        }),
      })
    }
    const label = this.nodes[id].label
    const summary =
      by === 'uct'
        ? `UCT chooses #${id} ${label}.`
        : uct === id
          ? `You chose #${id} ${label} — the leaf UCT would have chosen too.`
          : `You chose #${id} ${label}; UCT would have chosen ${uct === null ? 'nothing' : `#${uct} ${this.nodes[uct].label}`}.`
    this.target = id
    this.emit('selection', this.iteration, 'Selection', summary, {
      levels,
      exhausted: false,
      target: id,
      by,
      uct_target: uct,
    }, [id], path)
    this.phase = 'expand'
  }

  // -- 2. expansion -------------------------------------------------------

  expand(): void {
    if (this.phase !== 'expand' || this.target === null) throw new Error('not expanding')
    const parent = this.nodes[this.target]
    const children = this.sampleChildren(parent)
    this.fresh = children
    const records = children.map((id) => {
      const c = this.nodes[id]
      return {
        id,
        label: c.label,
        text: c.action ?? '',
        observation: c.observation ?? '',
        detail: c.detail,
        terminal: c.terminal,
        reward: c.reward,
        samples: Math.round((c.sc ?? 0) * this.cfg.n),
        prior: c.prior ?? 0,
        rejected: null,
      }
    })
    this.emit('expansion', this.iteration, 'Expansion',
      `${this.cfg.n} independent samples from the policy reached ${children.length} distinct state${children.length === 1 ? '' : 's'}, each executed.`,
      {
        parent: parent.id,
        n: this.cfg.n,
        children: records,
        reflections_in_context: this.notes.map((t) => t),
      },
      children.length ? children : [parent.id],
      this.pathTo(parent.id),
    )
    if (!children.length) {
      parent.terminal = true
      this.nextIteration()
      return
    }
    this.phase = 'evaluate'
  }

  /** Sample n moves, group the ones that agree, execute each group once. */
  private sampleChildren(parent: PNode): number[] {
    const samples = propose(parent.numbers, this.rng, this.cfg.n, this.blocked)
    this.calls += 1
    const groups = new Map<string, { move: Move; priors: number[] }>()
    for (const s of samples) {
      const key = s.move.remaining.join(',')
      const g = groups.get(key)
      if (g) g.priors.push(s.prior)
      else groups.set(key, { move: s.move, priors: [s.prior] })
    }
    const out: number[] = []
    for (const { move, priors } of groups.values()) {
      const steps = [...parent.steps, move.expr]
      const left = move.remaining
      const done = left.length === 1
      const hit = done && left[0].equals(new Frac(TARGET))
      const child = this.add({
        parent: parent.id,
        depth: parent.depth + 1,
        label: move.label,
        action: move.expr,
        observation: done
          ? `${move.expr}. Final value ${left[0]} - ${hit ? 'that is 24.' : 'that is not 24.'}`
          : `${move.expr}. Left to combine: ${left.join(', ')}.`,
        detail: done
          ? { steps, final: String(left[0]), target: TARGET, expression: steps.join(' ; ') }
          : { steps, remaining: left.map(String) },
        terminal: done || parent.depth + 1 >= MAX_DEPTH,
        numbers: left,
        steps,
        value: null,
      })
      child.reward = done ? (hit ? 1 : 0) : null
      child.prior = round(priors.reduce((a, b) => a + b, 0) / priors.length)
      child.sc = round(priors.length / samples.length)
      parent.children.push(child.id)
      out.push(child.id)
    }
    return out
  }

  // -- 3. evaluation ------------------------------------------------------

  /** The model's grade of every fresh child, for "let the model score". */
  modelGrades(): Map<number, number> {
    return new Map(this.fresh.map((id) => [id, this.gradeByModel(id).score]))
  }

  private gradeByModel(id: number) {
    const c = this.nodes[id]
    return modelGrade(c.steps, c.numbers, this.blocked)
  }

  /** Grade the fresh children: with the student's scores, or the model's. */
  evaluate(yours: Map<number, number> | null): void {
    if (this.phase !== 'evaluate') throw new Error('not evaluating')
    const lam = this.cfg.lambda
    const scores = this.fresh.map((id) => {
      const c = this.nodes[id]
      // Worked out either way, so a student's grade can be shown beside the
      // model's - but it is only a policy call when the model's is used.
      const model = this.gradeByModel(id)
      const mine = yours?.get(id)
      if (mine === undefined) this.calls += 1
      const lm = mine ?? model.score
      if (mine !== undefined) c.yours = mine
      c.lm = round(lm)
      c.value = round(lam * lm + (1 - lam) * (c.sc ?? 0))
      return {
        id,
        label: c.label,
        prior: c.prior ?? 0,
        lm: c.lm,
        sc: c.sc ?? 0,
        value: c.value,
        reward: c.reward,
        why:
          mine !== undefined
            ? `Your grade. The stand-in model said ${model.score.toFixed(2)}: ${model.why}`
            : model.why,
        by: mine !== undefined ? 'you' : 'model',
      }
    })
    this.emit('evaluation', this.iteration, 'Evaluation',
      `V(s) = ${lam}·LM(s) + ${round(1 - lam, 2)}·SC(s), with LM(s) ${yours ? 'from your grades' : 'from the model'}, taken after the environment replied.`,
      { lam, timing: 'after', scores },
      this.fresh,
      this.pathTo(this.target ?? 0),
    )
    this.phase = 'simulate'
  }

  // -- 4. simulation ------------------------------------------------------

  simulate(): void {
    if (this.phase !== 'simulate' || this.target === null) throw new Error('not simulating')
    const path = this.pathTo(this.target)
    const solved = this.fresh.find((id) => (this.nodes[id].reward ?? 0) >= 1)
    if (solved !== undefined) {
      this.leaf = solved
      this.truncated = false
      this.emit('simulation', this.iteration, 'Simulation not needed',
        `Expansion already reached 24 at node ${solved}; there is nothing to roll out.`,
        { rollout: [solved], terminal: solved, truncated: false, skipped: true,
          observation: this.nodes[solved].observation ?? '' },
        [solved], [...path, solved])
      this.phase = 'backprop'
      return
    }

    // Greedy by value from the best fresh child; new states graded by the model.
    let node = this.best(this.fresh)
    const walked = [node.id]
    const graded: Record<string, unknown>[] = []
    while (!node.terminal && node.depth < MAX_DEPTH) {
      if (!node.children.length) {
        const kids = this.sampleChildren(node)
        if (!kids.length) break
        for (const id of kids) {
          const c = this.nodes[id]
          const g = this.gradeByModel(id)
          this.calls += 1
          c.lm = g.score
          c.value = round(this.cfg.lambda * g.score + (1 - this.cfg.lambda) * (c.sc ?? 0))
          graded.push({ id, label: c.label, prior: c.prior, lm: c.lm, sc: c.sc, value: c.value, reward: c.reward, why: g.why })
        }
      }
      node = this.best(node.children)
      walked.push(node.id)
    }
    const finished = node.depth === MAX_DEPTH && node.reward !== null
    this.leaf = node.id
    this.truncated = !finished
    this.emit('simulation', this.iteration, 'Simulation',
      `Greedy descent by value, in the real arithmetic, ${finished ? `reached a final value after ${walked.length - 1} more step${walked.length === 2 ? '' : 's'}.` : 'ran out of moves.'}`,
      { rollout: walked, terminal: node.id, truncated: this.truncated,
        observation: node.observation ?? '', graded },
      walked, [...path, ...walked.slice(1)])
    this.phase = 'backprop'
  }

  private best(ids: number[]): PNode {
    // Ties go to the earlier child, as Python's max() does.
    return ids
      .map((id) => this.nodes[id])
      .reduce((a, b) => ((b.value ?? 0) > (a.value ?? 0) ? b : a))
  }

  // -- 5. backpropagation -------------------------------------------------

  backpropagate(): void {
    if (this.phase !== 'backprop' || this.leaf === null) throw new Error('not backing up')
    const leaf = this.nodes[this.leaf]
    const reward = this.truncated || leaf.reward === null ? 0 : leaf.reward
    leaf.reward = reward
    const updates: { id: number; before: { visits: number; value: number }; after: { visits: number; value: number } }[] = []
    for (const id of [...this.pathTo(leaf.id)].reverse()) {
      const n = this.nodes[id]
      const before = { visits: n.visits, value: round(n.value ?? 0) }
      n.visits += 1
      n.value = ((n.value ?? 0) * (n.visits - 1) + reward) / n.visits
      updates.push({ id, before, after: { visits: n.visits, value: round(n.value) } })
    }
    this.emit('backpropagation', this.iteration, 'Backpropagation',
      `Reward ${reward} flows from node ${leaf.id} up to the root, revising ${updates.length} values.`,
      { reward, leaf: leaf.id, updates },
      updates.map((u) => u.id),
      [...updates].reverse().map((u) => u.id),
    )
    if (reward >= 1) {
      this.finish('a trajectory reached 24')
    } else {
      this.phase = 'reflect'
    }
  }

  // -- 6. reflection ------------------------------------------------------

  /**
   * The note the task's template would write, and the one move it rules out.
   *
   * The same note `Game24Task.reflection_for` writes: the whole failed trail
   * in words, and the opening as the thing to avoid. Only the opening is ruled
   * out - the trail is there to be read, and ruling out every step of it would
   * forbid `9 + 15 = 24`, the last step of every solution to the hard puzzle.
   */
  suggestion(): { text: string; rulesOut: string[] } {
    const leaf = this.leaf === null ? null : this.nodes[this.leaf]
    if (!leaf || !leaf.steps.length) {
      return { text: 'That branch produced no arithmetic at all.', rulesOut: [] }
    }
    const final = leaf.numbers.length === 1 ? String(leaf.numbers[0]) : leaf.numbers.join(', ')
    return {
      text: `${leaf.steps.join(' then ')} ends at ${final}, not ${TARGET}. Do not open with that step again.`,
      rulesOut: [leaf.steps[0]],
    }
  }

  /** The moves a note would rule out if it were written now. */
  wouldRuleOut(text: string): string[] {
    const s = this.suggestion()
    return text.trim() === s.text ? s.rulesOut : movesNamed(text)
  }

  /**
   * Write a note, or skip it with `null`. The suggestion, sent unedited, is the
   * template's note and rules out what the template would; anything else is
   * the student's, and rules out every move it names.
   */
  reflect(text: string | null): void {
    if (this.phase !== 'reflect' || this.leaf === null) throw new Error('not reflecting')
    const leaf = this.nodes[this.leaf]
    if (text !== null && text.trim()) {
      const note = text.trim()
      const by = note === this.suggestion().text ? 'template' : 'you'
      const named = this.wouldRuleOut(note)
      for (const m of named) this.blocked.add(m)
      this.notes.push(note)
      leaf.reflected = true
      this.emit('reflection', this.iteration, 'Reflection',
        by === 'you'
          ? 'You wrote down why the trajectory failed. The note joins the context of every later expansion.'
          : 'The trajectory failed, so the template writes down why. The note joins the context for every later expansion.',
        { node: leaf.id, text: note, trajectory: this.pathTo(leaf.id), reward: leaf.reward ?? 0,
          total_notes: this.notes.length, by, rules_out: named },
        [leaf.id], this.pathTo(leaf.id))
    }
    this.nextIteration()
  }

  private nextIteration() {
    this.target = null
    this.fresh = []
    this.leaf = null
    this.iteration += 1
    if (this.iteration > this.cfg.iterations) {
      this.iteration -= 1
      this.finish('the iteration budget ran out')
    } else if (this.uctTarget() === null) {
      this.iteration -= 1
      this.finish('the tree was exhausted')
    } else {
      this.phase = 'select'
    }
  }

  private finish(stop: string) {
    const scored = this.nodes.filter((n) => n.reward !== null)
    const winner = scored.reduce<PNode | null>((a, b) => (!a || (b.reward ?? 0) > (a.reward ?? 0) ? b : a), null)
    const solved = !!winner && (winner.reward ?? 0) >= 1
    const result = {
      solved,
      best_reward: winner ? winner.reward : null,
      best_node: winner ? winner.id : null,
      best_path: winner ? this.pathTo(winner.id) : [],
      nodes: this.nodes.length,
      iterations_run: this.iteration,
      stopped_because: stop,
      reflections: [...this.notes],
    }
    this.emit('result', this.iteration, 'Search over',
      solved ? `Solved: 24 at node ${winner!.id}.` : `Stopped without solving - ${stop}.`,
      result, result.best_path, result.best_path)
    this.phase = 'done'
  }

  /** Every grade the student gave, against whether 24 was still reachable. */
  calibration(): Calibration[] {
    return this.nodes
      .filter((n) => n.yours !== undefined)
      .map((n) => ({
        id: n.id,
        label: n.label,
        yours: n.yours!,
        model: modelGrade(n.steps, n.numbers, new Set()).score,
        reachable: solvable(n.numbers),
      }))
  }

  /** The moves the notes so far have ruled out. */
  ruledOut(): string[] {
    return [...this.blocked]
  }

  // -- recording ----------------------------------------------------------

  private add(init: Omit<PNode, 'id' | 'created_at' | 'visits' | 'reward' | 'lm' | 'sc' | 'prior' | 'reflected' | 'children'>): PNode {
    const node: PNode = {
      ...init,
      id: this.nodes.length,
      created_at: this.steps.length,
      visits: 1,
      reward: null,
      lm: null,
      sc: null,
      prior: null,
      reflected: false,
      children: [],
    }
    this.nodes.push(node)
    return node
  }

  private emit(
    op: Step['op'],
    iteration: number,
    title: string,
    summary: string,
    detail: Record<string, unknown>,
    focus: number[],
    path: number[],
  ) {
    const state: Record<string, NodeState> = {}
    for (const n of this.nodes) {
      state[String(n.id)] = {
        visits: n.visits,
        value: n.value === null ? null : round(n.value),
        reward: n.reward,
        lm: n.lm,
        sc: n.sc,
        prior: n.prior,
        terminal: n.terminal,
        reflected: n.reflected,
      }
    }
    this.steps.push({
      index: this.steps.length,
      op,
      iteration,
      title,
      summary,
      detail,
      focus,
      path,
      tokens: 0,
      state,
    })
  }

  private brief() {
    return {
      id: 'game-of-24-play',
      family: 'game',
      title: this.title,
      prompt: `Use ${this.cfg.numbers.join(', ')} exactly once each, with + - * / and any bracketing, to make ${TARGET}.`,
      reward: '1 if the final value is exactly 24 and every number was used once, else 0',
      context: { Numbers: this.cfg.numbers.map(String) },
    }
  }

  private configDict() {
    return {
      n: this.cfg.n,
      w: this.cfg.w,
      lambda: this.cfg.lambda,
      iterations: this.cfg.iterations,
      max_depth: MAX_DEPTH,
      simulate: true,
      reflect: true,
      reflect_threshold: 1,
      solved_at: 1,
      seed: this.cfg.seed,
      evaluation: 'after' as const,
    }
  }

  /** The session so far, as a trace the viewer can render, save or compare. */
  toTrace(): Trace {
    const last = this.steps[this.steps.length - 1]
    const result =
      last.op === 'result'
        ? (last.detail as unknown as Trace['result'])
        : {
            solved: false,
            best_reward: null,
            best_node: null,
            best_path: [],
            nodes: this.nodes.length,
            iterations_run: this.iteration,
            stopped_because: 'the search is still running',
            reflections: [...this.notes],
          }
    return {
      schema: 'lats-trace/1',
      name: `you_game-of-24_${this.cfg.numbers.join('-')}`,
      method: 'lats',
      generated_by: { package: 'lats-visualizer play mode', version: '1.1.0' },
      task: this.brief(),
      config: this.configDict(),
      policy: {
        kind: 'mock',
        name: 'You, with the offline policy proposing',
        model: null,
        seed: this.cfg.seed,
        calls: this.calls,
        tokens: 0,
        tokens_are_estimated: true,
        sampling: 'independent',
      },
      result,
      nodes: this.nodes.map((n) => ({
        id: n.id,
        parent: n.parent,
        depth: n.depth,
        label: n.label,
        action: n.action,
        observation: n.observation,
        detail: n.detail,
        terminal: n.terminal,
        created_at: n.created_at,
      })),
      // Copied, so a React render of one session state cannot see a later one.
      steps: [...this.steps],
    }
  }
}
