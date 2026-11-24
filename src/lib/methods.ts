/**
 * What each search method is, which of LATS's six operations it has, and what
 * the viewer should say about a trace before anyone reads a step of it.
 *
 * The baselines are written by `scripts/run_lats/baselines.py` in the same
 * trace format as LATS, with LATS's names for the operations they share. So a
 * ReAct trace is a LATS trace in which only `expansion` ever happens, with one
 * sample - which is exactly the comparison the paper's Table 1 draws, and the
 * reason the operation strip greys out what a method does not have rather than
 * hiding it.
 */

import type { Method, Op, Step, Trace } from '../types'

export interface MethodInfo {
  id: Method
  /** As it is written in its paper. */
  name: string
  /** For a chip. */
  short: string
  cite: string
  /** One sentence: what it does differently. */
  blurb: string
  /** The LATS operations it performs. */
  ops: ReadonlySet<Op>
  /** Whether nodes carry a value at all. */
  valued: boolean
}

const ALL: Op[] = [
  'selection',
  'expansion',
  'evaluation',
  'simulation',
  'backpropagation',
  'reflection',
]

export const METHODS: Record<Method, MethodInfo> = {
  lats: {
    id: 'lats',
    name: 'LATS',
    short: 'LATS',
    cite: 'Zhou et al., ICML 2024',
    blurb:
      'Monte Carlo tree search over actions: UCT selection, n samples per expansion, a value taken after the environment replies, greedy rollouts, backpropagation and reflection.',
    ops: new Set(ALL),
    valued: true,
  },
  react: {
    id: 'react',
    name: 'ReAct',
    short: 'ReAct',
    cite: 'Yao et al., ICLR 2023',
    blurb:
      'One trajectory: sample a single action, execute it, read the observation, repeat. Nothing is compared, graded or undone.',
    ops: new Set<Op>(['expansion']),
    valued: false,
  },
  reflexion: {
    id: 'reflexion',
    name: 'Reflexion',
    short: 'Reflexion',
    cite: 'Shinn et al., NeurIPS 2023',
    blurb:
      'ReAct trials from the root, one after another, with a written note after each failure carried into every later trial. No tree, and no value.',
    ops: new Set<Op>(['expansion', 'reflection']),
    valued: false,
  },
  tot: {
    id: 'tot',
    name: 'Tree of Thoughts',
    short: 'ToT',
    cite: 'Yao et al., NeurIPS 2023',
    blurb:
      'Breadth-first search that keeps the best b states at each depth, valued by the policy’s own grade from before each step ran. No exploration bonus, no backpropagation, no reflection.',
    ops: new Set<Op>(['selection', 'expansion', 'evaluation']),
    valued: true,
  },
}

/** Rank for sorting: LATS first, then the baselines in the order above. */
export const METHOD_ORDER: Method[] = ['lats', 'react', 'reflexion', 'tot']

export function methodOf(trace: Pick<Trace, 'method'>): MethodInfo {
  return METHODS[trace.method ?? 'lats'] ?? METHODS.lats
}

/**
 * Written by version 1.0 of the generator. Those traces took LM(s) from the
 * policy's grade of each action before it was executed, and a real model was
 * asked for n distinct actions in one reply.
 */
export function isLegacy(trace: Pick<Trace, 'config'>): boolean {
  return trace.config.evaluation === undefined
}

/** When LM(s) was taken, with version 1.0 read as what it actually did. */
export function evaluationOf(trace: Pick<Trace, 'config'>): 'after' | 'before' | 'none' {
  return trace.config.evaluation ?? 'before'
}

export interface Setting {
  label: string
  value: string
  /** What the knob means, for a tooltip. */
  title: string
}

/** The settings this trace's method actually has, in reading order. */
export function settingsOf(trace: Trace): Setting[] {
  const c = trace.config
  const m = methodOf(trace).id
  const out: Setting[] = []
  if (m === 'lats' || m === 'tot') {
    out.push({ label: 'n', value: String(c.n), title: 'independent samples per expansion' })
  }
  if (m === 'lats') {
    out.push({ label: 'w', value: String(c.w), title: 'exploration weight in UCT' })
    out.push({
      label: 'λ',
      value: String(c.lambda),
      title: 'weight on LM(s), the value function’s grade, against self-consistency SC(s)',
    })
  }
  if (m === 'tot') {
    out.push({ label: 'b', value: String(c.beam ?? '?'), title: 'states the beam keeps at each depth' })
  }
  if (m === 'reflexion') {
    out.push({ label: 'trials', value: `≤ ${c.iterations}`, title: 'ReAct trials from the root, at most' })
  }
  if (m === 'lats') {
    out.push({ label: 'iterations', value: `≤ ${c.iterations}`, title: 'search iterations, at most' })
  }
  out.push({ label: 'depth', value: `≤ ${c.max_depth}`, title: 'hard depth limit' })
  if (m === 'lats') {
    const ev = evaluationOf(trace)
    out.push({
      label: 'LM(s)',
      value: ev === 'after' ? 'after reply' : 'before reply',
      title:
        ev === 'after'
          ? 'graded by a separate call once the environment had replied, from the state with the reply in it'
          : isLegacy(trace)
            ? 'version 1.0: the policy’s grade of each action from before it ran'
            : 'ablation: the policy’s grade of each action from before it ran, as in Tree of Thoughts',
    })
    if (!c.simulate) {
      out.push({ label: 'simulation', value: 'off', title: 'the paper skips it in the programming setting' })
    }
  }
  if ((m === 'lats' || m === 'reflexion') && !c.reflect) {
    out.push({ label: 'reflection', value: 'off', title: 'ablation' })
  }
  return out
}

/**
 * A step's summary, as it should be read.
 *
 * Version 1.0 wrote "scored after the environment replied" into every
 * evaluation step while taking LM(s) from the grade given before. The file is a
 * record and is left as it is; the viewer corrects the claim where it shows it.
 */
export function summaryOf(trace: Trace, step: Step): string {
  if (step.op !== 'evaluation' || !isLegacy(trace)) return step.summary
  return step.summary.replace(
    /,?\s*scored after the environment replied\.?$/,
    '. In this version-1.0 trace LM(s) is the policy’s grade from before each move ran.',
  )
}

/**
 * What a reader should know before trusting the arithmetic in this trace.
 * Empty for a trace that matches the paper's ordering.
 */
export function caveats(trace: Trace): string[] {
  if (!isLegacy(trace)) return []
  const out = [
    'Recorded by version 1.0 of the trace generator, which took LM(s) from the policy’s grade of each action before it ran — the Tree-of-Thoughts ordering, not the one LATS describes. Read this run as a simplified variant of LATS.',
  ]
  if (trace.policy.kind !== 'mock') {
    out.push(
      'Its real-model prompt also asked for n distinct actions in one reply, so samples could not agree and SC(s) is 1/n for almost every child: the same amount added to every sibling, which says nothing about which is better.',
    )
  }
  return out
}
