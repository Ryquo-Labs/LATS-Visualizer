/**
 * Game of 24, in the browser.
 *
 * A port of `scripts/run_lats/tasks/game24.py` - the legal moves, the tidiness
 * heuristic the offline policy samples with, and its grader - so that the play
 * mode can run a real search with no Python behind it. Arithmetic is exact:
 * 6 / 9 is two thirds, not 0.666..., because a solution that runs through a
 * fraction is the whole point of the hard puzzle.
 *
 * The sampler is seeded, but it is not Python's Mersenne Twister, so a session
 * here does not reproduce the trace `run_lats.py` writes for the same seed. It
 * draws from the same distribution.
 */

export const TARGET = 24

/** An exact rational, always in lowest terms with a positive denominator. */
export class Frac {
  readonly n: number
  readonly d: number

  constructor(n: number, d = 1) {
    if (d === 0) throw new RangeError('zero denominator')
    const g = gcd(Math.abs(n), Math.abs(d)) || 1
    const sign = d < 0 ? -1 : 1
    this.n = (sign * n) / g
    this.d = (sign * d) / g
  }

  add(o: Frac) {
    return new Frac(this.n * o.d + o.n * this.d, this.d * o.d)
  }
  sub(o: Frac) {
    return new Frac(this.n * o.d - o.n * this.d, this.d * o.d)
  }
  mul(o: Frac) {
    return new Frac(this.n * o.n, this.d * o.d)
  }
  div(o: Frac): Frac | null {
    return o.n === 0 ? null : new Frac(this.n * o.d, this.d * o.n)
  }
  equals(o: Frac) {
    return this.n === o.n && this.d === o.d
  }
  get value() {
    return this.n / this.d
  }
  toString() {
    return this.d === 1 ? String(this.n) : `${this.n}/${this.d}`
  }

  static parse(text: string): Frac | null {
    const m = /^\s*(-?\d+)(?:\s*\/\s*(\d+))?\s*$/.exec(text)
    if (!m) return null
    const d = m[2] ? Number(m[2]) : 1
    return d === 0 ? null : new Frac(Number(m[1]), d)
  }
}

function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b]
  return a
}

export interface Move {
  /** `8 / 2 = 4`: the step, and its result. */
  expr: string
  /** `8 / 2`: what the tree card prints. */
  label: string
  result: Frac
  remaining: Frac[]
}

type Op = '+' | '-' | '*' | '/'
const OPS: [Op, (x: Frac, y: Frac) => Frac | null][] = [
  ['+', (x, y) => x.add(y)],
  ['-', (x, y) => x.sub(y)],
  ['*', (x, y) => x.mul(y)],
  ['/', (x, y) => x.div(y)],
]

/** Python's sort key for the numbers left: denominator first, then value. */
function byDenominator(a: Frac, b: Frac) {
  return a.d - b.d || a.value - b.value
}

/**
 * Every distinct (pair, operator) move from `numbers`, in the order the
 * Python environment lists them. `+` and `*` are tried one way round only;
 * moves that leave the same numbers with the same operator are one move.
 */
export function legalMoves(numbers: Frac[]): Move[] {
  const seen = new Set<string>()
  const moves: Move[] = []
  for (let i = 0; i < numbers.length; i++) {
    for (let j = i + 1; j < numbers.length; j++) {
      const a = numbers[i]
      const b = numbers[j]
      const rest = numbers.filter((_, k) => k !== i && k !== j)
      for (const [symbol, fn] of OPS) {
        for (const [x, y] of [
          [a, b],
          [b, a],
        ] as const) {
          if ((symbol === '+' || symbol === '*') && x !== a) continue
          const result = fn(x, y)
          if (result === null) continue
          const remaining = [...rest, result].sort(byDenominator)
          const key = `${symbol}|${remaining.join(',')}`
          if (seen.has(key)) continue
          seen.add(key)
          moves.push({
            expr: `${x} ${symbol} ${y} = ${result}`,
            label: `${x} ${symbol} ${y}`,
            result,
            remaining,
          })
        }
      }
    }
  }
  return moves
}

/** Can these numbers still make 24? The ground truth a student's grade is checked against. */
export function solvable(numbers: Frac[], memo = new Map<string, boolean>()): boolean {
  if (numbers.length === 1) return numbers[0].equals(new Frac(TARGET))
  const key = [...numbers].sort(byDenominator).join(',')
  const hit = memo.get(key)
  if (hit !== undefined) return hit
  const ok = legalMoves(numbers).some((m) => solvable(m.remaining, memo))
  memo.set(key, ok)
  return ok
}

/** How promising a move looks to the offline policy. A port of `tidiness`. */
export function tidiness(result: Frac, remaining: Frac[]): number {
  let score = 0.45
  if (result.d !== 1) {
    score -= 0.3
  } else {
    score += 0.15
    const v = result.n
    if (v > 0 && (TARGET % v === 0 || v % TARGET === 0)) score += 0.2
    if (v > 0 && v <= TARGET) score += 0.1
  }
  if (result.value < 0) score -= 0.2
  if (Math.abs(result.value) > 100) score -= 0.15
  if (remaining.length === 1 && remaining[0].equals(new Frac(TARGET))) score = 0.98
  return Math.min(Math.max(score, 0.05), 0.98)
}

/** A small seeded generator (mulberry32). Same seed, same session. */
export function seeded(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export interface Sample {
  move: Move
  /** The policy's grade of its own move, before it runs. */
  prior: number
}

/** Softmax temperature of the offline policy, as in the Python task. */
const TEMPERATURE = 0.3

/**
 * `n` independent samples from the offline policy, with replacement - so two
 * of them agreeing is a measurement, which is what SC(s) counts. A move a note
 * has ruled out is weighted down, as in `mock_propose`.
 */
export function propose(
  numbers: Frac[],
  rng: () => number,
  n: number,
  blocked: ReadonlySet<string>,
): Sample[] {
  const moves = legalMoves(numbers)
  if (!moves.length) return []
  const scores = moves.map(
    (m) => tidiness(m.result, m.remaining) * (blocked.has(m.expr) ? 0.25 : 1),
  )
  const top = Math.max(...scores)
  const weights = scores.map((s) => Math.exp((s - top) / TEMPERATURE))
  const total = weights.reduce((a, b) => a + b, 0)
  const out: Sample[] = []
  for (let k = 0; k < n; k++) {
    let r = rng() * total
    let i = 0
    while (i < weights.length - 1 && r >= weights[i]) {
      r -= weights[i]
      i += 1
    }
    out.push({ move: moves[i], prior: round(scores[i], 3) })
  }
  return out
}

/**
 * The offline value function, graded after the step: the environment's
 * verdict at the end, otherwise the tidiness of the numbers left. A port of
 * `Game24Task.mock_evaluate`.
 */
export function modelGrade(
  steps: string[],
  left: Frac[],
  blocked: ReadonlySet<string>,
): { score: number; why: string } {
  if (left.length === 1) {
    return left[0].equals(new Frac(TARGET))
      ? { score: 0.98, why: `The environment confirms ${left[0]} is ${TARGET}.` }
      : { score: 0.02, why: `The environment says the final value ${left[0]} is not ${TARGET}.` }
  }
  const step = steps[steps.length - 1]
  const result = Frac.parse(step.split('=').pop() ?? '') ?? new Frac(0)
  let score = tidiness(result, left)
  let why = `Left with ${left.join(', ')}`
  if (blocked.has(step)) {
    score *= 0.25
    why += '; a note in context says not to open with this step'
  }
  return { score: round(score, 3), why: `${why}.` }
}

/**
 * The moves a note names, as the exact `expr` strings the sampler compares
 * against - so "don't open with 8/2" rules out `8 / 2 = 4` wherever it can be
 * played. Addition and multiplication are matched either way round.
 */
export function movesNamed(note: string): string[] {
  const found = new Set<string>()
  // A fraction operand is written tight - `1/4` - so that `-6 / 1/4` reads as
  // -6 divided by a quarter, not as -6/1 divided by 4.
  const pattern = /(-?\d+(?:\/\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\/\d+)?)/g
  for (const m of note.matchAll(pattern)) {
    const x = Frac.parse(m[1])
    const y = Frac.parse(m[3])
    const symbol = ({ x: '*', '×': '*', '÷': '/' } as Record<string, Op>)[m[2]] ?? (m[2] as Op)
    if (!x || !y) continue
    const fn = OPS.find(([s]) => s === symbol)?.[1]
    const result = fn?.(x, y)
    if (!result) continue
    found.add(`${x} ${symbol} ${y} = ${result}`)
    if (symbol === '+' || symbol === '*') found.add(`${y} ${symbol} ${x} = ${result}`)
  }
  return [...found]
}

export function round(v: number, digits = 4) {
  const k = 10 ** digits
  return Math.round(v * k) / k
}
