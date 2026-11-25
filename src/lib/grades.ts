/** Finding the grade a node was given, wherever in the trace it was given. */

import type { EvaluationDetail, ScoreEntry, SimulationDetail, Trace } from '../types'

/**
 * The value function's grade of node `id`: in the evaluation step after its
 * expansion, or along a rollout that created it. Grades are fixed once given,
 * so they live in the step that gave them rather than in every snapshot.
 */
export function gradeOf(trace: Trace, id: number): ScoreEntry | undefined {
  for (const step of trace.steps) {
    if (step.op === 'evaluation') {
      const hit = (step.detail as unknown as EvaluationDetail).scores?.find((s) => s.id === id)
      if (hit) return hit
    } else if (step.op === 'simulation') {
      const hit = (step.detail as unknown as SimulationDetail).graded?.find((s) => s.id === id)
      if (hit) return hit
    }
  }
  return undefined
}
