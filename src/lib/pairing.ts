/** Choosing what to open beside a trace in the comparison view. */

import type { Source } from '../components/TracePicker'

/**
 * What to put beside a trace when the comparison opens: on the same task, a
 * baseline beside LATS - Tree of Thoughts first, the other search method - and
 * LATS beside anything else, preferring the same policy.
 */
export function counterpart(sources: Source[], key: string | null): string | null {
  const me = sources.find((s) => s.key === key)
  if (!me) return sources.find((s) => s.key !== key)?.key ?? null
  const mine = me.method ?? 'lats'
  const mock = (s: Source) => s.label.startsWith('mock_')
  const same = sources.filter((s) => s.key !== me.key && s.task && s.task === me.task)
  const wanted = mine === 'lats' ? ['tot', 'react', 'reflexion'] : ['lats']
  for (const method of wanted) {
    const hit =
      same.find((s) => (s.method ?? 'lats') === method && mock(s) === mock(me)) ??
      same.find((s) => (s.method ?? 'lats') === method)
    if (hit) return hit.key
  }
  return same[0]?.key ?? sources.find((s) => s.key !== me.key)?.key ?? null
}
