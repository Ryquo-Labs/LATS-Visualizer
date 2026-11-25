/**
 * Fetching the bundled traces, once each.
 *
 * The replay view and both panes of the comparison can ask for the same file;
 * the first request is shared and the parsed result is kept. A failed load is
 * forgotten, so the next attempt tries the network again rather than repeating
 * a transient error for the rest of the session.
 */

import { type ParseResult, validateTrace } from './validate'

/** Where the bundled traces are served from, relative to the app's base. */
export const TRACES = 'traces/'
/** Their index. Beside the folder rather than in it: it is not a trace. */
export const MANIFEST = 'traces-manifest.json'

export const url = (file: string) => new URL(file, document.baseURI).href

const cache = new Map<string, Promise<ParseResult>>()

export function fetchBundled(file: string): Promise<ParseResult> {
  const hit = cache.get(file)
  if (hit) return hit
  const pending = (async (): Promise<ParseResult> => {
    try {
      const res = await fetch(url(`${TRACES}${file}`))
      if (!res.ok) return { ok: false, errors: [`traces/${file} returned ${res.status}`] }
      const result = validateTrace(await res.json())
      return result.ok
        ? result
        : { ok: false, errors: [`traces/${file} did not validate:`, ...result.errors] }
    } catch (err) {
      return { ok: false, errors: [`Could not load traces/${file}: ${String(err)}`] }
    }
  })()
  cache.set(file, pending)
  pending.then((r) => {
    if (!r.ok) cache.delete(file)
  })
  return pending
}
