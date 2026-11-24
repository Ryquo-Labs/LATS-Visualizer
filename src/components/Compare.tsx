/**
 * Two searches, side by side.
 *
 * The replay view explains one search in depth; this puts two next to each
 * other so the difference between them is the thing on screen - LATS against
 * ReAct, Reflexion or Tree of Thoughts on the same environment, an ablation
 * against the run it ablates, a played session against the search LATS ran on
 * its own. Each pane has its own picker, tree and transport, and the arrow keys
 * step both at once, so the two trees grow together: one stays a single line
 * while the other branches, one prunes what the other comes back to.
 *
 * Below each tree is the step's one-line summary rather than its full panel -
 * two panels would not fit beside two trees, and the summary is the part a
 * comparison needs. Every pane also counts the actions it has run so far, which
 * is the honest axis to compare cost on: a search that solves the task after
 * forty actions has not beaten one that failed after four for free.
 */

import { useEffect, useMemo, useState } from 'react'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import CircularProgress from '@mui/material/CircularProgress'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'

import Timeline from './Timeline'
import TracePicker, { type Source } from './TracePicker'
import TreeView from './TreeView'
import { LegacyChip, MethodChip } from './TraceHeader'
import { num } from '../lib/format'
import { viewAt } from '../lib/layout'
import { fetchBundled } from '../lib/load'
import { summaryOf } from '../lib/methods'
import type { Trace } from '../types'
import { ACCENT, GOOD, INK, INK_DIM, INK_FAINT, MONO, OP_COLOR, alpha } from '../theme'

const BASE_TICK = 1100

type Pair<T> = [T, T]

interface Props {
  sources: Source[]
  uploaded: Record<string, Trace>
  /** The two traces to open with. */
  initial: Pair<string | null>
  compact: boolean
  /** One pane above the other, for a screen too narrow for two trees. */
  stacked: boolean
}

export default function Compare({ sources, uploaded, initial, compact, stacked }: Props) {
  const [keys, setKeys] = useState<Pair<string | null>>(initial)
  const [traces, setTraces] = useState<Pair<Trace | null>>([null, null])
  const [problems, setProblems] = useState<Pair<string | null>>([null, null])
  const [indices, setIndices] = useState<Pair<number>>([0, 0])
  const [selected, setSelected] = useState<Pair<number | null>>([null, null])
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)

  // Load each side whenever its choice changes.
  useEffect(() => {
    let cancelled = false
    keys.forEach((key, side) => {
      const source = sources.find((s) => s.key === key)
      const set = (t: Trace | null, problem: string | null) => {
        if (cancelled) return
        setTraces((prev) => (side === 0 ? [t, prev[1]] : [prev[0], t]))
        setProblems((prev) => (side === 0 ? [problem, prev[1]] : [prev[0], problem]))
      }
      if (!source) return set(null, key ? 'That trace is no longer available.' : null)
      if (source.kind === 'uploaded') return set(uploaded[source.key] ?? null, null)
      if (!source.file) return set(null, 'That trace has no file.')
      void fetchBundled(source.file).then((r) =>
        r.ok ? set(r.trace, null) : set(null, r.errors.join(' ')),
      )
    })
    return () => {
      cancelled = true
    }
  }, [keys, sources, uploaded])

  const lengths = useMemo<Pair<number>>(
    () => [
      traces[0] ? traces[0].steps.length - 1 : 0,
      traces[1] ? traces[1].steps.length - 1 : 0,
    ],
    [traces],
  )

  // One clock for both: each pane advances until its own trace runs out, and
  // the clock stops itself on the tick that finds both at the end.
  useEffect(() => {
    if (!playing) return
    const id = window.setTimeout(() => {
      if (indices[0] >= lengths[0] && indices[1] >= lengths[1]) setPlaying(false)
      else setIndices(([a, b]) => [Math.min(a + 1, lengths[0]), Math.min(b + 1, lengths[1])])
    }, BASE_TICK / speed)
    return () => window.clearTimeout(id)
  }, [playing, indices, lengths, speed])

  // The arrow keys step both panes, so the two trees grow in step.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && /INPUT|TEXTAREA|SELECT/.test(target.tagName)) return
      const by = (d: number) =>
        setIndices(([a, b]) => [
          Math.max(0, Math.min(a + d, lengths[0])),
          Math.max(0, Math.min(b + d, lengths[1])),
        ])
      if (e.key === 'ArrowRight') {
        setPlaying(false)
        by(1)
      } else if (e.key === 'ArrowLeft') {
        setPlaying(false)
        by(-1)
      } else if (e.key === ' ') {
        e.preventDefault()
        setPlaying((p) => !p)
      } else if (e.key === 'Home') {
        setIndices([0, 0])
      } else if (e.key === 'End') {
        setIndices([lengths[0], lengths[1]])
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [lengths])

  const pane = (side: 0 | 1) => (
    <Pane
      key={side}
      sources={sources}
      current={keys[side]}
      onPick={(key) => {
        setKeys((prev) => (side === 0 ? [key, prev[1]] : [prev[0], key]))
        setIndices((prev) => (side === 0 ? [0, prev[1]] : [prev[0], 0]))
        setSelected((prev) => (side === 0 ? [null, prev[1]] : [prev[0], null]))
      }}
      trace={traces[side]}
      problem={problems[side]}
      index={Math.min(indices[side], lengths[side])}
      onIndex={(i) => setIndices((prev) => (side === 0 ? [i, prev[1]] : [prev[0], i]))}
      selected={selected[side]}
      onSelect={(id) => setSelected((prev) => (side === 0 ? [id, prev[1]] : [prev[0], id]))}
      playing={playing}
      onPlaying={setPlaying}
      speed={speed}
      onSpeed={setSpeed}
      compact={compact}
      stacked={stacked}
    />
  )

  const gap = compact ? 1 : 1.5
  return (
    <Stack
      direction={stacked ? 'column' : 'row'}
      spacing={gap}
      sx={{ height: stacked ? 'auto' : '100%', minHeight: 0 }}
    >
      {pane(0)}
      {pane(1)}
    </Stack>
  )
}

interface PaneProps {
  sources: Source[]
  current: string | null
  onPick: (key: string) => void
  trace: Trace | null
  problem: string | null
  index: number
  onIndex: (i: number) => void
  selected: number | null
  onSelect: (id: number | null) => void
  playing: boolean
  onPlaying: (p: boolean) => void
  speed: number
  onSpeed: (s: number) => void
  compact: boolean
  stacked: boolean
}

function Pane(props: PaneProps) {
  const { trace, index, compact, stacked } = props
  const view = useMemo(() => (trace ? viewAt(trace, index) : null), [trace, index])
  const gap = compact ? 1 : 1.5

  const picker = (
    <TracePicker
      sources={props.sources}
      current={props.current}
      onSelect={props.onPick}
      onUpload={() => {}}
      compact
      noUpload
    />
  )

  if (!trace || !view) {
    return (
      <Stack spacing={gap} sx={{ flex: 1, minWidth: 0 }}>
        <Paper elevation={0} sx={{ p: 1 }}>
          {picker}
        </Paper>
        <Paper elevation={0} sx={{ flex: 1, display: 'grid', placeItems: 'center', minHeight: 200 }}>
          {props.problem ? (
            <Typography variant="body2" sx={{ color: INK_DIM, p: 2 }}>
              {props.problem}
            </Typography>
          ) : (
            <CircularProgress size={22} />
          )}
        </Paper>
      </Stack>
    )
  }

  const step = view.step
  const actions = trace.nodes.length - 1
  const actionsNow = view.visible.length - 1
  const node = trace.nodes.find((n) => n.id === props.selected) ?? null
  const state = node ? view.state(node.id) : undefined

  return (
    <Stack spacing={gap} sx={{ flex: 1, minWidth: 0, minHeight: stacked ? 'auto' : 0 }}>
      <Paper elevation={0} sx={{ px: 1, py: 0.9 }}>
        {picker}
        <Stack
          direction="row"
          spacing={0.75}
          sx={{ mt: 0.9, alignItems: 'center', flexWrap: 'wrap', rowGap: 0.5 }}
        >
          <MethodChip trace={trace} />
          <Chip
            size="small"
            label={
              trace.result.solved
                ? 'solved'
                : `not solved · reward ${num(trace.result.best_reward)}`
            }
            sx={{
              bgcolor: alpha(trace.result.solved ? GOOD : ACCENT, 0.16),
              color: trace.result.solved ? GOOD : ACCENT,
              fontWeight: 700,
            }}
          />
          <LegacyChip trace={trace} />
          <Box sx={{ flex: 1 }} />
          <Typography sx={{ fontFamily: MONO, fontSize: '0.72rem', color: INK_DIM }}>
            {actionsNow}/{actions} actions run · {trace.policy.calls} calls
          </Typography>
        </Stack>
      </Paper>

      <Box sx={{ display: 'flex', flex: stacked ? 'none' : 1, height: stacked ? 300 : 'auto', minHeight: 0 }}>
        <TreeView
          key={trace.name}
          trace={trace}
          view={view}
          selected={props.selected}
          onSelect={props.onSelect}
          minHeight={stacked ? 300 : 200}
        />
      </Box>

      <Paper elevation={0} sx={{ px: 1.25, py: 0.9, borderLeft: `3px solid ${OP_COLOR[step.op]}` }}>
        <Stack direction="row" spacing={1} sx={{ alignItems: 'baseline' }}>
          <Typography variant="subtitle2" sx={{ color: OP_COLOR[step.op], textTransform: 'uppercase' }}>
            {step.op}
          </Typography>
          <Typography variant="caption" sx={{ color: INK_FAINT, fontFamily: MONO }}>
            {step.iteration ? `iteration ${step.iteration}` : ''}
          </Typography>
        </Stack>
        <Typography
          variant="body2"
          sx={{
            color: INK,
            display: '-webkit-box',
            WebkitLineClamp: 3,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
            minHeight: '3.2em',
          }}
        >
          {summaryOf(trace, step)}
        </Typography>
        {node && (
          <Typography variant="caption" sx={{ display: 'block', color: INK_DIM, mt: 0.5 }}>
            <Box component="span" sx={{ fontFamily: MONO, color: INK }}>
              #{node.id} {node.label}
            </Box>
            {state && state.value !== null && ` · V ${num(state.value)}`}
            {state?.reward !== null && state?.reward !== undefined && ` · reward ${num(state.reward)}`}
            {node.observation && ` — ${node.observation}`}
          </Typography>
        )}
      </Paper>

      <Timeline
        trace={trace}
        index={index}
        playing={props.playing}
        speed={props.speed}
        onIndex={props.onIndex}
        onPlaying={props.onPlaying}
        onSpeed={props.onSpeed}
        compact
      />
    </Stack>
  )
}
