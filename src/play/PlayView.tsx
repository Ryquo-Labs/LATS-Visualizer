/**
 * Play mode: you are the parts of LATS that judge.
 *
 * Replaying a trace shows what the algorithm decided. This asks the student to
 * decide instead - which leaf to expand, how good each new state is, what to
 * write down after a failure - and runs the rest of the loop around them: the
 * offline policy proposes, the arithmetic executes, the rollout and the backup
 * are computed. Each decision is checked against what the algorithm would have
 * done, and the session can be finished by the algorithm at any point.
 *
 * It runs Game of 24 entirely in the browser (`play/game24.ts`); the other
 * environments need Python, or a policy that could read a free-text note.
 * Because a session is recorded as an ordinary trace, everything on the right
 * below the card is the replay viewer's own panel for the step just taken, and
 * the finished search can be downloaded or put side by side with the one LATS
 * ran alone.
 */

import { useEffect, useMemo, useState } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Divider from '@mui/material/Divider'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Select from '@mui/material/Select'
import Slider from '@mui/material/Slider'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'

import NodeDetail from '../components/NodeDetail'
import OperationPanel from '../components/OperationPanels'
import Timeline from '../components/Timeline'
import TreeView from '../components/TreeView'
import { num } from '../lib/format'
import { gradeOf } from '../lib/grades'
import { viewAt } from '../lib/layout'
import type { Trace } from '../types'
import {
  ACCENT,
  BAD,
  GOOD,
  INK,
  INK_DIM,
  INK_FAINT,
  MONO,
  OP_COLOR,
  PRIMARY,
  STROKE,
  SURFACE_2,
  VIOLET,
  alpha,
} from '../theme'
import { PUZZLES, type PlayConfig, Session } from './session'

const BASE_TICK = 1100

interface Props {
  compact: boolean
  /** One column instead of two. */
  stacked: boolean
  /** Put the finished session beside the bundled LATS trace for the same puzzle. */
  onCompare: (trace: Trace, bundled: string | null) => void
}

const DEFAULTS: PlayConfig = { numbers: [2, 5, 8, 11], n: 5, w: 1, lambda: 0.5, iterations: 12, seed: 7 }

export default function PlayView({ compact, stacked, onCompare }: Props) {
  const [cfg, setCfg] = useState<PlayConfig>(DEFAULTS)
  const [custom, setCustom] = useState('')
  const [session, setSession] = useState(() => new Session(DEFAULTS))
  /** Bumped after every mutation of the session, which is not React state. */
  const [version, setVersion] = useState(0)
  /** `null` follows the newest step; a number is the student scrubbing back. */
  const [index, setIndex] = useState<number | null>(null)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [selected, setSelected] = useState<number | null>(null)
  const [grades, setGrades] = useState<Record<number, number>>({})
  const [note, setNote] = useState('')
  const [problem, setProblem] = useState<string | null>(null)

  // `version` is the dependency that matters: the session mutates in place.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const trace = useMemo(() => session.toTrace(), [session, version])
  const last = trace.steps.length - 1
  const at = index ?? last
  const live = at === last
  const view = useMemo(() => viewAt(trace, at), [trace, at])
  const step = view.step

  useEffect(() => {
    if (!playing) return
    if (at >= last) {
      setPlaying(false)
      setIndex(null)
      return
    }
    const id = window.setTimeout(() => setIndex(at + 1), BASE_TICK / speed)
    return () => window.clearTimeout(id)
  }, [playing, at, last, speed])

  /** Run one session action, then return the view to the present. */
  const act = (fn: () => void) => {
    try {
      fn()
      setProblem(null)
    } catch (err) {
      setProblem(err instanceof Error ? err.message : String(err))
    }
    setVersion((v) => v + 1)
    setIndex(null)
    setGrades({})
  }

  const restart = (next: PlayConfig) => {
    setCfg(next)
    setSession(new Session(next))
    setVersion((v) => v + 1)
    setIndex(null)
    setSelected(null)
    setGrades({})
    setNote('')
    setProblem(null)
  }

  /** The algorithm's choice at every remaining step. */
  const finish = () =>
    act(() => {
      for (let guard = 0; session.phase !== 'done' && guard < 400; guard++) {
        if (session.phase === 'select') session.select(session.uctTarget()!, 'uct')
        else if (session.phase === 'expand') session.expand()
        else if (session.phase === 'evaluate') session.evaluate(null)
        else if (session.phase === 'simulate') session.simulate()
        else if (session.phase === 'backprop') session.backpropagate()
        else if (session.phase === 'reflect') session.reflect(session.suggestion().text)
      }
    })

  const pick = (id: number | null) => {
    setSelected(id)
    if (id !== null && session.phase === 'select' && live) setProblem(session.whyNot(id))
  }

  const download = () => {
    const blob = new Blob([JSON.stringify(trace, null, 2) + '\n'], { type: 'application/json' })
    const href = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = href
    a.download = `${trace.name}.json`
    a.click()
    URL.revokeObjectURL(href)
  }

  const bundled =
    PUZZLES.find((p) => p.numbers.join() === cfg.numbers.join())?.bundled ?? null

  const setup = (
    <Paper elevation={0} sx={{ px: 1.5, py: 1 }}>
      <Stack direction="row" spacing={1} sx={{ alignItems: 'center', flexWrap: 'wrap', rowGap: 1 }}>
        <Typography variant="subtitle2" sx={{ color: INK_FAINT, textTransform: 'uppercase' }}>
          Puzzle
        </Typography>
        <Select
          size="small"
          value={PUZZLES.find((p) => p.numbers.join() === cfg.numbers.join())?.label ?? 'custom'}
          onChange={(e) => {
            const p = PUZZLES.find((x) => x.label === e.target.value)
            if (p) restart({ ...cfg, numbers: p.numbers })
            else setCustom(cfg.numbers.join(' '))
          }}
          sx={{ fontSize: '0.8rem', '& .MuiSelect-select': { py: 0.5 } }}
        >
          {PUZZLES.map((p) => (
            <MenuItem key={p.label} value={p.label} sx={{ fontSize: '0.8rem' }}>
              {p.label}
            </MenuItem>
          ))}
          <MenuItem value="custom" sx={{ fontSize: '0.8rem' }}>
            your own…
          </MenuItem>
        </Select>
        <TextField
          size="small"
          placeholder="4 numbers, e.g. 3 3 8 8"
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          onKeyDown={(e) => {
            if (e.key !== 'Enter') return
            const numbers = custom.trim().split(/[\s,]+/).map(Number)
            if (numbers.length === 4 && numbers.every((v) => Number.isInteger(v) && v >= 1 && v <= 99)) {
              restart({ ...cfg, numbers })
            } else {
              setProblem('A puzzle is four whole numbers between 1 and 99.')
            }
          }}
          sx={{ width: 170, '& input': { py: 0.6, fontSize: '0.8rem' } }}
        />
        <Labelled label="n">
          <Select
            size="small"
            value={cfg.n}
            onChange={(e) => restart({ ...cfg, n: Number(e.target.value) })}
            sx={{ fontSize: '0.8rem', '& .MuiSelect-select': { py: 0.5 } }}
          >
            {[3, 5, 8].map((v) => (
              <MenuItem key={v} value={v} sx={{ fontSize: '0.8rem' }}>
                {v}
              </MenuItem>
            ))}
          </Select>
        </Labelled>
        <Labelled label="λ">
          <Select
            size="small"
            value={cfg.lambda}
            onChange={(e) => restart({ ...cfg, lambda: Number(e.target.value) })}
            sx={{ fontSize: '0.8rem', '& .MuiSelect-select': { py: 0.5 } }}
          >
            {[0.5, 0.8, 1].map((v) => (
              <MenuItem key={v} value={v} sx={{ fontSize: '0.8rem' }}>
                {v}
              </MenuItem>
            ))}
          </Select>
        </Labelled>
        <Box sx={{ flex: 1 }} />
        <Button size="small" variant="outlined" onClick={() => restart(cfg)}>
          Start over
        </Button>
      </Stack>
    </Paper>
  )

  const card = (
    <TurnCard
      session={session}
      live={live}
      selected={selected}
      grades={grades}
      setGrades={setGrades}
      note={note}
      setNote={setNote}
      problem={problem}
      act={act}
      finish={finish}
      back={() => setIndex(null)}
      at={at}
      download={download}
      compare={() => onCompare(trace, bundled)}
      restart={() => restart(cfg)}
    />
  )

  const selectedNode = trace.nodes.find((n) => n.id === selected) ?? null

  const panel = (
    <Paper
      elevation={0}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        ...(stacked
          ? { width: '100%', flexShrink: 0 }
          : { width: compact ? 360 : 460, flexShrink: 0, minHeight: 0 }),
      }}
    >
      <Box sx={{ flex: 1, overflowY: 'auto', p: 1.5, minHeight: 0 }}>
        {card}
        <Divider sx={{ my: 1.5 }} />
        <Stack direction="row" spacing={1} sx={{ alignItems: 'baseline', mb: 1 }}>
          <Typography
            variant="subtitle2"
            sx={{ color: OP_COLOR[step.op], textTransform: 'uppercase' }}
          >
            {step.op}
          </Typography>
          <Typography variant="caption" sx={{ color: INK_FAINT, fontFamily: MONO }}>
            step {step.index + 1}
            {step.iteration ? ` · iteration ${step.iteration}` : ''}
          </Typography>
        </Stack>
        <Typography variant="body2" sx={{ color: INK, mb: 1 }}>
          {step.summary}
        </Typography>
        <OperationPanel trace={trace} step={step} onSelect={pick} />
        {selectedNode && (
          <>
            <Divider sx={{ my: 1.5 }} />
            <NodeDetail
              node={selectedNode}
              state={view.state(selectedNode.id)}
              grade={gradeOf(trace, selectedNode.id)}
              onClose={() => setSelected(null)}
            />
          </>
        )}
      </Box>
    </Paper>
  )

  const tree = (
    <TreeView
      key={`${cfg.numbers.join()}-${cfg.n}-${cfg.lambda}-${session.cfg.seed}`}
      trace={trace}
      view={view}
      selected={selected}
      onSelect={pick}
      minHeight={stacked ? 300 : 280}
    />
  )

  const timeline = (
    <Timeline
      trace={trace}
      index={at}
      playing={playing}
      speed={speed}
      onIndex={(i) => setIndex(i >= last ? null : i)}
      onPlaying={setPlaying}
      onSpeed={setSpeed}
      compact={compact}
    />
  )

  const gap = compact ? 1 : 1.5
  if (stacked) {
    return (
      <Stack spacing={gap} sx={{ minHeight: '100%' }}>
        {setup}
        <Paper elevation={0} sx={{ p: 1.5 }}>
          {card}
        </Paper>
        <Box sx={{ display: 'flex', height: 360 }}>{tree}</Box>
        {timeline}
        <Paper elevation={0} sx={{ p: 1.5 }}>
          <OperationPanel trace={trace} step={step} onSelect={pick} />
        </Paper>
      </Stack>
    )
  }
  return (
    <Stack direction="row" spacing={gap} sx={{ height: '100%', minHeight: 0 }}>
      <Stack spacing={gap} sx={{ flex: 1, minWidth: 0, minHeight: 0 }}>
        {setup}
        {tree}
        {timeline}
      </Stack>
      {panel}
    </Stack>
  )
}

function Labelled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Stack direction="row" spacing={0.5} sx={{ alignItems: 'center' }}>
      <Typography variant="caption" sx={{ color: INK_FAINT, fontFamily: MONO }}>
        {label}
      </Typography>
      {children}
    </Stack>
  )
}

// -- the card that asks for the next decision ---------------------------------

interface CardProps {
  session: Session
  live: boolean
  selected: number | null
  grades: Record<number, number>
  /** A state setter: grade with the functional form, so two quick clicks both land. */
  setGrades: React.Dispatch<React.SetStateAction<Record<number, number>>>
  note: string
  setNote: (n: string) => void
  problem: string | null
  act: (fn: () => void) => void
  finish: () => void
  back: () => void
  at: number
  download: () => void
  compare: () => void
  restart: () => void
}

function TurnCard(props: CardProps) {
  const { session, live, problem, act, finish } = props
  if (!live) {
    return (
      <Alert severity="info" variant="outlined" action={<Button size="small" onClick={props.back}>Back</Button>}>
        You are looking back at step {props.at + 1}. Decisions are made at the newest step.
      </Alert>
    )
  }

  const color = OP_COLOR[
    { select: 'selection', expand: 'expansion', evaluate: 'evaluation', simulate: 'simulation',
      backprop: 'backpropagation', reflect: 'reflection', done: 'result' }[session.phase]
  ]

  return (
    <Box
      sx={{
        borderLeft: `3px solid ${color}`,
        bgcolor: alpha(color, 0.06),
        borderRadius: '0 6px 6px 0',
        px: 1.5,
        py: 1.25,
      }}
    >
      <Stack direction="row" spacing={1} sx={{ alignItems: 'baseline', mb: 0.75 }}>
        <Typography variant="subtitle2" sx={{ color, textTransform: 'uppercase' }}>
          {session.phase === 'done' ? 'Search over' : 'Your move'}
        </Typography>
        {session.phase !== 'done' && (
          <Typography variant="caption" sx={{ color: INK_FAINT, fontFamily: MONO }}>
            iteration {session.iteration} of {session.cfg.iterations}
          </Typography>
        )}
        <Box sx={{ flex: 1 }} />
        {session.phase !== 'done' && (
          <Button size="small" onClick={finish} sx={{ color: INK_DIM, fontSize: '0.72rem' }}>
            Let LATS finish
          </Button>
        )}
      </Stack>

      {session.phase === 'select' && <SelectTurn {...props} />}
      {session.phase === 'expand' && (
        <Auto
          text={`The policy draws ${session.cfg.n} independent samples from #${session.target}, and each distinct move is executed. Samples that agree become one child; how many agreed is SC(s).`}
          label="Draw the samples"
          onClick={() => act(() => session.expand())}
        />
      )}
      {session.phase === 'evaluate' && <EvaluateTurn {...props} />}
      {session.phase === 'simulate' && (
        <Auto
          text="Roll out greedily from the best-valued child until one number is left. The model grades the states the rollout creates — scoring every one yourself would take all day."
          label="Run the rollout"
          onClick={() => act(() => session.simulate())}
        />
      )}
      {session.phase === 'backprop' && (
        <Auto
          text="Back the reward up the path: every ancestor's value becomes a running average that includes it."
          label="Back up the reward"
          onClick={() => act(() => session.backpropagate())}
        />
      )}
      {session.phase === 'reflect' && <ReflectTurn {...props} />}
      {session.phase === 'done' && <DoneTurn {...props} />}

      {problem && (
        <Typography variant="caption" sx={{ display: 'block', color: BAD, mt: 1 }}>
          {problem}
        </Typography>
      )}
    </Box>
  )
}

function Auto({ text, label, onClick }: { text: string; label: string; onClick: () => void }) {
  return (
    <Stack spacing={1}>
      <Typography variant="body2" sx={{ color: INK_DIM }}>
        {text}
      </Typography>
      <Box>
        <Button size="small" variant="contained" disableElevation onClick={onClick}>
          {label}
        </Button>
      </Box>
    </Stack>
  )
}

function SelectTurn({ session, selected, act }: CardProps) {
  const uct = session.uctTarget()
  const ok = selected !== null && session.choosable(selected)
  const node = ok ? session.node(selected!) : null
  return (
    <Stack spacing={1}>
      <Typography variant="body2" sx={{ color: INK_DIM }}>
        <b>Selection.</b> Click the leaf you would expand next. UCT weighs what a branch has
        earned against how little it has been tried; you can follow it or overrule it, and
        the selection panel will show you both.
      </Typography>
      {node && (
        <Box sx={{ border: `1px solid ${STROKE}`, borderRadius: 1, p: 1, bgcolor: SURFACE_2 }}>
          <Typography sx={{ fontFamily: MONO, fontSize: '0.78rem' }}>
            #{node.id} {node.steps.join(' → ') || 'root'}
          </Typography>
          <Typography variant="caption" sx={{ color: INK_DIM }}>
            {node.numbers.length > 1 ? `left: ${node.numbers.join(', ')}` : ''} · V{' '}
            {num(node.value)} · N {node.visits}
          </Typography>
        </Box>
      )}
      <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', rowGap: 1 }}>
        <Button
          size="small"
          variant="contained"
          disableElevation
          disabled={!ok}
          onClick={() => act(() => session.select(selected!, 'you'))}
        >
          {ok ? `Expand #${selected}` : 'Pick a leaf in the tree'}
        </Button>
        <Button
          size="small"
          variant="outlined"
          disabled={uct === null}
          onClick={() => act(() => session.select(uct!, 'uct'))}
        >
          Let UCT choose{uct !== null ? ` (#${uct})` : ''}
        </Button>
      </Stack>
    </Stack>
  )
}

/** Three marks a person can give at a glance, as Tree of Thoughts' value prompt does. */
const MARKS = [
  { v: 0.1, label: 'dead end' },
  { v: 0.5, label: 'maybe' },
  { v: 0.9, label: 'likely' },
]

function EvaluateTurn({ session, grades, setGrades, act }: CardProps) {
  const fresh = session.fresh
  const done = fresh.every((id) => grades[id] !== undefined)
  const model = session.modelGrades()
  return (
    <Stack spacing={1.25}>
      <Typography variant="body2" sx={{ color: INK_DIM }}>
        <b>Evaluation.</b> You are LM(s). The environment has replied to every new state —
        grade each one: how likely is 24 from here? Your grade is blended with SC(s) into
        V(s), and the search will steer by it.
      </Typography>
      {fresh.map((id) => {
        const n = session.node(id)
        const g = grades[id]
        return (
          <Box key={id} sx={{ border: `1px solid ${STROKE}`, borderRadius: 1, p: 1, bgcolor: SURFACE_2 }}>
            <Stack direction="row" spacing={1} sx={{ alignItems: 'baseline' }}>
              <Typography sx={{ fontFamily: MONO, fontSize: '0.78rem', color: ACCENT }}>
                #{id} {n.label}
              </Typography>
              <Box sx={{ flex: 1 }} />
              <Typography sx={{ fontFamily: MONO, fontSize: '0.78rem', color: g === undefined ? INK_FAINT : PRIMARY }}>
                {g === undefined ? '—' : num(g)}
              </Typography>
            </Stack>
            <Typography variant="caption" sx={{ display: 'block', color: INK_DIM }}>
              {n.observation}
            </Typography>
            <Stack direction="row" spacing={0.5} sx={{ mt: 0.5, alignItems: 'center' }}>
              {MARKS.map((m) => (
                <Chip
                  key={m.v}
                  size="small"
                  label={m.label}
                  onClick={() => setGrades((prev) => ({ ...prev, [id]: m.v }))}
                  variant={g === m.v ? 'filled' : 'outlined'}
                  sx={g === m.v ? { bgcolor: alpha(PRIMARY, 0.16), color: PRIMARY } : undefined}
                />
              ))}
              <Slider
                size="small"
                min={0}
                max={1}
                step={0.05}
                value={g ?? 0.5}
                onChange={(_, v) => setGrades((prev) => ({ ...prev, [id]: v as number }))}
                sx={{ ml: 1, flex: 1, minWidth: 60 }}
                aria-label={`grade for node ${id}`}
              />
            </Stack>
          </Box>
        )
      })}
      <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', rowGap: 1 }}>
        <Button
          size="small"
          variant="contained"
          disableElevation
          disabled={!done}
          onClick={() => act(() => session.evaluate(new Map(fresh.map((id) => [id, grades[id]]))))}
        >
          {done ? 'Submit my grades' : `Grade all ${fresh.length}`}
        </Button>
        <Button size="small" variant="outlined" onClick={() => act(() => session.evaluate(null))}>
          Let the model grade
        </Button>
      </Stack>
      {done && (
        <Typography variant="caption" sx={{ color: INK_FAINT }}>
          The model would have said{' '}
          {fresh.map((id) => `#${id} ${num(model.get(id))}`).join(', ')}.
        </Typography>
      )}
    </Stack>
  )
}

function ReflectTurn({ session, note, setNote, act }: CardProps) {
  const suggestion = session.suggestion()
  const rules = note.trim() ? session.wouldRuleOut(note) : []
  return (
    <Stack spacing={1}>
      <Typography variant="body2" sx={{ color: INK_DIM }}>
        <b>Reflection.</b> The trajectory failed. Write the note the next expansions will
        read. The stand-in policy cannot read prose, but it does read moves: a note that
        names one — <i>don’t open with 8 / 2</i> — makes it avoid that move from now on.
      </Typography>
      <TextField
        multiline
        minRows={2}
        size="small"
        placeholder="What went wrong, and what to avoid…"
        value={note}
        onChange={(e) => setNote(e.target.value)}
        sx={{ '& textarea': { fontSize: '0.82rem' } }}
      />
      <Typography variant="caption" sx={{ color: rules.length ? VIOLET : INK_FAINT }}>
        {note.trim()
          ? rules.length
            ? `This note rules out: ${rules.map((r) => r.split(' =')[0]).join(', ')}.`
            : 'This note names no move, so it will sit in context without changing what the policy proposes.'
          : `Suggested: ${suggestion.text}`}
      </Typography>
      <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', rowGap: 1 }}>
        <Button
          size="small"
          variant="contained"
          disableElevation
          disabled={!note.trim()}
          onClick={() =>
            act(() => {
              session.reflect(note)
              setNote('')
            })
          }
        >
          Add the note
        </Button>
        <Button size="small" variant="outlined" onClick={() => setNote(suggestion.text)}>
          Use the suggestion
        </Button>
        <Button
          size="small"
          onClick={() =>
            act(() => {
              session.reflect(null)
              setNote('')
            })
          }
          sx={{ color: INK_DIM }}
        >
          Skip
        </Button>
      </Stack>
    </Stack>
  )
}

function DoneTurn({ session, download, compare, restart }: CardProps) {
  const trace = session.toTrace()
  const graded = session.calibration()
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null)
  const live = graded.filter((g) => g.reachable)
  const dead = graded.filter((g) => !g.reachable)
  return (
    <Stack spacing={1.25}>
      <Typography variant="body2" sx={{ color: trace.result.solved ? GOOD : INK }}>
        {trace.result.solved
          ? `Solved in ${trace.result.iterations_run} iteration${trace.result.iterations_run === 1 ? '' : 's'}.`
          : `Not solved — ${trace.result.stopped_because}.`}
      </Typography>
      {graded.length > 0 && (
        <Box>
          <Typography variant="subtitle2" sx={{ color: INK_FAINT, textTransform: 'uppercase' }}>
            How good were your grades?
          </Typography>
          <Typography variant="body2" sx={{ color: INK_DIM }}>
            You graded {graded.length} state{graded.length === 1 ? '' : 's'}. On the{' '}
            {live.length} that could still reach 24 your average was{' '}
            <b>{num(mean(live.map((g) => g.yours)))}</b>; on the {dead.length} that could not,{' '}
            <b>{num(mean(dead.map((g) => g.yours)))}</b>. The model’s were{' '}
            {num(mean(live.map((g) => g.model)))} and {num(mean(dead.map((g) => g.model)))}.
            A value function earns its place by putting the first number above the second.
          </Typography>
          <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 0.5, mt: 0.75 }}>
            {graded.map((g) => (
              <Chip
                key={g.id}
                size="small"
                label={`#${g.id} ${g.label} · ${num(g.yours)}`}
                sx={{
                  fontFamily: MONO,
                  bgcolor: alpha(g.reachable ? GOOD : BAD, 0.1),
                  color: g.reachable ? GOOD : BAD,
                }}
                title={g.reachable ? '24 was still reachable from here' : 'no way to 24 from here'}
              />
            ))}
          </Stack>
        </Box>
      )}
      <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', rowGap: 1 }}>
        <Button size="small" variant="contained" disableElevation onClick={compare}>
          Compare with LATS
        </Button>
        <Button size="small" variant="outlined" onClick={download}>
          Download the trace
        </Button>
        <Button size="small" onClick={restart} sx={{ color: INK_DIM }}>
          Play again
        </Button>
      </Stack>
    </Stack>
  )
}
