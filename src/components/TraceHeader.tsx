/**
 * The task being searched, which method searched it, and how this particular
 * run was configured - including, for a trace written before the scoring fix,
 * a warning chip that says so before anyone reads a number from it.
 */

import { useState } from 'react'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Collapse from '@mui/material/Collapse'
import IconButton from '@mui/material/IconButton'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Tooltip from '@mui/material/Tooltip'
import Typography from '@mui/material/Typography'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'

import type { Trace } from '../types'
import { num } from '../lib/format'
import { caveats, isLegacy, methodOf, settingsOf } from '../lib/methods'
import { ACCENT, GOOD, INK, INK_DIM, INK_FAINT, MONO, PRIMARY, TEAL, alpha } from '../theme'

/** Which algorithm ran, with its citation on hover. */
export function MethodChip({ trace }: { trace: Pick<Trace, 'method'> }) {
  const method = methodOf(trace)
  return (
    <Tooltip title={`${method.name} · ${method.cite}. ${method.blurb}`} arrow>
      <Chip
        size="small"
        label={method.short}
        sx={{
          bgcolor: alpha(PRIMARY, method.id === 'lats' ? 0.12 : 0.06),
          color: PRIMARY,
          border: `1px solid ${alpha(PRIMARY, 0.35)}`,
          fontWeight: 700,
        }}
      />
    </Tooltip>
  )
}

/** Shown on a trace written by version 1.0, which scored before execution. */
export function LegacyChip({ trace }: { trace: Trace }) {
  if (!isLegacy(trace)) return null
  return (
    <Tooltip title={caveats(trace).join(' ')} arrow>
      <Chip
        size="small"
        variant="outlined"
        label="v1.0 scoring"
        sx={{ color: ACCENT, borderColor: alpha(ACCENT, 0.6), fontWeight: 600 }}
      />
    </Tooltip>
  )
}

export default function TraceHeader({
  trace,
  tokens,
  compact,
}: {
  trace: Trace
  /** Tokens spent as of the current step, not the total for the run. */
  tokens: number
  /**
   * Phone: the prompt and the settings fold away behind the title, because on
   * a screen this size every line they take comes out of the tree.
   */
  compact?: boolean
}) {
  const { result, policy } = trace
  const [open, setOpen] = useState(false)

  const solved = (
    <Chip
      size="small"
      label={result.solved ? 'solved' : 'not solved'}
      sx={{
        bgcolor: alpha(result.solved ? GOOD : ACCENT, 0.16),
        color: result.solved ? GOOD : ACCENT,
        fontWeight: 700,
      }}
    />
  )

  const spent = (
    <Tooltip
      title={
        policy.tokens_are_estimated
          ? 'Estimated at four characters per token: the offline policy makes no API calls, so there is nothing to measure.'
          : 'Reported by the API.'
      }
    >
      <Typography sx={{ fontFamily: MONO, fontSize: '0.75rem', color: TEAL }}>
        {tokens.toLocaleString()} tokens{policy.tokens_are_estimated ? '*' : ''}
      </Typography>
    </Tooltip>
  )

  const settings = (
    <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
      {settingsOf(trace)
        // The header has room for the knobs, not the budget; the init step
        // lists everything.
        .filter((s) => s.label !== 'iterations')
        .map((s) => (
          <Meta key={s.label} label={s.label} value={s.value} title={s.title} />
        ))}
      <Meta
        label="best"
        value={num(result.best_reward)}
        title={`stopped because ${result.stopped_because}`}
      />
      <Meta
        label="policy"
        value={policy.kind === 'mock' ? `mock · seed ${policy.seed}` : (policy.model ?? policy.name)}
        title={
          policy.kind === 'mock'
            ? 'Deterministic offline sampler, and a per-task heuristic in place of the model’s grade. The environment is real; the policy is not.'
            : 'Sampled from a real model.'
        }
      />
    </Stack>
  )

  if (compact) {
    return (
      <Paper elevation={0} sx={{ px: 1.25, py: 0.6 }}>
        <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center', minWidth: 0 }}>
          <Typography
            noWrap
            sx={{ fontWeight: 650, fontSize: '0.9rem', color: INK, flex: 1, minWidth: 0 }}
          >
            {trace.task.title}
          </Typography>
          <MethodChip trace={trace} />
          {solved}
          {spent}
          <IconButton
            size="small"
            onClick={() => setOpen((o) => !o)}
            aria-label={open ? 'hide the task and its settings' : 'show the task and its settings'}
            sx={{ flexShrink: 0, color: INK_FAINT, p: 0.25 }}
          >
            <ExpandMoreIcon
              fontSize="small"
              sx={{
                transform: open ? 'rotate(180deg)' : 'none',
                transition: 'transform 160ms ease',
              }}
            />
          </IconButton>
        </Stack>
        <Collapse in={open} unmountOnExit>
          <Typography variant="body2" sx={{ color: INK_DIM, mt: 0.5 }}>
            {trace.task.prompt}
          </Typography>
          {isLegacy(trace) && (
            <Box sx={{ mt: 0.75 }}>
              <LegacyChip trace={trace} />
            </Box>
          )}
          {settings}
        </Collapse>
      </Paper>
    )
  }

  return (
    <Paper elevation={0} sx={{ px: 1.5, py: 1.25 }}>
      <Stack
        direction="row"
        spacing={1}
        sx={{ alignItems: 'center', flexWrap: 'wrap', rowGap: 0.5 }}
      >
        <Typography variant="h6" sx={{ color: INK }}>
          {trace.task.title}
        </Typography>
        <MethodChip trace={trace} />
        {solved}
        <LegacyChip trace={trace} />
        <Box sx={{ flex: 1 }} />
        {spent}
      </Stack>

      <Typography variant="body2" sx={{ color: INK_DIM, mt: 0.25 }}>
        {trace.task.prompt}
      </Typography>

      {settings}
    </Paper>
  )
}

function Meta({
  label,
  value,
  title,
}: {
  label: string
  value: string
  title: string
}) {
  return (
    <Tooltip title={title} arrow>
      <Chip
        size="small"
        variant="outlined"
        label={
          <>
            <Box component="span" sx={{ color: INK_FAINT }}>
              {label}
            </Box>{' '}
            <Box component="span" sx={{ fontFamily: MONO }}>
              {value}
            </Box>
          </>
        }
      />
    </Tooltip>
  )
}
