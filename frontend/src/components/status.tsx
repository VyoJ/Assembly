import type { ExperimentStatus } from '@/api/types'
import type { SseStatus } from '@/api/sse'
import { Badge, type Tone } from './ui/primitives'

const RUN_STATUS: Record<string, { label: string; tone: Tone; live?: boolean }> = {
  running: { label: 'Running', tone: 'accent', live: true },
  awaiting_approval: { label: 'Awaiting approval', tone: 'warn', live: true },
  done: { label: 'Done', tone: 'ok' },
  failed: { label: 'Failed', tone: 'bad' },
  rejected: { label: 'Rejected', tone: 'bad' },
  no_tasks: { label: 'No tasks', tone: 'neutral' },
}

export function RunStatusBadge({ status }: { status: string | undefined }) {
  if (!status) return <Badge>Unknown</Badge>
  const s = RUN_STATUS[status] ?? { label: status, tone: 'neutral' as Tone }
  return (
    <Badge tone={s.tone} dot pulse={s.live}>
      {s.label}
    </Badge>
  )
}

const EXP_STATUS: Record<ExperimentStatus, { label: string; tone: Tone }> = {
  keep: { label: 'keep', tone: 'ok' },
  discard: { label: 'discard', tone: 'neutral' },
  crash: { label: 'crash', tone: 'bad' },
  timeout: { label: 'timeout', tone: 'warn' },
  invalid_output: { label: 'invalid output', tone: 'bad' },
}

export function ExperimentStatusBadge({ status }: { status: ExperimentStatus | string }) {
  const s = EXP_STATUS[status as ExperimentStatus] ?? { label: status, tone: 'neutral' as Tone }
  return <Badge tone={s.tone}>{s.label}</Badge>
}

const STREAM_STATUS: Record<SseStatus, { label: string; tone: Tone; pulse?: boolean }> = {
  connecting: { label: 'Connecting', tone: 'neutral', pulse: true },
  open: { label: 'Live', tone: 'teal', pulse: true },
  reconnecting: { label: 'Reconnecting', tone: 'warn', pulse: true },
  closed: { label: 'Stream closed', tone: 'neutral' },
  error: { label: 'Stream error', tone: 'bad' },
}

export function StreamStatusBadge({ status, title }: { status: SseStatus; title?: string }) {
  const s = STREAM_STATUS[status]
  return (
    <span title={title}>
      <Badge tone={s.tone} dot pulse={s.pulse}>
        {s.label}
      </Badge>
    </span>
  )
}

export function KpiBadge({ label, passed }: { label: string; passed: boolean | undefined }) {
  return (
    <Badge tone={passed === undefined ? 'neutral' : passed ? 'ok' : 'bad'} className="font-mono">
      {label}
    </Badge>
  )
}
