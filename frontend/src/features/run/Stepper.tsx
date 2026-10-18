import { Check, CircleDashed, Clock, Loader2, Minus, Square, X } from 'lucide-react'
import { STAGES, type Stage } from '@/api/events'
import type { RunModel, StepState } from '@/lib/runModel'
import { cn, formatDuration } from '@/lib/utils'

const ICON: Record<StepState, React.ReactNode> = {
  pending: <CircleDashed className="size-3.5" />,
  active: <Loader2 className="size-3.5 animate-spin" />,
  waiting: <Clock className="size-3.5" />,
  complete: <Check className="size-3.5" />,
  failed: <X className="size-3.5" />,
  stopped: <Square className="size-3" />,
  skipped: <Minus className="size-3.5" />,
}

const RING: Record<StepState, string> = {
  pending: 'border-line-strong text-fg-subtle',
  active: 'border-accent bg-accent-soft text-accent',
  waiting: 'border-warn bg-warn-soft text-warn',
  complete: 'border-ok/50 bg-ok-soft text-ok',
  failed: 'border-bad bg-bad-soft text-bad',
  stopped: 'border-fg-subtle bg-surface-3 text-fg-muted',
  skipped: 'border-line text-fg-subtle/60',
}

const STATE_TEXT: Partial<Record<StepState, string>> = {
  waiting: 'Waiting for you',
  failed: 'Failed here',
  stopped: 'Stopped here',
  skipped: 'Not reached',
}

export function Stepper({ model, states, now }: { model: RunModel; states: Record<Stage, StepState>; now: number }) {
  return (
    <ol className="flex gap-0 overflow-x-auto pb-1">
      {STAGES.map((s, i) => {
        const st = states[s.id]
        const timing = model.stages[s.id]
        const end = timing?.endedAt ?? (st === 'active' || st === 'waiting' ? now / 1000 : undefined)
        const duration = timing?.startedAt !== undefined && end !== undefined ? end - timing.startedAt : undefined
        const last = i === STAGES.length - 1
        return (
          <li key={s.id} className="flex min-w-[118px] flex-1 flex-col">
            <div className="flex items-center">
              <div className={cn('flex size-7 shrink-0 items-center justify-center rounded-full border', RING[st])}>{ICON[st]}</div>
              {!last && (
                <div
                  className={cn(
                    'mx-2 h-px flex-1',
                    st === 'complete' ? 'bg-ok/40' : st === 'active' || st === 'waiting' ? 'bg-gradient-to-r from-accent/50 to-line' : 'bg-line',
                  )}
                />
              )}
            </div>
            <div className="mt-2 pr-3">
              <div className={cn('text-xs font-medium', st === 'pending' || st === 'skipped' ? 'text-fg-subtle' : 'text-fg')}>
                {s.label}
              </div>
              <div className="mt-0.5 text-[11px] text-fg-subtle">
                {STATE_TEXT[st] ?? (duration !== undefined ? formatDuration(duration) : s.hint)}
              </div>
            </div>
          </li>
        )
      })}
    </ol>
  )
}
