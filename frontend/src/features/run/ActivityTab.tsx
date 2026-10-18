import { useMemo, useState } from 'react'
import { AlertTriangle, CircleDot, Cpu, Database, FlaskConical, ListChecks } from 'lucide-react'
import type { AnyEvent } from '@/api/events'
import type { RunModel, TaskView } from '@/lib/runModel'
import { bestExperiment, taskPhase } from '@/lib/runModel'
import { cn, fmt, formatClock, kpiLabel } from '@/lib/utils'
import { Badge, Card, CardHeader, Label, Mono } from '@/components/ui/primitives'
import { Callout, EmptyState } from '@/components/ui/feedback'
import { Disclosure, Markdown } from '@/components/ui/content'
import { KpiBadge } from '@/components/status'
import { describe, eventTaskId } from './describe'

function phaseTone(t: TaskView) {
  if (t.dropped || t.fullTrainError || (t.finalTest && !t.finalTest.passed)) return 'bad' as const
  if (t.finalTest?.passed) return 'ok' as const
  if (t.researchDone) return 'teal' as const
  return 'accent' as const
}

function MiniStat({ icon, label, value, sub }: { icon: React.ReactNode; label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="min-w-0 rounded-lg border border-line bg-surface-2 px-3 py-2">
      <div className="flex items-center gap-1.5 text-[11px] text-fg-subtle">
        {icon}
        {label}
      </div>
      <div className="mt-1 truncate text-sm font-medium tabular text-fg">{value}</div>
      {sub && <div className="truncate text-[11px] text-fg-subtle">{sub}</div>}
    </div>
  )
}

function TaskCard({ t, live }: { t: TaskView; live: boolean }) {
  const lastVal = t.validations.at(-1)
  const best = bestExperiment(t.experiments) ?? t.researchDone?.best ?? undefined
  const kept = t.experiments.filter((e) => e.status === 'keep').length
  const failed = t.experiments.filter((e) => !['keep', 'discard'].includes(e.status)).length
  const progress = t.lastProgress
  const progressParts = progress
    ? Object.entries(progress)
        .filter(([k]) => !['ts', 'task_id', 'line'].includes(k))
        .map(([k, v]) => `${k} ${typeof v === 'number' ? fmt(v) : String(v)}`)
    : []
  const primary = t.spec?.primary_metric
  const researching = live && !t.dropped && !t.researchDone && (t.proposals > 0 || t.experiments.length > 0)

  return (
    <Card className={cn(t.dropped && 'opacity-80')}>
      <div className="flex flex-wrap items-start justify-between gap-2 px-4 pt-4">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium text-fg">{t.spec?.name ?? t.id}</div>
          <Mono className="text-fg-subtle">{t.id}</Mono>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {t.spec && <Badge>{t.spec.prediction_format}</Badge>}
          <Badge tone={phaseTone(t)} dot pulse={researching}>
            {taskPhase(t)}
          </Badge>
        </div>
      </div>
      {t.spec?.framing && <p className="px-4 pt-2 text-xs leading-relaxed text-fg-muted">{t.spec.framing}</p>}

      <div className="grid grid-cols-2 gap-2 px-4 pt-3 sm:grid-cols-4">
        <MiniStat
          icon={<Database className="size-3" />}
          label="Streams"
          value={t.discovery ? `${t.discovery.kept} kept` : '—'}
          sub={t.discovery ? `of ${t.discovery.scanned} scanned` : 'not profiled yet'}
        />
        <MiniStat
          icon={<ListChecks className="size-3" />}
          label="prepare.py"
          value={lastVal ? (lastVal.ok ? 'Valid' : 'Invalid') : t.designAttempts ? 'Writing…' : '—'}
          sub={t.designAttempts ? `${t.designAttempts} attempt${t.designAttempts > 1 ? 's' : ''}` : undefined}
        />
        <MiniStat
          icon={<FlaskConical className="size-3" />}
          label="Experiments"
          value={t.experiments.length || '—'}
          sub={t.experiments.length ? `${kept} kept · ${failed} failed` : undefined}
        />
        <MiniStat
          icon={<CircleDot className="size-3" />}
          label="Best"
          value={best ? best.exp_id : '—'}
          sub={best && primary && best.metrics[primary] !== undefined ? `${primary} ${fmt(best.metrics[primary])}` : best ? `score ${fmt(best.score)}` : undefined}
        />
      </div>

      <div className="space-y-3 px-4 py-4">
        {t.spec?.kpis.length ? (
          <div>
            <Label className="mb-1.5">KPI targets {best ? `· ${best.exp_id} (validation)` : ''}</Label>
            <div className="flex flex-wrap gap-1.5">
              {best && Object.keys(best.kpis_passed).length
                ? Object.entries(best.kpis_passed).map(([k, ok]) => <KpiBadge key={k} label={k} passed={ok} />)
                : t.spec.kpis.map((k) => <KpiBadge key={kpiLabel(k)} label={kpiLabel(k)} passed={undefined} />)}
            </div>
          </div>
        ) : null}

        {researching && t.lastProposal && (
          <div className="rounded-lg border border-accent/20 bg-accent-soft/50 px-3 py-2 text-xs">
            <div className="flex items-center gap-1.5 font-medium text-accent">
              <Cpu className="size-3" /> {t.lastProposal.exp_id}
            </div>
            <div className="mt-0.5 text-fg-muted">{t.lastProposal.description || t.lastProposal.hypothesis}</div>
            {progressParts.length > 0 && <Mono className="mt-1 block text-[11px] text-fg-subtle">{progressParts.join(' · ')}</Mono>}
          </div>
        )}

        {t.dropped && (
          <Callout tone="bad" icon={<AlertTriangle className="size-3.5" />} title="Task dropped">
            {t.dropped}
          </Callout>
        )}
        {lastVal && !lastVal.ok && !t.dropped && (
          <Callout tone="warn" title={`Validation failed (attempt ${lastVal.attempt})`}>
            {lastVal.messages.join(' · ')}
          </Callout>
        )}
        {t.researchDone && (
          <div className="text-xs text-fg-muted">
            Research stopped after {t.researchDone.n_exp} experiments: <span className="text-fg">{t.researchDone.reason}</span>
          </div>
        )}

        {(t.notes || t.plan) && (
          <div className="space-y-2 border-t border-line pt-3">
            {t.notes && (
              <Disclosure summary="Exploration notes">
                <Markdown text={t.notes} className="rounded-lg border border-line bg-surface-2 p-3" />
              </Disclosure>
            )}
            {t.plan && (
              <Disclosure summary={`Research plan${t.planRevisions ? ` (revised ${t.planRevisions}×)` : ''}`}>
                <Markdown text={t.plan.text} className="rounded-lg border border-line bg-surface-2 p-3" />
              </Disclosure>
            )}
          </div>
        )}
      </div>
    </Card>
  )
}

const TONE_DOT = {
  neutral: 'bg-fg-subtle',
  accent: 'bg-accent',
  teal: 'bg-teal',
  ok: 'bg-ok',
  warn: 'bg-warn',
  bad: 'bg-bad',
}

export function LiveFeed({ events, limit = 150 }: { events: AnyEvent[]; limit?: number }) {
  const [showNoisy, setShowNoisy] = useState(false)
  const items = useMemo(() => {
    const out: { e: AnyEvent; i: number; line: ReturnType<typeof describe> }[] = []
    for (let i = events.length - 1; i >= 0 && out.length < limit; i--) {
      const line = describe(events[i])
      if (!showNoisy && line.noisy) continue
      out.push({ e: events[i], i, line })
    }
    return out
  }, [events, showNoisy, limit])

  return (
    <Card className="flex max-h-[720px] flex-col">
      <CardHeader
        title="Live activity"
        description={`${events.length} events`}
        actions={
          <label className="flex cursor-pointer items-center gap-1.5 text-[11px] text-fg-muted">
            <input type="checkbox" checked={showNoisy} onChange={(e) => setShowNoisy(e.target.checked)} className="accent-accent" />
            tool calls & progress
          </label>
        }
      />
      {items.length === 0 ? (
        <EmptyState title="No events yet" description="Events appear here as the agent works." />
      ) : (
        <ol className="min-h-0 flex-1 overflow-y-auto py-2">
          {items.map(({ e, i, line }) => (
            <li key={i} className="flex gap-3 px-4 py-1.5">
              <span className={cn('mt-1.5 size-1.5 shrink-0 rounded-full', TONE_DOT[line.tone])} />
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate text-xs text-fg">{line.title}</span>
                  <Mono className="shrink-0 text-[10.5px] text-fg-subtle">{formatClock(e.ts)}</Mono>
                </div>
                {line.detail && <div className="line-clamp-2 text-[11px] text-fg-muted">{line.detail}</div>}
                {!line.detail && eventTaskId(e) && <div className="text-[11px] text-fg-subtle">{eventTaskId(e)}</div>}
              </div>
            </li>
          ))}
        </ol>
      )}
    </Card>
  )
}

export function ActivityTab({ model, events, live }: { model: RunModel; events: AnyEvent[]; live: boolean }) {
  const tasks = model.taskOrder.map((id) => model.tasks[id])
  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="space-y-4">
        {model.warnings.length > 0 && (
          <Callout tone="warn" icon={<AlertTriangle className="size-3.5" />} title="Spec problems repaired automatically">
            {model.warnings.join(' · ')}
          </Callout>
        )}
        {tasks.length === 0 ? (
          <Card>
            <EmptyState
              icon={<ListChecks className="size-5" />}
              title={live ? 'Formulating tasks…' : 'No tasks'}
              description={live ? 'The agent is turning the goal into ML task specifications.' : 'This run produced no task specs.'}
            />
          </Card>
        ) : (
          tasks.map((t) => <TaskCard key={t.id} t={t} live={live} />)
        )}
      </div>
      <div className="xl:sticky xl:top-4 xl:self-start">
        <LiveFeed events={events} />
      </div>
    </div>
  )
}
