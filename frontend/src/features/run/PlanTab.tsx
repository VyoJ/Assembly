import { useMemo, useState } from 'react'
import { AlertTriangle, Check, CheckCircle2, Clock, Info, Plus, Trash2, X, XCircle } from 'lucide-react'
import { api, ApiError } from '@/api/client'
import type { ApprovalBody, KPI, TaskSpec } from '@/api/types'
import type { RunModel, TaskView } from '@/lib/runModel'
import { cn, formatClock, kpiLabel } from '@/lib/utils'
import { Badge, Button, Card, CardHeader, Label, Mono, inputClass } from '@/components/ui/primitives'
import { Callout, EmptyState } from '@/components/ui/feedback'
import { Disclosure, KeyValue, Markdown } from '@/components/ui/content'

// ---- read-only spec

function SpecDetails({ spec, task }: { spec: TaskSpec; task?: TaskView }) {
  return (
    <div className="space-y-4">
      <p className="text-[13px] leading-relaxed text-fg-muted">{spec.rationale}</p>
      <KeyValue
        items={[
          ['Framing', spec.framing],
          ['Target', spec.target_definition],
          ['Prediction format', <Mono key="f">{spec.prediction_format}</Mono>],
          ['Split', spec.split_strategy],
          ['Label sources', spec.label_sources.map((l) => <Mono key={l} className="mr-2">{l}</Mono>)],
          [
            'Input streams',
            <span key="s" className="flex flex-wrap gap-1">
              {spec.input_streams.map((s) => (
                <Mono key={s} className="rounded bg-surface-3 px-1.5 py-0.5 text-[11px]">
                  {s}
                </Mono>
              ))}
            </span>,
          ],
        ]}
      />
      {task?.discovery && (
        <Disclosure summary={`Stream profiles: ${task.discovery.kept} kept of ${task.discovery.scanned} scanned`}>
          <div className="overflow-x-auto rounded-lg border border-line">
            <table className="w-full text-xs">
              <thead className="bg-surface-2 text-left text-fg-subtle">
                <tr>
                  <th className="px-3 py-1.5 font-medium">Stream</th>
                  <th className="px-3 py-1.5 font-medium">Relevance</th>
                  <th className="px-3 py-1.5 font-medium">Kept</th>
                  <th className="px-3 py-1.5 font-medium">Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {[...task.discovery.profiles]
                  .sort((a, b) => b.relevance - a.relevance)
                  .map((p) => (
                    <tr key={p.stream_id}>
                      <td className="px-3 py-1.5 font-mono text-[11px] text-fg">{p.stream_id}</td>
                      <td className="px-3 py-1.5 tabular text-fg-muted">{p.relevance}</td>
                      <td className="px-3 py-1.5">{p.keep ? <Check className="size-3.5 text-ok" /> : <X className="size-3.5 text-fg-subtle" />}</td>
                      <td className="px-3 py-1.5 text-fg-muted">{p.reason}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </Disclosure>
      )}
      {task && task.validations.length > 0 && (
        <Disclosure summary={`Dataset validation: ${task.validations.length} attempt${task.validations.length > 1 ? 's' : ''}`}>
          <div className="space-y-2">
            {task.validations.map((v) => (
              <div key={v.attempt} className="rounded-lg border border-line bg-surface-2 px-3 py-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-fg">Attempt {v.attempt}</span>
                  <Badge tone={v.ok ? 'ok' : 'bad'}>{v.ok ? 'passed' : 'failed'}</Badge>
                </div>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {Object.entries(v.checks).map(([k, ok]) => (
                    <Badge key={k} tone={ok ? 'ok' : 'bad'} className="font-mono">
                      {ok ? '✓' : '✗'} {k}
                    </Badge>
                  ))}
                </div>
                {v.messages.length > 0 && <ul className="mt-1.5 list-disc pl-4 text-[11px] text-fg-muted">{v.messages.map((m, i) => <li key={i}>{m}</li>)}</ul>}
              </div>
            ))}
          </div>
        </Disclosure>
      )}
    </div>
  )
}

function SpecTargets({ spec }: { spec: TaskSpec }) {
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <div>
        <Label>Primary metric</Label>
        <Mono className="mt-1 block text-fg">{spec.primary_metric}</Mono>
      </div>
      <div>
        <Label>Time budget / experiment</Label>
        <div className="mt-1 text-sm tabular text-fg">{spec.time_budget_s}s</div>
      </div>
      <div>
        <Label>Metrics</Label>
        <div className="mt-1 flex flex-wrap gap-1">
          {spec.metrics.map((m) => (
            <Mono key={m} className="text-fg-muted">
              {m}
            </Mono>
          ))}
        </div>
      </div>
      <div className="sm:col-span-3">
        <Label>KPIs</Label>
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {spec.kpis.length ? spec.kpis.map((k) => <Badge key={kpiLabel(k)} tone="accent" className="font-mono">{kpiLabel(k)}</Badge>) : <span className="text-xs text-fg-subtle">none</span>}
        </div>
      </div>
    </div>
  )
}

// ---- editable form

interface Draft {
  kpis: KPI[]
  time_budget_s: number
  primary_metric: string
  metrics: string[]
  drop: boolean
}

const draftOf = (s: TaskSpec): Draft => ({
  kpis: s.kpis.map((k) => ({ ...k })),
  time_budget_s: s.time_budget_s,
  primary_metric: s.primary_metric,
  metrics: [...s.metrics],
  drop: false,
})

const changed = (s: TaskSpec, d: Draft) =>
  JSON.stringify([s.kpis, s.time_budget_s, s.primary_metric, s.metrics]) !==
  JSON.stringify([d.kpis, d.time_budget_s, d.primary_metric, d.metrics])

/** Metric names known valid for this task: everything the (sanitized) spec already references. */
const knownMetrics = (s: TaskSpec) => [...new Set([s.primary_metric, ...s.metrics, ...s.kpis.map((k) => k.metric)])]

function TaskEditor({
  spec,
  draft,
  onChange,
  disabled,
}: {
  spec: TaskSpec
  draft: Draft
  onChange: (d: Draft) => void
  disabled: boolean
}) {
  const options = knownMetrics(spec)
  const set = (patch: Partial<Draft>) => onChange({ ...draft, ...patch })
  const setKpi = (i: number, patch: Partial<KPI>) => set({ kpis: draft.kpis.map((k, j) => (j === i ? { ...k, ...patch } : k)) })
  const metricProblems = !draft.metrics.includes(draft.primary_metric)

  return (
    <fieldset disabled={disabled || draft.drop} className={cn('space-y-4', draft.drop && 'opacity-40')}>
      <div className="grid gap-4 sm:grid-cols-2">
        <label className="block">
          <Label className="mb-1.5">Primary metric</Label>
          <select value={draft.primary_metric} onChange={(e) => set({ primary_metric: e.target.value })} className={inputClass}>
            {options.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <label className="block">
          <Label className="mb-1.5">Time budget per experiment (s)</Label>
          <input
            type="number"
            min={1}
            step={1}
            value={draft.time_budget_s}
            onChange={(e) => set({ time_budget_s: Math.max(1, Math.round(Number(e.target.value) || 0)) })}
            className={cn(inputClass, 'tabular')}
          />
          <span className="mt-1 block text-[11px] text-fg-subtle">The backend clamps this to its MIN/MAX_BUDGET_S bounds.</span>
        </label>
      </div>

      <div>
        <Label className="mb-1.5">Metrics to report</Label>
        <div className="flex flex-wrap gap-2">
          {options.map((m) => {
            const on = draft.metrics.includes(m)
            return (
              <button
                key={m}
                type="button"
                onClick={() => set({ metrics: on ? draft.metrics.filter((x) => x !== m) : [...draft.metrics, m] })}
                className={cn(
                  'rounded-md border px-2 py-1 font-mono text-[11px] transition-colors',
                  on ? 'border-accent/40 bg-accent-soft text-accent' : 'border-line text-fg-subtle hover:text-fg',
                )}
              >
                {m}
              </button>
            )
          })}
        </div>
        {metricProblems && <div className="mt-1.5 text-[11px] text-warn">The primary metric should be one of the reported metrics.</div>}
      </div>

      <div>
        <div className="mb-1.5 flex items-center justify-between">
          <Label>KPIs</Label>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={() => set({ kpis: [...draft.kpis, { metric: draft.primary_metric, op: '>=', threshold: 0 }] })}
          >
            <Plus className="size-3" /> Add KPI
          </Button>
        </div>
        <div className="space-y-2">
          {draft.kpis.map((k, i) => (
            <div key={i} className="grid grid-cols-[1fr_72px_1fr_auto] gap-2">
              <select value={k.metric} onChange={(e) => setKpi(i, { metric: e.target.value })} className={cn(inputClass, 'font-mono text-xs')}>
                {options.map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
              <select value={k.op} onChange={(e) => setKpi(i, { op: e.target.value as KPI['op'] })} className={cn(inputClass, 'font-mono')}>
                <option value=">=">≥</option>
                <option value="<=">≤</option>
              </select>
              <input
                type="number"
                step="any"
                value={Number.isFinite(k.threshold) ? k.threshold : ''}
                onChange={(e) => setKpi(i, { threshold: e.target.value === '' ? NaN : Number(e.target.value) })}
                className={cn(inputClass, 'tabular')}
              />
              <Button type="button" size="icon" variant="ghost" onClick={() => set({ kpis: draft.kpis.filter((_, j) => j !== i) })} aria-label="Remove KPI">
                <Trash2 className="size-3.5" />
              </Button>
            </div>
          ))}
          {draft.kpis.length === 0 && <div className="text-[11px] text-warn">No KPIs: research will only optimise the primary metric.</div>}
        </div>
      </div>
    </fieldset>
  )
}

// ---- tab

export function PlanTab({
  runId,
  model,
  status,
  onDecision,
}: {
  runId: string
  model: RunModel
  status: string | undefined
  onDecision: () => void
}) {
  const request = model.approvalRequest
  const outcome = model.approvalOutcome
  const awaiting = status === 'awaiting_approval' && !!request && !outcome
  const initialDrafts = useMemo(
    () => Object.fromEntries(Object.entries(request?.tasks ?? {}).map(([id, t]) => [id, draftOf(t.spec)])),
    [request],
  )
  const [drafts, setDrafts] = useState<Record<string, Draft>>({})
  const [reason, setReason] = useState('')
  const [submitting, setSubmitting] = useState<'approve' | 'reject' | null>(null)
  const [submitted, setSubmitted] = useState<'approve' | 'reject' | null>(null)
  const [error, setError] = useState<string | null>(null)

  const draft = (id: string) => drafts[id] ?? initialDrafts[id]
  const ids = Object.keys(request?.tasks ?? {})
  const kept = ids.filter((id) => !draft(id)?.drop)
  const invalid = ids.some((id) => {
    const d = draft(id)
    return !d.drop && (d.kpis.some((k) => !Number.isFinite(k.threshold)) || d.metrics.length === 0)
  })

  const submit = async (approved: boolean) => {
    if (!request) return
    setSubmitting(approved ? 'approve' : 'reject')
    setError(null)
    const body: ApprovalBody = { approved, tasks: [], drop: [], reason: approved ? '' : reason.trim() }
    if (approved) {
      for (const id of ids) {
        const d = draft(id)
        if (d.drop) body.drop.push(id)
        else if (changed(request.tasks[id].spec, d))
          body.tasks.push({ id, kpis: d.kpis, time_budget_s: d.time_budget_s, primary_metric: d.primary_metric, metrics: d.metrics })
      }
    }
    try {
      await api.approve(runId, body)
      setSubmitted(approved ? 'approve' : 'reject')
      onDecision()
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setError('The run is no longer waiting for approval (it was resolved elsewhere or the agent moved on).')
        onDecision()
      } else setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setSubmitting(null)
    }
  }

  // ---- header state
  let banner: React.ReactNode = null
  if (outcome?.approved)
    banner = (
      <Callout tone="ok" icon={<CheckCircle2 className="size-4" />} title={request ? 'Plan approved' : 'Plan auto-approved'}>
        Approved at {formatClock(outcome.ts)} · {outcome.tasks.length} task(s) frozen for research. The specs below are the final, frozen versions.
      </Callout>
    )
  else if (outcome && !outcome.approved)
    banner = (
      <Callout tone="bad" icon={<XCircle className="size-4" />} title="Plan rejected">
        {outcome.reason ? `Reason: ${outcome.reason}` : 'No reason given.'} The run stopped before research.
      </Callout>
    )
  else if (submitted)
    banner = (
      <Callout tone="accent" icon={<Clock className="size-4" />} title="Decision submitted">
        Waiting for the agent to acknowledge…
      </Callout>
    )
  else if (awaiting)
    banner = (
      <Callout tone="warn" icon={<Clock className="size-4" />} title="Your approval is required">
        Review each task's specification and data card. You can adjust KPIs, metrics and time budgets, drop tasks, or reject the plan.
        Other fields are fixed by the approval API.
      </Callout>
    )
  else if (status === 'awaiting_approval')
    banner = <Callout tone="warn" icon={<Clock className="size-4" />} title="Approval pending">Loading the plan from the event stream…</Callout>
  else if (!request && model.taskOrder.length)
    banner = (
      <Callout icon={<Info className="size-4" />} title="Draft plan">
        These specs are still being worked on. {model.autoApprove ? 'This run auto-approves its plan.' : 'Approval opens once prepare.py is validated for each task.'}
      </Callout>
    )

  // Which specs to show: final (approved) > under review > drafts
  const approvedSpecs = outcome?.approved ? Object.fromEntries(outcome.tasks.map((t) => [t.id, t])) : null
  const specIds = approvedSpecs ? Object.keys(approvedSpecs) : request ? ids : model.taskOrder.filter((id) => model.tasks[id].spec)
  const droppedIds = model.taskOrder.filter((id) => model.tasks[id].dropped && !specIds.includes(id))

  if (!specIds.length && !droppedIds.length)
    return (
      <Card>
        <EmptyState title="No plan yet" description="Task specifications appear once the agent formulates them." />
      </Card>
    )

  return (
    <div className="space-y-4">
      {banner}
      {error && <Callout tone="bad" icon={<AlertTriangle className="size-4" />}>{error}</Callout>}

      {specIds.map((id) => {
        const task = model.tasks[id]
        const spec = approvedSpecs?.[id] ?? request?.tasks[id]?.spec ?? task?.spec
        if (!spec) return null
        const dataCard = request?.tasks[id]?.data_card
        const d = awaiting ? draft(id) : undefined
        return (
          <Card key={id} className={cn(d?.drop && 'border-dashed')}>
            <CardHeader
              title={spec.name}
              description={<Mono>{spec.id}</Mono>}
              actions={
                <>
                  <Badge>{spec.prediction_format}</Badge>
                  {awaiting && d && (
                    <Button
                      size="sm"
                      variant={d.drop ? 'secondary' : 'ghost'}
                      onClick={() => setDrafts((all) => ({ ...all, [id]: { ...d, drop: !d.drop } }))}
                      disabled={!!submitting || !!submitted}
                    >
                      {d.drop ? 'Keep task' : 'Drop task'}
                    </Button>
                  )}
                </>
              }
            />
            <div className="grid gap-6 p-4 lg:grid-cols-2">
              <div className="space-y-4">
                <SpecDetails spec={spec} task={task} />
              </div>
              <div className="space-y-4">
                <div className="rounded-lg border border-line bg-surface-2 p-4">
                  <div className="mb-3 text-xs font-medium text-fg">{awaiting ? 'Targets (editable)' : 'Targets'}</div>
                  {awaiting && d ? (
                    <TaskEditor
                      spec={spec}
                      draft={d}
                      disabled={!!submitting || !!submitted}
                      onChange={(nd) => setDrafts((all) => ({ ...all, [id]: nd }))}
                    />
                  ) : (
                    <SpecTargets spec={spec} />
                  )}
                </div>
                {dataCard ? (
                  <div className="rounded-lg border border-line p-4">
                    <div className="mb-1 text-xs font-medium text-fg">Data card</div>
                    <Markdown text={dataCard} className="max-h-96 overflow-y-auto" />
                  </div>
                ) : null}
              </div>
            </div>
          </Card>
        )
      })}

      {droppedIds.length > 0 && (
        <Card>
          <CardHeader title="Dropped tasks" />
          <ul className="divide-y divide-line">
            {droppedIds.map((id) => (
              <li key={id} className="px-4 py-2.5 text-xs">
                <Mono className="text-fg">{id}</Mono>
                <div className="mt-0.5 text-fg-muted">{model.tasks[id].dropped}</div>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {awaiting && !submitted && (
        <Card className="sticky bottom-4 z-10 border-line-strong bg-surface/95 shadow-2xl backdrop-blur">
          <div className="flex flex-col gap-3 p-4 lg:flex-row lg:items-center">
            <input
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Reason (sent with a rejection)"
              className={cn(inputClass, 'lg:max-w-sm')}
            />
            <div className="flex-1 text-xs text-fg-muted">
              {kept.length === ids.length ? `Approve all ${ids.length} task(s)` : `Approve ${kept.length} of ${ids.length} task(s)`}
              {kept.length === 0 && <span className="text-warn"> — dropping every task ends the run without research</span>}
            </div>
            <div className="flex gap-2">
              <Button variant="danger" onClick={() => submit(false)} loading={submitting === 'reject'} disabled={!!submitting}>
                <XCircle className="size-4" /> Reject
              </Button>
              <Button variant="success" onClick={() => submit(true)} loading={submitting === 'approve'} disabled={!!submitting || invalid}>
                <CheckCircle2 className="size-4" /> Approve plan
              </Button>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
