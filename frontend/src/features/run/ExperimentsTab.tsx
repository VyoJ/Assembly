import { useEffect, useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { FlaskConical, Info, Star } from 'lucide-react'
import { api } from '@/api/client'
import type { ExperimentResult } from '@/api/types'
import type { RunModel } from '@/lib/runModel'
import { bestExperiment } from '@/lib/runModel'
import { useApi } from '@/hooks/useApi'
import { cn, fmt, formatDuration, kpiLabel } from '@/lib/utils'
import { Badge, Card, CardHeader, Label, Mono, Skeleton, Stat } from '@/components/ui/primitives'
import { EmptyState, ErrorState } from '@/components/ui/feedback'
import { CodeBlock, KeyValue, LongText } from '@/components/ui/content'
import { ExperimentStatusBadge, KpiBadge } from '@/components/status'
import { ChartTooltip, LegendDot } from '@/components/charts'
import { axisProps, CHART } from '@/components/chartTheme'

const STATUS_COLOR: Record<string, string> = { keep: CHART.ok, discard: CHART.neutral }
const statusColor = (s: string) => STATUS_COLOR[s] ?? CHART.bad

const SCORE = '__score__'

function MetricChart({ rows, metric, kpis }: { rows: ExperimentResult[]; metric: string; kpis: { op: string; threshold: number }[] }) {
  const data = rows.map((r) => ({
    exp: r.exp_id.replace(/^exp_/, ''),
    exp_id: r.exp_id,
    status: r.status,
    value: metric === SCORE ? r.score : r.metrics[metric],
  }))
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 12, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid stroke={CHART.grid} vertical={false} />
          <XAxis dataKey="exp" {...axisProps} />
          <YAxis {...axisProps} width={56} tickFormatter={(v: number) => fmt(v, 3)} domain={['auto', 'auto']} />
          {kpis.map((k) => (
            <ReferenceLine
              key={`${k.op}${k.threshold}`}
              y={k.threshold}
              stroke={CHART.warn}
              strokeDasharray="4 4"
              label={{ value: `KPI ${k.op} ${k.threshold}`, fill: CHART.warn, fontSize: 10, position: 'insideTopRight' }}
            />
          ))}
          <Tooltip
            cursor={{ stroke: CHART.axis, strokeDasharray: '3 3' }}
            content={({ active, payload }) => {
              const p = active && payload?.[0]?.payload
              if (!p) return null
              return (
                <ChartTooltip
                  title={p.exp_id}
                  rows={[
                    { label: metric === SCORE ? 'score' : metric, value: p.value ?? '—' },
                    { label: 'status', value: p.status },
                  ]}
                />
              )
            }}
          />
          <Line
            type="linear"
            dataKey="value"
            stroke={CHART.line}
            strokeWidth={2}
            connectNulls
            isAnimationActive={false}
            dot={(props: { cx?: number; cy?: number; payload?: { status: string; exp_id: string }; index?: number }) => {
              const { cx, cy, payload } = props
              if (cx == null || cy == null || !payload) return <g key={props.index} />
              return <circle key={payload.exp_id} cx={cx} cy={cy} r={4.5} fill={statusColor(payload.status)} stroke={CHART.surface} strokeWidth={2} />
            }}
            activeDot={{ r: 6, stroke: CHART.surface, strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function ExperimentDetail({ row, live }: { row: ExperimentResult; live?: ExperimentResult }) {
  const diff = live?.diff
  const log = live?.log_tail
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Mono className="text-sm text-fg">{row.exp_id}</Mono>
        <ExperimentStatusBadge status={row.status} />
      </div>
      {row.hypothesis && (
        <div>
          <Label>Hypothesis</Label>
          <p className="mt-1 text-[13px] leading-relaxed text-fg-muted">{row.hypothesis}</p>
        </div>
      )}
      {row.description && (
        <div>
          <Label>Change</Label>
          <p className="mt-1 text-[13px] leading-relaxed text-fg-muted">{row.description}</p>
        </div>
      )}
      <KeyValue
        items={[
          ['Score (lower is better)', <Mono key="s">{fmt(row.score, 6)}</Mono>],
          ['Train time', formatDuration(row.train_seconds)],
          ['Peak memory', row.peak_mem_gb != null ? `${row.peak_mem_gb} GB` : '—'],
        ]}
      />
      {Object.keys(row.kpis_passed).length > 0 && (
        <div>
          <Label className="mb-1.5">KPIs (validation)</Label>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(row.kpis_passed).map(([k, ok]) => (
              <KpiBadge key={k} label={k} passed={ok} />
            ))}
          </div>
        </div>
      )}
      {Object.keys(row.metrics).length > 0 && (
        <div>
          <Label className="mb-1.5">Validation metrics</Label>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-3">
            {Object.entries(row.metrics).map(([k, v]) => (
              <div key={k} className="flex justify-between gap-2 text-xs">
                <span className="truncate font-mono text-fg-subtle">{k}</span>
                <span className="font-mono text-fg tabular">{fmt(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {diff ? (
        <div>
          <Label className="mb-1.5">train.py diff</Label>
          <CodeBlock diff>{diff}</CodeBlock>
        </div>
      ) : null}
      {log ? (
        <div>
          <Label className="mb-1.5">Log tail</Label>
          <CodeBlock maxHeight="max-h-56">{log}</CodeBlock>
        </div>
      ) : null}
      {!live && (
        <div className="flex gap-2 text-[11px] text-fg-subtle">
          <Info className="mt-px size-3 shrink-0" />
          Diff and log tail are only carried by the live event stream; they are on disk in the run workspace.
        </div>
      )}
    </div>
  )
}

function TaskExperiments({ runId, model, taskId }: { runId: string; model: RunModel; taskId: string }) {
  const task = model.tasks[taskId]
  const liveCount = task?.experiments.length ?? 0
  const q = useApi(() => api.experiments(runId, taskId), [runId, taskId])
  const { refresh } = q
  // The endpoint reads experiments.jsonl; refetch whenever the stream reports a new experiment.
  useEffect(() => {
    if (liveCount) void refresh()
  }, [liveCount, refresh])

  const rows = useMemo(() => q.data ?? [], [q.data])
  const best = bestExperiment(rows)
  const spec = task?.spec
  const metricNames = useMemo(() => [...new Set(rows.flatMap((r) => Object.keys(r.metrics)))], [rows])
  const [metric, setMetric] = useState<string>()
  const chosen = metric ?? (spec && metricNames.includes(spec.primary_metric) ? spec.primary_metric : metricNames[0] ?? SCORE)
  const [selected, setSelected] = useState<string>()
  const selectedRow = rows.find((r) => r.exp_id === selected) ?? best ?? rows.at(-1)
  const liveById = useMemo(() => Object.fromEntries((task?.experiments ?? []).map((e) => [e.exp_id, e])), [task?.experiments])

  if (q.loading && !q.data)
    return (
      <div className="space-y-3">
        <Skeleton className="h-24" />
        <Skeleton className="h-64" />
      </div>
    )
  if (q.error && !q.data) return <ErrorState error={q.error} onRetry={q.refresh} />
  if (!rows.length)
    return (
      <Card>
        <EmptyState
          icon={<FlaskConical className="size-5" />}
          title="No experiments yet"
          description={
            task?.dropped ? (
              <>
                This task was dropped: <LongText text={task.dropped} />
              </>
            ) : (
              'Experiments appear once the plan is approved and the research loop runs train.py.'
            )
          }
        />
      </Card>
    )

  const counts = rows.reduce<Record<string, number>>((a, r) => ({ ...a, [r.status]: (a[r.status] ?? 0) + 1 }), {})
  const kpisForMetric = spec?.kpis.filter((k) => k.metric === chosen) ?? []

  return (
    <div className="space-y-4">
      <Card className="grid grid-cols-2 gap-4 p-4 md:grid-cols-4">
        <Stat label="Experiments" value={rows.length} sub={`${counts.keep ?? 0} kept · ${counts.discard ?? 0} discarded`} />
        <Stat
          label="Failures"
          value={rows.length - (counts.keep ?? 0) - (counts.discard ?? 0)}
          sub={['crash', 'timeout', 'invalid_output'].filter((s) => counts[s]).map((s) => `${counts[s]} ${s}`).join(' · ') || 'none'}
        />
        <Stat label="Best experiment" value={best?.exp_id ?? '—'} sub={best ? `score ${fmt(best.score)}` : undefined} />
        <Stat
          label={spec ? `Best ${spec.primary_metric}` : 'Primary metric'}
          value={best && spec ? fmt(best.metrics[spec.primary_metric]) : '—'}
          sub={task?.researchDone ? `stopped: ${task.researchDone.reason}` : 'research in progress'}
        />
      </Card>

      <Card>
        <CardHeader
          title="Experiment history"
          description={chosen === SCORE ? 'Score combines KPI penalties and the primary metric; lower is better.' : 'Validation split'}
          actions={
            <select
              value={chosen}
              onChange={(e) => setMetric(e.target.value)}
              className="h-7 rounded-md border border-line-strong bg-surface-2 px-2 font-mono text-[11px] text-fg"
              aria-label="Metric"
            >
              {metricNames.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
              <option value={SCORE}>score (lower is better)</option>
            </select>
          }
        />
        <div className="px-2 pt-2">
          <MetricChart rows={rows} metric={chosen} kpis={kpisForMetric} />
        </div>
        <div className="flex flex-wrap gap-4 px-4 pt-1 pb-3">
          <LegendDot color={CHART.ok} label="keep (new best)" />
          <LegendDot color={CHART.neutral} label="discard" />
          <LegendDot color={CHART.bad} label="crash / timeout / invalid" />
          {kpisForMetric.length > 0 && <span className="text-[11px] text-warn">- - KPI threshold</span>}
        </div>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
        <Card className="self-start overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="border-b border-line bg-surface-2 text-left text-fg-subtle">
                <tr>
                  <th className="px-3 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 text-right font-medium">Score</th>
                  {spec && <th className="px-3 py-2 text-right font-medium">{spec.primary_metric}</th>}
                  <th className="px-3 py-2 font-medium">KPIs</th>
                  <th className="px-3 py-2 font-medium">Change</th>
                  <th className="px-3 py-2 text-right font-medium">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {[...rows].reverse().map((r) => {
                  const kp = Object.values(r.kpis_passed)
                  return (
                    <tr
                      key={r.exp_id}
                      onClick={() => setSelected(r.exp_id)}
                      className={cn('cursor-pointer transition-colors hover:bg-surface-2', selectedRow?.exp_id === r.exp_id && 'bg-surface-2')}
                    >
                      <td className="px-3 py-2 font-mono whitespace-nowrap text-fg">
                        {r.exp_id}
                        {best?.exp_id === r.exp_id && <Star className="ml-1 inline size-3 fill-teal text-teal" />}
                      </td>
                      <td className="px-3 py-2">
                        <ExperimentStatusBadge status={r.status} />
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular text-fg-muted">{fmt(r.score)}</td>
                      {spec && <td className="px-3 py-2 text-right font-mono tabular text-fg">{fmt(r.metrics[spec.primary_metric])}</td>}
                      <td className="px-3 py-2 whitespace-nowrap">
                        {kp.length ? (
                          <span className={cn('font-mono', kp.every(Boolean) ? 'text-ok' : 'text-fg-muted')}>
                            {kp.filter(Boolean).length}/{kp.length}
                          </span>
                        ) : (
                          <span className="text-fg-subtle">—</span>
                        )}
                      </td>
                      <td className="max-w-[280px] truncate px-3 py-2 text-fg-muted" title={r.description}>
                        {r.description || '—'}
                      </td>
                      <td className="px-3 py-2 text-right whitespace-nowrap text-fg-subtle tabular">{formatDuration(r.train_seconds)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
        {selectedRow && (
          <Card className="p-4 xl:sticky xl:top-4 xl:self-start">
            <ExperimentDetail row={selectedRow} live={liveById[selectedRow.exp_id]} />
          </Card>
        )}
      </div>

      {spec && (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-fg-subtle">
          <span>KPI targets:</span>
          {spec.kpis.map((k) => (
            <Badge key={kpiLabel(k)} className="font-mono">
              {kpiLabel(k)}
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}

export function ExperimentsTab({ runId, model }: { runId: string; model: RunModel }) {
  const ids = model.taskOrder
  const [active, setActive] = useState<string>()
  const current = active && ids.includes(active) ? active : ids[0]
  if (!current)
    return (
      <Card>
        <EmptyState icon={<FlaskConical className="size-5" />} title="No tasks yet" description="Experiments are grouped by task." />
      </Card>
    )
  return (
    <div className="space-y-4">
      {ids.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {ids.map((id) => {
            const t = model.tasks[id]
            return (
              <button
                key={id}
                onClick={() => setActive(id)}
                className={cn(
                  'flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs transition-colors',
                  current === id ? 'border-accent/50 bg-accent-soft text-fg' : 'border-line text-fg-muted hover:text-fg',
                )}
              >
                <span className="font-medium">{t.spec?.name ?? id}</span>
                <span className="font-mono text-fg-subtle">{t.experiments.length}</span>
              </button>
            )
          })}
        </div>
      )}
      <TaskExperiments key={current} runId={runId} model={model} taskId={current} />
    </div>
  )
}
