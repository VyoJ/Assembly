import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { AlertTriangle, Bell, BellOff, FlaskRound, Info, Play, Radio, Rocket, Send, Square } from 'lucide-react'
import { api, ApiError, url } from '@/api/client'
import { connectSse, type SseStatus } from '@/api/sse'
import type { Deployment, LiveTick, Prediction } from '@/api/types'
import { PageBody, PageHeader } from '@/components/Layout'
import { StreamStatusBadge } from '@/components/status'
import { Badge, Button, ButtonLink, Card, CardHeader, Label, Mono, Skeleton, inputClass } from '@/components/ui/primitives'
import { Callout, EmptyState, ErrorState } from '@/components/ui/feedback'
import { CodeBlock, KeyValue } from '@/components/ui/content'
import { ChartTooltip } from '@/components/charts'
import { axisProps, CHART } from '@/components/chartTheme'
import { useApi } from '@/hooks/useApi'
import { useHealth } from '@/hooks/health'
import { cn, fmt } from '@/lib/utils'

// ---- prediction rendering

/** The headline number of a prediction, by format (ml/package.py: scalar→value, binary_score→score). */
function headline(p: Prediction): { key: string; value: number } | undefined {
  for (const key of ['score', 'value']) if (typeof p[key] === 'number') return { key, value: p[key] as number }
  return undefined
}

function PredictionView({ taskId, p }: { taskId: string; p: Prediction }) {
  const h = headline(p)
  const rest = Object.entries(p).filter(([k]) => !['alarm', 'latency_ms', h?.key].includes(k))
  return (
    <div className="rounded-lg border border-line bg-surface-2 p-3">
      <div className="flex items-center justify-between gap-2">
        <Mono className="text-fg">{taskId}</Mono>
        {typeof p.alarm === 'boolean' && (
          <Badge tone={p.alarm ? 'bad' : 'ok'} dot>
            {p.alarm ? 'ALARM' : 'no alarm'}
          </Badge>
        )}
      </div>
      <div className="mt-2 flex items-baseline gap-4">
        {h && (
          <div>
            <Label>{h.key}</Label>
            <div className="font-mono text-xl text-fg tabular">{fmt(h.value)}</div>
          </div>
        )}
        {typeof p.latency_ms === 'number' && (
          <div>
            <Label>latency</Label>
            <div className="font-mono text-sm text-fg-muted tabular">{fmt(p.latency_ms)} ms</div>
          </div>
        )}
      </div>
      {rest.length > 0 && <CodeBlock className="mt-2" maxHeight="max-h-40">{JSON.stringify(Object.fromEntries(rest), null, 2)}</CodeBlock>}
    </div>
  )
}

// ---- inference tester

function InferenceTester({ dep }: { dep: Deployment }) {
  const health = useHealth()
  const templates: { label: string; body: string; note: string }[] = [
    { label: 'Replay test sample', body: '{\n  "replay": 0\n}', note: 'Held-out test sample i: its inputs and the winner\'s scored prediction (real packages).' },
    { label: 'Feature values', body: '{\n  "feature_name": 0.0\n}', note: 'Tabular models: named features; missing features fall back to training medians.' },
    { label: 'Image path', body: '{\n  "image_path": "/path/on/server.png"\n}', note: 'File models: a path readable by the API server.' },
  ]
  if (health.data?.ml_mock)
    templates.unshift({ label: 'Mock risk', body: '{\n  "risk": 0.85\n}', note: 'ML_MOCK=1: the mock predictor derives its output from "risk" (0–1).' })
  const [body, setBody] = useState(templates[0].body)
  const [note, setNote] = useState(templates[0].note)
  const [taskId, setTaskId] = useState('')
  const [result, setResult] = useState<Record<string, Prediction> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  let parsed: Record<string, unknown> | null = null
  let parseError: string | null = null
  try {
    const v = JSON.parse(body)
    if (v && typeof v === 'object' && !Array.isArray(v)) parsed = v
    else parseError = 'The sample must be a JSON object.'
  } catch (e) {
    parseError = (e as Error).message
  }

  const send = async () => {
    if (!parsed) return
    setLoading(true)
    setError(null)
    try {
      setResult(await api.predict(dep.id, parsed, taskId || undefined))
    } catch (e) {
      setResult(null)
      setError(e instanceof ApiError ? (e.status === 500 ? 'The predictor raised an error for this sample (HTTP 500). Check the input shape for this model.' : e.message) : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader title="Inference tester" icon={<Send className="size-4" />} description={<>POST <Mono>{dep.endpoint}/predict</Mono></>} />
      <div className="grid gap-4 p-4 lg:grid-cols-2">
        <div className="space-y-3">
          <div className="flex flex-wrap gap-1.5">
            {templates.map((t) => (
              <button
                key={t.label}
                onClick={() => {
                  setBody(t.body)
                  setNote(t.note)
                }}
                className="rounded-md border border-line px-2 py-1 text-[11px] text-fg-muted transition-colors hover:border-line-strong hover:text-fg"
              >
                {t.label}
              </button>
            ))}
          </div>
          <div className="text-[11px] text-fg-subtle">{note}</div>
          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            rows={8}
            spellCheck={false}
            className={cn(inputClass, 'font-mono text-xs leading-relaxed', parseError && 'border-bad/50')}
            aria-label="Sample JSON"
          />
          {parseError && <div className="text-[11px] text-bad">{parseError}</div>}
          <div className="flex gap-2">
            <select value={taskId} onChange={(e) => setTaskId(e.target.value)} className={cn(inputClass, 'flex-1')} aria-label="Task">
              <option value="">All tasks in this deployment</option>
              {dep.task_ids.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <Button variant="primary" onClick={send} loading={loading} disabled={!parsed}>
              <Send className="size-3.5" /> Predict
            </Button>
          </div>
        </div>
        <div className="space-y-2">
          {error && <Callout tone="bad" icon={<AlertTriangle className="size-3.5" />}>{error}</Callout>}
          {result ? (
            Object.keys(result).length ? (
              Object.entries(result).map(([tid, p]) => <PredictionView key={tid} taskId={tid} p={p} />)
            ) : (
              <EmptyState title="Empty response" description="No predictor matched the selected task." />
            )
          ) : (
            !error && <EmptyState className="h-full rounded-lg border border-dashed border-line" title="No prediction yet" description="Send a sample to see the model output and alarm state." />
          )}
        </div>
      </div>
    </Card>
  )
}

// ---- live simulator

const MAX_TICKS = 120

function LiveSimulator({ dep }: { dep: Deployment }) {
  const [running, setRunning] = useState(false)
  const [hz, setHz] = useState(1)
  const [status, setStatus] = useState<SseStatus>('closed')
  const [ticks, setTicks] = useState<LiveTick[]>([])

  useEffect(() => {
    if (!running) return
    setTicks([])
    const stop = connectSse(url(api.livePath(dep.id, hz)), {
      onMessage: ({ event, data }) => {
        if (event !== 'prediction') return
        try {
          const tick = JSON.parse(data) as LiveTick
          setTicks((t) => [...t.slice(-(MAX_TICKS - 1)), tick])
        } catch {
          /* ignore malformed */
        }
      },
      onStatus: (s) => setStatus(s),
    })
    return () => stop()
  }, [running, hz, dep.id])

  const last = ticks.at(-1)
  return (
    <Card>
      <CardHeader
        title="Live simulator"
        icon={<Radio className="size-4" />}
        description={<>SSE <Mono>{dep.endpoint}/live</Mono></>}
        actions={
          <>
            {running && <StreamStatusBadge status={status} />}
            <select value={hz} onChange={(e) => setHz(Number(e.target.value))} className="h-7 rounded-md border border-line-strong bg-surface-2 px-2 text-[11px] text-fg" aria-label="Rate">
              {[0.5, 1, 2, 5].map((v) => (
                <option key={v} value={v}>
                  {v} Hz
                </option>
              ))}
            </select>
            <Button size="sm" variant={running ? 'danger' : 'primary'} onClick={() => setRunning((r) => !r)}>
              {running ? <Square className="size-3" /> : <Play className="size-3" />}
              {running ? 'Stop' : 'Start'}
            </Button>
          </>
        }
      />
      <div className="space-y-4 p-4">
        <Callout tone="warn" icon={<FlaskRound className="size-3.5" />} title="Simulated plant data">
          Samples come from <Mono>plant/simulator.py</Mono>: a synthetic press that degrades, fails and is repaired, with generated sensor values.
          This is not real industrial monitoring, and the alarm rate is not a validated KPI.
        </Callout>
        {!running && !ticks.length ? (
          <EmptyState title="Simulator stopped" description="Start it to stream simulated samples through every model in this deployment." />
        ) : (
          <>
            {last && (
              <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs text-fg-muted">
                <span>
                  simulated cycle <Mono className="text-fg">{last.cycle}</Mono>
                </span>
                <span>
                  ticks <Mono className="text-fg">{Object.values(last.kpi_tracker)[0]?.ticks ?? ticks.length}</Mono>
                </span>
              </div>
            )}
            <div className="grid gap-4 lg:grid-cols-2">
              {dep.task_ids.map((tid) => {
                const series = ticks.map((t) => {
                  const p = t.predictions[tid]
                  const h = p ? headline(p) : undefined
                  return { cycle: t.cycle, t: t.t, value: h?.value, key: h?.key, alarm: !!p?.alarm }
                })
                const key = series.find((s) => s.key)?.key
                const tracker = last?.kpi_tracker[tid]
                const lastP = last?.predictions[tid]
                return (
                  <div key={tid} className="rounded-lg border border-line bg-surface-2 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <Mono className="truncate text-fg">{tid}</Mono>
                      {lastP && typeof lastP.alarm === 'boolean' && (
                        <Badge tone={lastP.alarm ? 'bad' : 'ok'} dot>
                          {lastP.alarm ? <Bell className="size-3" /> : <BellOff className="size-3" />}
                          {lastP.alarm ? 'ALARM' : 'normal'}
                        </Badge>
                      )}
                    </div>
                    {tracker && (
                      <div className="mt-1 text-[11px] text-fg-subtle">
                        simulated alarm rate <span className="font-mono text-fg-muted">{(tracker.alarm_rate * 100).toFixed(1)}%</span> ({tracker.alarms}/{tracker.ticks})
                      </div>
                    )}
                    {key ? (
                      <div className="mt-2 h-36">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={series} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                            <CartesianGrid stroke={CHART.grid} vertical={false} />
                            <XAxis dataKey="cycle" {...axisProps} minTickGap={24} />
                            <YAxis {...axisProps} width={44} tickFormatter={(v: number) => fmt(v, 3)} domain={['auto', 'auto']} />
                            <Tooltip
                              content={({ active, payload }) => {
                                const p = active && payload?.[0]?.payload
                                if (!p) return null
                                return <ChartTooltip title={`cycle ${p.cycle}`} rows={[{ label: key, value: p.value ?? '—' }, { label: 'alarm', value: p.alarm ? 'yes' : 'no' }]} />
                              }}
                            />
                            <Line
                              dataKey="value"
                              stroke={CHART.teal}
                              strokeWidth={2}
                              isAnimationActive={false}
                              dot={(d: { cx?: number; cy?: number; index?: number; payload?: { alarm: boolean } }) =>
                                d.payload?.alarm && d.cx != null && d.cy != null ? (
                                  <circle key={d.index} cx={d.cx} cy={d.cy} r={3} fill={CHART.bad} />
                                ) : (
                                  <g key={d.index} />
                                )
                              }
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      lastP && <CodeBlock className="mt-2" maxHeight="max-h-36">{JSON.stringify(lastP, null, 2)}</CodeBlock>
                    )}
                  </div>
                )
              })}
            </div>
          </>
        )}
      </div>
    </Card>
  )
}

// ---- page

function DeploymentDetail({ dep }: { dep: Deployment }) {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title={<Mono className="text-sm">{dep.id}</Mono>}
          icon={<Rocket className="size-4 text-teal" />}
          actions={
            <Badge tone={dep.status === 'live' ? 'teal' : 'neutral'} dot>
              {dep.status}
            </Badge>
          }
        />
        <div className="p-4">
          <KeyValue
            items={[
              ['Run', <Link key="r" to={`/runs/${encodeURIComponent(dep.run_id)}`} className="font-mono text-accent hover:underline">{dep.run_id}</Link>],
              ['Endpoint', <Mono key="e">{dep.endpoint}</Mono>],
              ['Tasks', dep.task_ids.map((t, i) => <Mono key={t} className="mr-3">{t} <span className="text-fg-subtle">({dep.exp_ids[i] ?? '—'})</span></Mono>)],
              ['Packages', <div key="p" className="space-y-0.5">{Object.entries(dep.package_paths).map(([t, p]) => <div key={t} className="font-mono text-[11px] break-all text-fg-muted">{p}</div>)}</div>],
            ]}
          />
          <div className="mt-3 flex gap-2 text-[11px] text-fg-subtle">
            <Info className="mt-px size-3 shrink-0" />
            Registered automatically at the end of the run. Deployment gating on the final test result is not enforced by the backend yet.
          </div>
        </div>
      </Card>
      <InferenceTester dep={dep} />
      <LiveSimulator dep={dep} />
    </div>
  )
}

export function DeploymentsPage() {
  const { depId } = useParams()
  const deps = useApi(api.deployments, ['deployments'], { intervalMs: 10_000 })
  const list = deps.data ?? []
  const selected = list.find((d) => d.id === depId) ?? (!depId ? list[0] : undefined)

  return (
    <>
      <PageHeader title="Deployments" description="Models packaged and registered by completed runs. Held in API memory; cleared on restart." />
      <PageBody>
        {deps.loading && !deps.data ? (
          <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
            <Skeleton className="h-40" />
            <Skeleton className="h-96" />
          </div>
        ) : deps.error && !deps.data ? (
          <Card>
            <ErrorState error={deps.error} onRetry={deps.refresh} />
          </Card>
        ) : list.length === 0 ? (
          <Card>
            <EmptyState
              icon={<Rocket className="size-5" />}
              title="No deployments yet"
              description="A run registers a deployment after research, full training and the final gate."
              action={<ButtonLink to="/runs/new" size="sm" variant="primary">Start a run</ButtonLink>}
            />
          </Card>
        ) : (
          <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)]">
            <Card className="self-start overflow-hidden">
              <div className="divide-y divide-line">
                {list.map((d) => (
                  <Link
                    key={d.id}
                    to={`/deployments/${encodeURIComponent(d.id)}`}
                    className={cn('block px-4 py-3 transition-colors hover:bg-surface-2', selected?.id === d.id && 'bg-surface-2')}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <Mono className="text-fg">{d.id}</Mono>
                      <span className={cn('size-1.5 rounded-full', d.status === 'live' ? 'bg-teal' : 'bg-fg-subtle')} />
                    </div>
                    <div className="mt-0.5 truncate text-xs text-fg-subtle">{d.task_ids.join(', ')}</div>
                  </Link>
                ))}
              </div>
            </Card>
            {selected ? (
              <DeploymentDetail key={selected.id} dep={selected} />
            ) : (
              <Card>
                <EmptyState title="Deployment not found" description={`No deployment "${depId}" on the backend.`} />
              </Card>
            )}
          </div>
        )}
      </PageBody>
    </>
  )
}
