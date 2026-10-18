import { useEffect, useMemo, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router'
import { AlertTriangle, ChevronLeft, Clock, RotateCw } from 'lucide-react'
import { api } from '@/api/client'
import { useApi } from '@/hooks/useApi'
import { useRunEvents } from '@/hooks/useRunEvents'
import { buildModel, stageStates, TERMINAL } from '@/lib/runModel'
import { formatDuration, runIdDate } from '@/lib/utils'
import { PageBody } from '@/components/Layout'
import { RunStatusBadge, StreamStatusBadge } from '@/components/status'
import { Badge, Button, Card, Mono, Skeleton } from '@/components/ui/primitives'
import { Callout, ErrorState } from '@/components/ui/feedback'
import { Tabs } from '@/components/ui/content'
import { Stepper } from '@/features/run/Stepper'
import { ActivityTab } from '@/features/run/ActivityTab'
import { PlanTab } from '@/features/run/PlanTab'
import { ExperimentsTab } from '@/features/run/ExperimentsTab'
import { ResultsTab } from '@/features/run/ResultsTab'
import { EventLogTab } from '@/features/run/EventLogTab'

type TabId = 'activity' | 'plan' | 'experiments' | 'results' | 'events'
const TAB_IDS: TabId[] = ['activity', 'plan', 'experiments', 'results', 'events']

/** Re-render every second while something is live (stage timers). */
function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [active])
  return now
}

// Events after which the run summary (status, error) is worth refetching immediately.
const STATUS_EVENTS = new Set(['stage', 'approval_required', 'approved', 'rejected', 'run_finished', 'run_failed', 'deployed'])

export function RunPage() {
  const { runId = '' } = useParams()
  const [params, setParams] = useSearchParams()
  const [polling, setPolling] = useState(true)
  const detail = useApi(() => api.getRun(runId), [runId], { intervalMs: 3000, poll: polling })
  const stream = useRunEvents(runId)
  const model = useMemo(() => buildModel(stream.events), [stream.events])

  // Status: the backend's answer, unless the stream already shows the run ended.
  const status = model.failedError ? 'failed' : detail.data?.status ?? (model.finishedStatus || undefined)
  const terminal = !!status && TERMINAL.has(status)
  const live = !terminal && !!detail.data
  const now = useNow(live)
  useEffect(() => setPolling(!terminal), [terminal])

  const { refresh } = detail
  const lastType = stream.events.at(-1)?.type
  useEffect(() => {
    if (lastType && STATUS_EVENTS.has(lastType)) void refresh()
  }, [lastType, stream.version, refresh])

  const states = stageStates(model, status)
  const requested = params.get('tab') as TabId | null
  const tab: TabId = requested && TAB_IDS.includes(requested) ? requested : status === 'awaiting_approval' ? 'plan' : 'activity'
  const setTab = (t: TabId) => setParams((p) => ({ ...Object.fromEntries(p), tab: t }), { replace: true })

  const experimentCount = model.taskOrder.reduce((n, id) => n + model.tasks[id].experiments.length, 0)
  const startedAt = model.startedAt ?? (runIdDate(runId)?.getTime() ?? 0) / 1000
  const elapsed = startedAt ? (model.endedAt ?? now / 1000) - startedAt : undefined

  if (detail.error?.status === 404)
    return (
      <ErrorState
        className="py-32"
        title="Run not found"
        error={`The backend has no run "${runId}". Runs are held in memory, so they disappear when the API restarts.`}
      />
    )
  if (detail.error && !detail.data && !stream.events.length) return <ErrorState className="py-32" error={detail.error} onRetry={detail.refresh} />

  const goal = detail.data?.goal ?? model.goal
  const error = detail.data?.error ?? model.failedError

  return (
    <>
      <div className="border-b border-line px-4 pt-5 lg:px-8 lg:pt-6">
        <Link to="/" className="mb-3 inline-flex items-center gap-1 text-xs text-fg-subtle hover:text-fg">
          <ChevronLeft className="size-3.5" /> Overview
        </Link>
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            {goal ? <h1 className="text-lg leading-snug font-semibold tracking-tight text-fg lg:text-xl">{goal}</h1> : <Skeleton className="h-7 w-96 max-w-full" />}
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-fg-subtle">
              <Mono>{runId}</Mono>
              <RunStatusBadge status={status} />
              <StreamStatusBadge status={stream.status} title={stream.detail} />
              {model.autoApprove !== undefined && <Badge>{model.autoApprove ? 'auto-approve' : 'manual approval'}</Badge>}
              {elapsed !== undefined && elapsed > 0 && (
                <span className="inline-flex items-center gap-1">
                  <Clock className="size-3" /> {formatDuration(elapsed)}
                </span>
              )}
            </div>
          </div>
          <Button size="sm" variant="ghost" onClick={() => void detail.refresh()} className="self-start">
            <RotateCw className="size-3.5" /> Refresh
          </Button>
        </div>

        <div className="mt-6 mb-5">
          <Stepper model={model} states={states} now={now} />
        </div>

        <Tabs<TabId>
          value={tab}
          onChange={setTab}
          className="border-b-0"
          tabs={[
            { id: 'activity', label: 'Activity' },
            {
              id: 'plan',
              label: 'Plan & approval',
              badge: status === 'awaiting_approval' ? <span className="size-1.5 animate-pulse-dot rounded-full bg-warn" /> : undefined,
            },
            { id: 'experiments', label: 'Experiments', badge: experimentCount ? <span className="font-mono text-[11px] text-fg-subtle">{experimentCount}</span> : undefined },
            { id: 'results', label: 'Results' },
            { id: 'events', label: 'Event log', badge: <span className="font-mono text-[11px] text-fg-subtle">{stream.events.length}</span> },
          ]}
        />
      </div>

      <PageBody className="space-y-4">
        {error && (
          <Callout tone="bad" icon={<AlertTriangle className="size-4" />} title="Run failed">
            <span className="font-mono break-all">{error}</span>
          </Callout>
        )}
        {status === 'awaiting_approval' && tab !== 'plan' && (
          <Card className="border-warn/30 bg-warn-soft/40">
            <div className="flex flex-col gap-2 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-2 text-sm text-fg">
                <Clock className="size-4 text-warn" /> The agent is waiting for you to approve its plan.
              </div>
              <Button size="sm" onClick={() => setTab('plan')}>
                Review plan
              </Button>
            </div>
          </Card>
        )}
        {stream.status === 'error' && (
          <Callout tone="bad" icon={<AlertTriangle className="size-4" />} title="Event stream unavailable">
            {stream.detail}
          </Callout>
        )}
        {stream.status === 'reconnecting' && (
          <Callout tone="warn" title="Reconnecting to the event stream">
            {stream.detail ?? 'Connection lost.'} Already-received events are kept; the backend replays history on reconnect.
          </Callout>
        )}

        {tab === 'activity' && <ActivityTab model={model} events={stream.events} live={live} />}
        {tab === 'plan' && <PlanTab runId={runId} model={model} status={status} onDecision={() => {
              setTab('plan') // keep the decision outcome in view once the status leaves awaiting_approval
              void detail.refresh()
            }}
          />}
        {tab === 'experiments' && <ExperimentsTab runId={runId} model={model} />}
        {tab === 'results' && <ResultsTab model={model} detail={detail.data} />}
        {tab === 'events' && <EventLogTab events={stream.events} runId={runId} />}
      </PageBody>
    </>
  )
}
