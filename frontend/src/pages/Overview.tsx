import { Link } from 'react-router'
import { Activity, ArrowRight, Clock, Info, Plus, Rocket, Workflow } from 'lucide-react'
import { api } from '@/api/client'
import type { Deployment, RunSummary } from '@/api/types'
import { PageBody, PageHeader } from '@/components/Layout'
import { RunStatusBadge } from '@/components/status'
import { Badge, ButtonLink, Card, CardHeader, Mono, Skeleton } from '@/components/ui/primitives'
import { EmptyState, ErrorState } from '@/components/ui/feedback'
import { useApi } from '@/hooks/useApi'
import { TERMINAL } from '@/lib/runModel'
import { runIdDate, timeAgo } from '@/lib/utils'

function sortRuns(runs: RunSummary[]): RunSummary[] {
  return [...runs].sort((a, b) => b.run_id.localeCompare(a.run_id)) // ids start with a timestamp
}

function RunRow({ run }: { run: RunSummary }) {
  const started = runIdDate(run.run_id)
  return (
    <Link
      to={`/runs/${encodeURIComponent(run.run_id)}`}
      className="group flex items-center gap-4 px-4 py-3 transition-colors hover:bg-surface-2"
    >
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm text-fg">{run.goal}</div>
        <div className="mt-0.5 flex items-center gap-2 text-xs text-fg-subtle">
          <Mono>{run.run_id}</Mono>
          {started && (
            <>
              <span>·</span>
              <span>{timeAgo(started)}</span>
            </>
          )}
        </div>
        {run.error && <div className="mt-1 truncate font-mono text-[11px] text-bad">{run.error}</div>}
      </div>
      <RunStatusBadge status={run.status} />
      <ArrowRight className="size-4 shrink-0 text-fg-subtle opacity-0 transition-opacity group-hover:opacity-100" />
    </Link>
  )
}

function ListSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="divide-y divide-line">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="px-4 py-3.5">
          <Skeleton className="h-4 w-2/3" />
          <Skeleton className="mt-2 h-3 w-1/3" />
        </div>
      ))}
    </div>
  )
}

function DeploymentRow({ dep }: { dep: Deployment }) {
  return (
    <Link
      to={`/deployments/${encodeURIComponent(dep.id)}`}
      className="flex items-center gap-3 px-4 py-3 transition-colors hover:bg-surface-2"
    >
      <Rocket className="size-4 shrink-0 text-teal" />
      <div className="min-w-0 flex-1">
        <Mono className="text-fg">{dep.id}</Mono>
        <div className="mt-0.5 truncate text-xs text-fg-subtle">{dep.task_ids.join(', ') || 'no tasks'}</div>
      </div>
      <Badge tone={dep.status === 'live' ? 'teal' : 'neutral'} dot>
        {dep.status}
      </Badge>
    </Link>
  )
}

export function OverviewPage() {
  const runs = useApi(api.listRuns, ['runs'], { intervalMs: 4000 })
  const deployments = useApi(api.deployments, ['deployments'], { intervalMs: 8000 })

  const all = runs.data ? sortRuns(runs.data) : []
  const active = all.filter((r) => !TERMINAL.has(r.status))
  const counts = all.reduce<Record<string, number>>((acc, r) => ({ ...acc, [r.status]: (acc[r.status] ?? 0) + 1 }), {})

  return (
    <>
      <PageHeader
        title="Mission control"
        description="Give Assembly an industrial goal; it formulates ML tasks, prepares data, researches models and deploys the winners."
        actions={
          <ButtonLink to="/runs/new" variant="primary">
            <Plus className="size-4" /> New run
          </ButtonLink>
        }
      />
      <PageBody className="space-y-6">
        {/* Summary strip: counts straight from GET /api/runs and /api/deployments */}
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {[
            { label: 'Active runs', value: runs.data ? active.length : undefined, icon: Activity },
            { label: 'Awaiting approval', value: runs.data ? (counts.awaiting_approval ?? 0) : undefined, icon: Clock },
            { label: 'Completed', value: runs.data ? (counts.done ?? 0) : undefined, icon: Workflow },
            { label: 'Deployments', value: deployments.data?.length, icon: Rocket },
          ].map(({ label, value, icon: Icon }) => (
            <Card key={label} className="px-4 py-3.5">
              <div className="flex items-center justify-between text-xs text-fg-muted">
                {label}
                <Icon className="size-3.5 text-fg-subtle" />
              </div>
              <div className="mt-1.5 text-2xl font-semibold tabular">
                {value === undefined ? <Skeleton className="h-7 w-10" /> : value}
              </div>
            </Card>
          ))}
        </div>

        {active.some((r) => r.status === 'awaiting_approval') && (
          <Card className="border-warn/30 bg-warn-soft/40">
            <div className="flex flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-start gap-2.5 text-sm">
                <Clock className="mt-0.5 size-4 shrink-0 text-warn" />
                <div>
                  <div className="font-medium text-fg">A run is waiting for your approval</div>
                  <div className="text-xs text-fg-muted">The agent paused after designing its data plan.</div>
                </div>
              </div>
              <ButtonLink
                size="sm"
                to={`/runs/${encodeURIComponent(active.find((r) => r.status === 'awaiting_approval')!.run_id)}?tab=plan`}
              >
                Review plan
              </ButtonLink>
            </div>
          </Card>
        )}

        <div className="grid gap-6 xl:grid-cols-[1fr_380px]">
          <div className="space-y-6">
            <Card>
              <CardHeader
                title="Active runs"
                icon={<Activity className="size-4" />}
                actions={active.length > 0 && <Badge tone="accent">{active.length}</Badge>}
              />
              {runs.loading && !runs.data ? (
                <ListSkeleton rows={2} />
              ) : runs.error && !runs.data ? (
                <ErrorState error={runs.error} onRetry={runs.refresh} />
              ) : active.length === 0 ? (
                <EmptyState
                  icon={<Activity className="size-5" />}
                  title="Nothing running"
                  description="Start a run to watch Assembly formulate tasks and research models live."
                />
              ) : (
                <div className="divide-y divide-line">
                  {active.map((r) => (
                    <RunRow key={r.run_id} run={r} />
                  ))}
                </div>
              )}
            </Card>

            <Card>
              <CardHeader title="Run history" icon={<Workflow className="size-4" />} />
              {runs.loading && !runs.data ? (
                <ListSkeleton />
              ) : runs.error && !runs.data ? (
                <ErrorState error={runs.error} onRetry={runs.refresh} />
              ) : all.length === 0 ? (
                <EmptyState
                  icon={<Workflow className="size-5" />}
                  title="No runs yet"
                  description="Runs you start appear here."
                  action={
                    <ButtonLink to="/runs/new" size="sm" variant="primary">
                      <Plus className="size-3.5" /> New run
                    </ButtonLink>
                  }
                />
              ) : (
                <div className="divide-y divide-line">
                  {all.map((r) => (
                    <RunRow key={r.run_id} run={r} />
                  ))}
                </div>
              )}
            </Card>
          </div>

          <div className="space-y-6">
            <Card>
              <CardHeader
                title="Deployments"
                icon={<Rocket className="size-4" />}
                actions={
                  <Link to="/deployments" className="text-xs text-fg-muted hover:text-fg">
                    View all
                  </Link>
                }
              />
              {deployments.loading && !deployments.data ? (
                <ListSkeleton rows={2} />
              ) : deployments.error && !deployments.data ? (
                <ErrorState error={deployments.error} onRetry={deployments.refresh} />
              ) : !deployments.data?.length ? (
                <EmptyState
                  icon={<Rocket className="size-5" />}
                  title="No deployments"
                  description="A run registers a deployment when it reaches the deploy stage."
                />
              ) : (
                <div className="divide-y divide-line">
                  {deployments.data.map((d) => (
                    <DeploymentRow key={d.id} dep={d} />
                  ))}
                </div>
              )}
            </Card>

            <div className="flex gap-2.5 rounded-lg border border-line px-3.5 py-3 text-xs leading-relaxed text-fg-muted">
              <Info className="mt-0.5 size-3.5 shrink-0 text-fg-subtle" />
              <span>
                Runs and deployments live in the API process's memory. Restarting the backend clears this list (workspace
                files under <Mono>runs/</Mono> stay on disk).
              </span>
            </div>
          </div>
        </div>
      </PageBody>
    </>
  )
}
