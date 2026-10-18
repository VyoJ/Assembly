import { CheckCircle2, FileText, Rocket, XCircle } from 'lucide-react'
import type { Deployment, FinalReport, RunDetail } from '@/api/types'
import type { RunModel } from '@/lib/runModel'
import { fmt } from '@/lib/utils'
import { Badge, ButtonLink, Card, CardHeader, Label, Mono } from '@/components/ui/primitives'
import { Callout, EmptyState } from '@/components/ui/feedback'
import { KeyValue, LongText, Markdown } from '@/components/ui/content'

function MetricsGrid({ metrics }: { metrics: Record<string, number> }) {
  return (
    <div className="grid max-w-3xl grid-cols-2 gap-x-8 gap-y-1 sm:grid-cols-3">
      {Object.entries(metrics).map(([k, v]) => (
        <div key={k} className="flex justify-between gap-2 text-xs">
          <span className="truncate font-mono text-fg-subtle">{k}</span>
          <span className="font-mono text-fg tabular">{fmt(v)}</span>
        </div>
      ))}
    </div>
  )
}

export function ResultsTab({ model, detail }: { model: RunModel; detail?: RunDetail }) {
  // Events are primary; GET /api/runs/{id} state fills gaps (e.g. events missed before a reconnect).
  const state = detail?.state
  const summary = model.report ?? state?.summary
  const deployment: Deployment | undefined = model.deployment ?? state?.deployment
  const ids = [...new Set([...model.taskOrder, ...Object.keys(state?.reports ?? {})])]
  const reports: Record<string, FinalReport | undefined> = Object.fromEntries(
    ids.map((id) => [id, model.tasks[id]?.finalTest ?? state?.reports?.[id]]),
  )
  const anything = summary || deployment || ids.some((id) => reports[id] || model.tasks[id]?.fullTrain || model.tasks[id]?.researchDone)

  if (!anything)
    return (
      <Card>
        <EmptyState
          icon={<FileText className="size-5" />}
          title="No results yet"
          description="Final evaluation on the hidden test split, the agent's report and the deployment appear here when the run reaches those stages."
        />
      </Card>
    )

  return (
    <div className="space-y-4">
      {deployment && (
        <Card className="border-teal/30">
          <div className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <Rocket className="mt-0.5 size-5 text-teal" />
              <div>
                <div className="text-sm font-medium text-fg">
                  Deployment <Mono>{deployment.id}</Mono> registered
                </div>
                <div className="mt-0.5 text-xs text-fg-muted">
                  {deployment.task_ids.length} model(s): {deployment.task_ids.join(', ') || '—'} · endpoint <Mono>{deployment.endpoint}</Mono>
                </div>
              </div>
            </div>
            <ButtonLink to={`/deployments/${encodeURIComponent(deployment.id)}`} size="sm" variant="primary">
              Open deployment
            </ButtonLink>
          </div>
        </Card>
      )}

      {summary && (
        <Card>
          <CardHeader title="Agent report" icon={<FileText className="size-4" />} description="Written by the agent after the final gate" />
          <div className="p-4">
            <Markdown text={summary} />
          </div>
        </Card>
      )}

      {ids.map((id) => {
        const t = model.tasks[id]
        const report = reports[id]
        const full = t?.fullTrain ?? state?.full?.[id]
        if (!report && !full && !t?.researchDone && !t?.dropped && !t?.fullTrainError) return null
        return (
          <Card key={id}>
            <CardHeader
              title={t?.spec?.name ?? id}
              description={<Mono>{id}</Mono>}
              actions={
                t?.dropped ? (
                  <Badge tone="bad">dropped</Badge>
                ) : report ? (
                  <Badge tone={report.passed ? 'ok' : 'bad'} dot>
                    {report.passed ? 'Passed final gate' : 'Failed final gate'}
                  </Badge>
                ) : null
              }
            />
            <div className="space-y-4 p-4">
              {t?.dropped && (
                <Callout tone="bad">
                  <LongText text={t.dropped} />
                </Callout>
              )}
              {report && (
                <div>
                  <div className="mb-2 flex items-center gap-2">
                    {report.passed ? <CheckCircle2 className="size-4 text-ok" /> : <XCircle className="size-4 text-bad" />}
                    <Label>Hidden test split · {report.exp_id}</Label>
                  </div>
                  <MetricsGrid metrics={report.test_metrics} />
                  {report.failed_kpis.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {report.failed_kpis.map((k) => (
                        <Badge key={k} tone="bad" className="font-mono">
                          ✗ {k}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              )}
              {full && (
                <div className="border-t border-line pt-4">
                  <Label className="mb-2">Full train (validation) · {full.description}</Label>
                  <MetricsGrid metrics={full.metrics} />
                </div>
              )}
              {t?.fullTrainError && <Callout tone="bad" title="Full train failed">{t.fullTrainError}</Callout>}
              {t?.researchDone && (
                <KeyValue
                  className="border-t border-line pt-4"
                  items={[
                    ['Research stop reason', t.researchDone.reason],
                    ['Experiments run', t.researchDone.n_exp],
                    ['Winning experiment', t.researchDone.best?.exp_id ?? '—'],
                  ]}
                />
              )}
            </div>
          </Card>
        )
      })}
    </div>
  )
}
