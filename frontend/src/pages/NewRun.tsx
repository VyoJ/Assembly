import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router'
import { ArrowRight, Info, ShieldCheck, Sparkles, Zap } from 'lucide-react'
import { api, ApiError } from '@/api/client'
import { STAGES } from '@/api/events'
import { PageBody, PageHeader } from '@/components/Layout'
import { Button, Card, Kbd } from '@/components/ui/primitives'
import { Callout } from '@/components/ui/feedback'
import { useHealth } from '@/hooks/health'
import { useApi } from '@/hooks/useApi'
import { cn } from '@/lib/utils'

const EXAMPLES = [
  'Predict press failures on Line 3 with at least 90% recall and fewer than 5% false alarms.',
  'Prevent unplanned stoppages on Press Line 3.',
  'Detect defective metal nuts at the camera station before they reach assembly.',
  'Classify bearing faults on Motor 4 from vibration data.',
]

export function NewRunPage() {
  const navigate = useNavigate()
  const health = useHealth()
  const [goal, setGoal] = useState('')
  const [autoApprove, setAutoApprove] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)
  // The backend runs one pipeline at a time (409 otherwise): show the active one instead of a dead-end error.
  const runs = useApi(api.listRuns, ['runs'], { intervalMs: 5000 })
  const active = runs.data?.find((r) => r.source !== 'recorded' && (r.status === 'running' || r.status === 'awaiting_approval'))

  const submit = async (e?: FormEvent) => {
    e?.preventDefault()
    if (!goal.trim() || submitting) return
    setSubmitting(true)
    setError(null)
    try {
      const run = await api.startRun({ goal: goal.trim(), auto_approve: autoApprove })
      navigate(`/runs/${encodeURIComponent(run.run_id)}`)
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)))
      setSubmitting(false)
    }
  }

  return (
    <>
      <PageHeader
        eyebrow="Runs"
        title="New run"
        description="Describe the outcome you need in plain language. Assembly decides which ML tasks to build, which plant data to use, and how to measure success."
      />
      <PageBody>
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
          <form onSubmit={submit} className="space-y-5">
            <Card className="p-1">
              <label htmlFor="goal" className="sr-only">
                Goal
              </label>
              <textarea
                id="goal"
                autoFocus
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void submit()
                }}
                rows={6}
                placeholder="e.g. Predict motor failures with at least 90% recall and fewer than 5% false alarms."
                className="w-full resize-y rounded-lg bg-transparent px-4 py-3.5 text-[15px] leading-relaxed text-fg placeholder:text-fg-subtle focus:outline-none"
              />
              <div className="flex items-center justify-between gap-3 border-t border-line px-3 py-2 text-xs text-fg-subtle">
                <span>Include KPIs in the sentence (recall, false-alarm rate, error…); the agent turns them into measurable targets.</span>
                <span className="hidden shrink-0 items-center gap-1 sm:flex">
                  <Kbd>⌘</Kbd>
                  <Kbd>Enter</Kbd>
                </span>
              </div>
            </Card>

            <div>
              <div className="mb-2 text-xs font-medium text-fg-muted">Examples</div>
              <div className="flex flex-wrap gap-2">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex}
                    type="button"
                    onClick={() => setGoal(ex)}
                    className="rounded-full border border-line bg-surface px-3 py-1.5 text-left text-xs text-fg-muted transition-colors hover:border-line-strong hover:text-fg"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>

            <fieldset>
              <legend className="mb-2 text-xs font-medium text-fg-muted">Plan approval</legend>
              <div className="grid gap-3 sm:grid-cols-2">
                {[
                  {
                    value: false,
                    icon: ShieldCheck,
                    title: 'Manual approval',
                    body: 'The agent pauses after designing the data plan. You review task specs, KPIs and budgets before any training.',
                    recommended: true,
                  },
                  {
                    value: true,
                    icon: Zap,
                    title: 'Auto-approve',
                    body: 'The plan is approved automatically and research starts immediately.',
                  },
                ].map(({ value, icon: Icon, title, body, recommended }) => (
                  <label
                    key={title}
                    className={cn(
                      'flex cursor-pointer gap-3 rounded-lg border p-3.5 transition-colors',
                      autoApprove === value ? 'border-accent/50 bg-accent-soft' : 'border-line bg-surface hover:border-line-strong',
                    )}
                  >
                    <input
                      type="radio"
                      name="approval"
                      className="sr-only"
                      checked={autoApprove === value}
                      onChange={() => setAutoApprove(value)}
                    />
                    <Icon className={cn('mt-0.5 size-4 shrink-0', autoApprove === value ? 'text-accent' : 'text-fg-subtle')} />
                    <div>
                      <div className="flex items-center gap-2 text-sm font-medium text-fg">
                        {title}
                        {recommended && <span className="text-[10px] font-normal text-fg-subtle">Recommended</span>}
                      </div>
                      <div className="mt-0.5 text-xs leading-relaxed text-fg-muted">{body}</div>
                    </div>
                  </label>
                ))}
              </div>
            </fieldset>

            {active && (
              <Callout tone="warn" title="A run is already in progress">
                Assembly runs one pipeline at a time. <Link to={`/runs/${encodeURIComponent(active.run_id)}`} className="underline underline-offset-2">Open {active.run_id}</Link>{' '}
                ({active.status === 'awaiting_approval' ? 'waiting for your approval' : 'running'}) and start a new one once it finishes.
              </Callout>
            )}
            {error && !(active && error.status === 409) && (
              <Callout tone="bad" title={error.unreachable ? 'Backend unavailable' : 'Could not start the run'}>
                {error.unreachable ? 'The Assembly API is not reachable right now (it may be starting up); try again in a moment.' : error.detail}
              </Callout>
            )}

            <div className="flex items-center gap-3">
              <Button type="submit" variant="primary" size="lg" loading={submitting} disabled={!goal.trim() || !!active}>
                <Sparkles className="size-4" /> Start run <ArrowRight className="size-4" />
              </Button>
              <span className="text-xs text-fg-subtle">
                Sends <code className="font-mono">POST /api/runs</code>
              </span>
            </div>
          </form>

          <aside className="space-y-4">
            <Card className="p-4">
              <div className="mb-3 text-xs font-medium text-fg-muted">What happens next</div>
              <ol className="space-y-2.5">
                {STAGES.map((s, i) => (
                  <li key={s.id} className="flex gap-3 text-xs">
                    <span className="flex size-5 shrink-0 items-center justify-center rounded-full border border-line-strong font-mono text-[10px] text-fg-subtle">
                      {i + 1}
                    </span>
                    <div>
                      <div className="text-fg">{s.label}</div>
                      <div className="text-fg-subtle">{s.hint}</div>
                    </div>
                  </li>
                ))}
              </ol>
            </Card>
            {health.data && (health.data.ml_mock || health.data.llm_mock) && (
              <Callout tone="warn" icon={<Info className="size-3.5" />} title="Backend is in mock mode">
                {health.data.ml_mock && 'ML results are simulated (ML_MOCK=1). '}
                {health.data.llm_mock && 'Agent replies are canned (LLM_MOCK=1). '}
                Metrics shown for this run will not reflect real models.
              </Callout>
            )}
          </aside>
        </div>
      </PageBody>
    </>
  )
}
