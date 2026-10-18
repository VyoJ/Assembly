// Folds the run's SSE event log into a view model. Pure: same events in, same model out.
// Only facts carried by events are recorded; nothing is interpolated or estimated.
import type { AnyEvent, RunEvent, Stage } from '@/api/events'
import { STAGES } from '@/api/events'
import type { Deployment, ExperimentResult, FinalReport, StreamProfile, TaskSpec } from '@/api/types'

export interface Validation {
  attempt: number
  ok: boolean
  checks: Record<string, boolean>
  messages: string[]
  ts: number
}

export interface TaskView {
  id: string
  spec?: TaskSpec
  discovery?: { scanned: number; kept: number; profiles: StreamProfile[] }
  designAttempts: number
  validations: Validation[]
  toolCalls: number
  dropped?: string
  notes?: string
  plan?: { text: string; revision: boolean; ts: number }
  planRevisions: number
  proposals: number
  invalidProposals: number
  lastInvalid?: string
  repairs: number
  fixes: number
  /** From `experiment` events: includes diff and log_tail, which the experiments endpoint omits. */
  experiments: ExperimentResult[]
  lastProposal?: { exp_id: string; hypothesis: string; description: string; ts: number }
  lastProgress?: Record<string, unknown> & { ts: number }
  researchDone?: { reason: string; best: ExperimentResult | null; n_exp: number }
  fullTrain?: ExperimentResult
  fullTrainError?: string
  finalTest?: FinalReport
}

export interface StageTiming {
  startedAt?: number
  endedAt?: number
}

export interface RunModel {
  goal?: string
  autoApprove?: boolean
  startedAt?: number
  endedAt?: number
  stages: Partial<Record<Stage, StageTiming>>
  currentStage?: Stage
  taskOrder: string[]
  tasks: Record<string, TaskView>
  warnings: string[]
  approvalRequest?: { tasks: Record<string, { spec: TaskSpec; data_card: string }>; ts: number }
  approvalOutcome?: { approved: true; tasks: TaskSpec[]; ts: number } | { approved: false; reason: string; ts: number }
  report?: string
  deployment?: Deployment
  finishedStatus?: string | null
  failedError?: string
  lastEventAt?: number
}

function task(m: RunModel, id: string): TaskView {
  if (!m.tasks[id]) {
    m.tasks[id] = {
      id, designAttempts: 0, validations: [], toolCalls: 0, planRevisions: 0, proposals: 0,
      invalidProposals: 0, repairs: 0, fixes: 0, experiments: [],
    }
    m.taskOrder.push(id)
  }
  return m.tasks[id]
}

export function emptyModel(): RunModel {
  return { stages: {}, taskOrder: [], tasks: {}, warnings: [] }
}

export function applyEvent(m: RunModel, raw: AnyEvent): void {
  const e = raw as RunEvent
  m.lastEventAt = e.ts
  switch (e.type) {
    case 'run_started':
      m.goal = e.goal
      m.autoApprove = e.auto_approve
      m.startedAt = e.ts
      break
    case 'stage': {
      if (m.currentStage) m.stages[m.currentStage] = { ...m.stages[m.currentStage], endedAt: e.ts }
      m.stages[e.stage] = { startedAt: e.ts }
      m.currentStage = e.stage
      break
    }
    case 'task_specs':
      for (const t of e.tasks) task(m, t.id).spec = t
      m.warnings = e.warnings ?? []
      break
    case 'discovery':
      task(m, e.task_id).discovery = { scanned: e.scanned, kept: e.kept, profiles: e.profiles }
      break
    case 'design_attempt':
      task(m, e.task_id).designAttempts = Math.max(task(m, e.task_id).designAttempts, e.attempt)
      break
    case 'tool_call':
      task(m, e.task_id).toolCalls += 1
      break
    case 'validation':
      task(m, e.task_id).validations.push({ attempt: e.attempt, ok: e.ok, checks: e.checks, messages: e.messages, ts: e.ts })
      break
    case 'task_dropped':
      task(m, e.task_id).dropped = e.reason
      break
    case 'approval_required':
      m.approvalRequest = { tasks: e.tasks, ts: e.ts }
      for (const [id, t] of Object.entries(e.tasks)) task(m, id).spec = t.spec
      break
    case 'approved':
      m.approvalOutcome = { approved: true, tasks: e.tasks, ts: e.ts }
      for (const t of e.tasks) task(m, t.id).spec = t
      break
    case 'rejected':
      m.approvalOutcome = { approved: false, reason: e.reason, ts: e.ts }
      break
    case 'exploration':
      task(m, e.task_id).notes = e.notes
      break
    case 'plan': {
      const t = task(m, e.task_id)
      t.plan = { text: e.plan, revision: e.revision, ts: e.ts }
      if (e.revision) t.planRevisions += 1
      break
    }
    case 'proposal': {
      const t = task(m, e.task_id)
      t.proposals += 1
      t.lastProposal = { exp_id: e.exp_id, hypothesis: e.hypothesis, description: e.description, ts: e.ts }
      break
    }
    case 'proposal_invalid': {
      const t = task(m, e.task_id)
      t.invalidProposals += 1
      t.lastInvalid = e.error
      break
    }
    case 'repair':
      task(m, e.task_id).repairs += 1
      break
    case 'fix':
      task(m, e.task_id).fixes += 1
      break
    case 'progress': {
      const { type: _t, run_id: _r, ...rest } = e
      task(m, e.task_id).lastProgress = rest
      break
    }
    case 'experiment': {
      const { type: _t, ts: _ts, run_id: _r, ...result } = e
      task(m, e.task_id).experiments.push(result as ExperimentResult)
      break
    }
    case 'research_done':
      task(m, e.task_id).researchDone = { reason: e.reason, best: e.best, n_exp: e.n_exp }
      break
    case 'full_train_done': {
      const { type: _t, ts: _ts, run_id: _r, ...result } = e
      task(m, e.task_id).fullTrain = result as ExperimentResult
      break
    }
    case 'full_train_failed':
      task(m, e.task_id).fullTrainError = e.error
      break
    case 'final_test': {
      const { type: _t, ts: _ts, run_id: _r, ...report } = e
      task(m, e.task_id).finalTest = report as FinalReport
      break
    }
    case 'report':
      m.report = e.summary
      break
    case 'deployed':
      m.deployment = e.deployment
      break
    case 'run_finished':
      m.finishedStatus = e.status
      m.endedAt = e.ts
      if (m.currentStage) m.stages[m.currentStage] = { ...m.stages[m.currentStage], endedAt: e.ts }
      break
    case 'run_failed':
      m.failedError = e.error
      m.endedAt = e.ts
      break
  }
}

export function buildModel(events: AnyEvent[]): RunModel {
  const m = emptyModel()
  for (const e of events) applyEvent(m, e)
  return m
}

// ---- derived views

export const TERMINAL = new Set(['done', 'failed', 'rejected', 'no_tasks'])

export type StepState = 'pending' | 'active' | 'waiting' | 'complete' | 'failed' | 'stopped' | 'skipped'

/** Per-stage state for the stepper, from stage events plus the backend's run status. */
export function stageStates(m: RunModel, status: string | undefined): Record<Stage, StepState> {
  const terminal = !!status && TERMINAL.has(status)
  const out = {} as Record<Stage, StepState>
  for (const { id } of STAGES) {
    const started = m.stages[id]?.startedAt !== undefined
    if (!started) out[id] = terminal ? 'skipped' : 'pending'
    else if (id !== m.currentStage) out[id] = 'complete'
    else if (!terminal) out[id] = status === 'awaiting_approval' ? 'waiting' : 'active'
    else if (status === 'done') out[id] = 'complete'
    else if (status === 'failed') out[id] = 'failed'
    else out[id] = 'stopped' // rejected / no_tasks end the graph at this stage
  }
  return out
}

/**
 * The backend marks an experiment `keep` when it beat the best so far (lower score), so the last `keep` is the
 * current best. Falls back to the lowest score if no row is marked keep.
 */
export function bestExperiment(rows: ExperimentResult[]): ExperimentResult | undefined {
  for (let i = rows.length - 1; i >= 0; i--) if (rows[i].status === 'keep') return rows[i]
  let best: ExperimentResult | undefined
  for (const r of rows) if (r.score != null && (best?.score == null || r.score < best.score)) best = r
  return best
}

/** Short human phase for a task card, from the furthest fact we have. */
export function taskPhase(t: TaskView): string {
  if (t.dropped) return 'Dropped'
  if (t.finalTest) return t.finalTest.passed ? 'Passed final gate' : 'Failed final gate'
  if (t.fullTrainError) return 'Full train failed'
  if (t.fullTrain) return 'Full train done'
  if (t.researchDone) return 'Research done'
  if (t.experiments.length || t.proposals) return 'Researching'
  if (t.plan || t.notes) return 'Exploring data'
  const v = t.validations.at(-1)
  if (v?.ok) return 'prepare.py validated'
  if (t.designAttempts) return `Designing (attempt ${t.designAttempts})`
  if (t.discovery) return 'Streams profiled'
  return 'Specified'
}
