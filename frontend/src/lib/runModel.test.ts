// Test fixtures below are hand-written in the shape of events captured from the backend in mock mode.
import { describe, expect, it } from 'vitest'
import type { AnyEvent } from '@/api/events'
import type { ExperimentResult, TaskSpec } from '@/api/types'
import { mergeReplayed, SseParser } from '@/api/sse'
import { bestExperiment, buildModel, stageStates, taskPhase } from './runModel'

const spec: TaskSpec = {
  id: 'press_failure_30c', name: 'Press failure within 30 cycles', rationale: 'r', framing: 'f',
  input_streams: ['press3.fan_speed'], label_sources: ['maintenance_log'], target_definition: 't',
  prediction_format: 'binary_score', primary_metric: 'pr_auc', metrics: ['pr_auc', 'recall'],
  kpis: [{ metric: 'recall', op: '>=', threshold: 0.9 }], split_strategy: 'by unit', time_budget_s: 180,
}

const exp = (n: number, status: ExperimentResult['status'], score: number | null): ExperimentResult => ({
  exp_id: `exp_${String(n).padStart(3, '0')}`, task_id: spec.id, status, score, metrics: { pr_auc: 0.5 + n / 10 },
  kpis_passed: { 'recall >= 0.9': status === 'keep' }, train_seconds: 10, peak_mem_gb: null, description: '', hypothesis: '',
})

let ts = 100
const ev = (type: string, data: Record<string, unknown> = {}): AnyEvent => ({ type, ts: ts++, run_id: 'r1', ...data }) as AnyEvent

const upToApproval: AnyEvent[] = [
  ev('run_started', { goal: 'Predict press failures', auto_approve: false }),
  ev('stage', { stage: 'formulate' }),
  ev('task_specs', { tasks: [spec], warnings: [] }),
  ev('stage', { stage: 'discover' }),
  ev('discovery', { task_id: spec.id, scanned: 20, kept: 12, profiles: [] }),
  ev('stage', { stage: 'design' }),
  ev('design_attempt', { task_id: spec.id, attempt: 1 }),
  ev('validation', { task_id: spec.id, attempt: 1, ok: false, checks: { group_disjoint: false }, messages: ['overlap'] }),
  ev('design_attempt', { task_id: spec.id, attempt: 2 }),
  ev('validation', { task_id: spec.id, attempt: 2, ok: true, checks: { group_disjoint: true }, messages: [] }),
  ev('stage', { stage: 'approve_plan' }),
  ev('approval_required', { tasks: { [spec.id]: { spec, data_card: '# card' } } }),
]

describe('buildModel', () => {
  it('folds the design phase and the approval request', () => {
    const m = buildModel(upToApproval)
    expect(m.goal).toBe('Predict press failures')
    expect(m.autoApprove).toBe(false)
    expect(m.currentStage).toBe('approve_plan')
    expect(m.stages.formulate?.endedAt).toBeDefined()
    const t = m.tasks[spec.id]
    expect(t.discovery?.kept).toBe(12)
    expect(t.designAttempts).toBe(2)
    expect(t.validations.map((v) => v.ok)).toEqual([false, true])
    expect(m.approvalRequest?.tasks[spec.id].data_card).toBe('# card')
    expect(taskPhase(t)).toBe('prepare.py validated')
  })

  it('records approval edits, experiments and the end of the run', () => {
    const edited = { ...spec, time_budget_s: 300 }
    const m = buildModel([
      ...upToApproval,
      ev('approved', { tasks: [edited] }),
      ev('stage', { stage: 'research' }),
      ev('proposal', { task_id: spec.id, exp_id: 'exp_000', hypothesis: 'h', description: 'baseline' }),
      ev('experiment', { ...exp(0, 'keep', 1000.4), diff: '+x' }),
      ev('experiment', { ...exp(1, 'crash', null) }),
      ev('research_done', { task_id: spec.id, reason: 'kpis met', best: exp(0, 'keep', 1000.4), n_exp: 2 }),
      ev('final_test', { task_id: spec.id, exp_id: 'exp_000', test_metrics: { recall: 0.93 }, passed: true, failed_kpis: [] }),
      ev('run_finished', { status: 'done' }),
    ])
    const t = m.tasks[spec.id]
    expect(m.approvalOutcome?.approved).toBe(true)
    expect(t.spec?.time_budget_s).toBe(300)
    expect(t.experiments).toHaveLength(2)
    expect(t.experiments[0].diff).toBe('+x')
    expect(t.experiments[0]).not.toHaveProperty('type')
    expect(t.finalTest?.passed).toBe(true)
    expect(m.finishedStatus).toBe('done')
    expect(taskPhase(t)).toBe('Passed final gate')
  })

  it('ignores unknown event types', () => {
    const m = buildModel([ev('something_new', { foo: 1 })])
    expect(m.taskOrder).toEqual([])
  })
})

describe('stageStates', () => {
  it('marks the approval stage as waiting while the backend awaits approval', () => {
    const s = stageStates(buildModel(upToApproval), 'awaiting_approval')
    expect(s.formulate).toBe('complete')
    expect(s.design).toBe('complete')
    expect(s.approve_plan).toBe('waiting')
    expect(s.research).toBe('pending')
  })

  it('marks unreached stages as skipped after a rejection', () => {
    const m = buildModel([...upToApproval, ev('rejected', { reason: 'no' }), ev('run_finished', { status: 'rejected' })])
    const s = stageStates(m, 'rejected')
    expect(s.approve_plan).toBe('stopped')
    expect(s.research).toBe('skipped')
    expect(s.deploy).toBe('skipped')
  })

  it('marks the current stage failed when the run failed', () => {
    expect(stageStates(buildModel(upToApproval.slice(0, 6)), 'failed').design).toBe('failed')
  })
})

describe('bestExperiment', () => {
  it('uses the last keep, as the backend marks new bests with keep', () => {
    expect(bestExperiment([exp(0, 'keep', 5), exp(1, 'discard', 1), exp(2, 'keep', 3)])?.exp_id).toBe('exp_002')
  })
  it('falls back to the lowest score (lower is better)', () => {
    expect(bestExperiment([exp(0, 'discard', 5), exp(1, 'discard', 2), exp(2, 'crash', null)])?.exp_id).toBe('exp_001')
  })
  it('returns undefined for no scored rows', () => {
    expect(bestExperiment([exp(0, 'crash', null)])).toBeUndefined()
  })
})

describe('SseParser', () => {
  it('parses named events split across chunks and skips pings', () => {
    const p = new SseParser()
    expect(p.push('event: stage\r\ndata: {"a"')).toEqual([])
    expect(p.push(':1}\r\n\r\n: ping - 2026\r\n\r\nevent: x\ndata: 1\ndata: 2\n\n')).toEqual([
      { event: 'stage', data: '{"a":1}' },
      { event: 'x', data: '1\n2' },
    ])
  })
  it('defaults the event name to message', () => {
    expect(new SseParser().push('data: hi\n\n')).toEqual([{ event: 'message', data: 'hi' }])
  })
})

describe('mergeReplayed', () => {
  const e = (t: number, type = 'stage') => ({ ts: t, type })
  it('skips the replayed prefix after a reconnect and appends only new events', () => {
    const known = [e(1), e(2), e(3)]
    let cursor = 0 // new connection
    const appended: number[] = []
    for (const ev of [e(1), e(2), e(3), e(4), e(5)]) {
      const r = mergeReplayed(known, cursor, ev)
      cursor = r.cursor
      if (r.appended) appended.push(ev.ts)
    }
    expect(appended).toEqual([4, 5])
    expect(known.map((x) => x.ts)).toEqual([1, 2, 3, 4, 5])
  })
  it('truncates local history when the server history diverges', () => {
    const known = [e(1), e(2), e(3)]
    let cursor = mergeReplayed(known, 0, e(1)).cursor
    cursor = mergeReplayed(known, cursor, e(9, 'other')).cursor
    expect(cursor).toBe(2)
    expect(known.map((x) => x.ts)).toEqual([1, 9])
  })
})
