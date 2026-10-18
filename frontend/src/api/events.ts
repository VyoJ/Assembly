// Run events as emitted with core.events.bus.emit(...) in src/agent. Every event has type/ts/run_id;
// the rest is per type. Unknown types are kept as-is and shown in the raw log.
import type { ExperimentResult, FinalReport, Deployment, StreamProfile, TaskSpec } from './types'

interface Base {
  type: string
  ts: number
  run_id: string
}

export type Stage =
  | 'formulate'
  | 'discover'
  | 'design'
  | 'approve_plan'
  | 'research'
  | 'full_train'
  | 'final_gate'
  | 'deploy'

export type RunEvent = Base &
  (
    | { type: 'run_started'; goal: string; auto_approve: boolean }
    | { type: 'run_finished'; status: string | null }
    | { type: 'run_failed'; error: string }
    | { type: 'stage'; stage: Stage }
    | { type: 'task_specs'; tasks: TaskSpec[]; warnings: string[] }
    | { type: 'discovery'; task_id: string; scanned: number; kept: number; profiles: StreamProfile[] }
    | { type: 'design_attempt'; task_id: string; attempt: number }
    | { type: 'tool_call'; task_id: string; tool: string; args: unknown }
    | { type: 'validation'; task_id: string; attempt: number; ok: boolean; checks: Record<string, boolean>; messages: string[] }
    | { type: 'task_dropped'; task_id: string; reason: string }
    | { type: 'approval_required'; tasks: Record<string, { spec: TaskSpec; data_card: string }> }
    | { type: 'rejected'; reason: string }
    | { type: 'approved'; tasks: TaskSpec[] }
    | { type: 'exploration'; task_id: string; notes: string; raw: string }
    | { type: 'plan'; task_id: string; plan: string; revision: boolean }
    | { type: 'proposal_invalid'; task_id: string; error: string }
    | { type: 'proposal'; task_id: string; exp_id: string; hypothesis: string; description: string }
    | { type: 'progress'; task_id: string; stage?: string; exp_id?: string; line?: string; [k: string]: unknown }
    | ({ type: 'experiment' } & ExperimentResult)
    | { type: 'repair'; task_id: string; step: string; attempt: number; problems: string }
    | { type: 'fix'; task_id: string; of: string; description: string }
    | { type: 'research_done'; task_id: string; reason: string; best: ExperimentResult | null; n_exp: number }
    | ({ type: 'full_train_done' } & ExperimentResult)
    | { type: 'full_train_failed'; task_id: string; error: string }
    | ({ type: 'final_test' } & FinalReport)
    | { type: 'report'; summary: string }
    | { type: 'deployed'; deployment: Deployment }
  )

export type AnyEvent = RunEvent | (Base & Record<string, unknown>)

export const STAGES: { id: Stage; label: string; hint: string }[] = [
  { id: 'formulate', label: 'Formulate', hint: 'Goal → ML task specs' },
  { id: 'discover', label: 'Discover', hint: 'Profile plant streams' },
  { id: 'design', label: 'Design', hint: 'Write + validate prepare.py' },
  { id: 'approve_plan', label: 'Approve', hint: 'Human review of the plan' },
  { id: 'research', label: 'Research', hint: 'Autoresearch experiments' },
  { id: 'full_train', label: 'Full train', hint: 'Retrain the winner' },
  { id: 'final_gate', label: 'Final gate', hint: 'Hidden test split' },
  { id: 'deploy', label: 'Deploy', hint: 'Package + register' },
]
