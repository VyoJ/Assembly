import type { AnyEvent, RunEvent } from '@/api/events'
import { STAGES } from '@/api/events'
import type { Tone } from '@/components/ui/primitives'
import { fmt } from '@/lib/utils'

export interface EventLine {
  title: string
  detail?: string
  tone: Tone
  /** Low-signal events (tool calls, progress ticks) the feed can hide. */
  noisy?: boolean
}

const stageLabel = (id: string) => STAGES.find((s) => s.id === id)?.label ?? id

/** One-line, factual rendering of an event. Unknown types fall back to their type name. */
export function describe(raw: AnyEvent): EventLine {
  const e = raw as RunEvent
  switch (e.type) {
    case 'run_started':
      return { title: 'Run started', detail: e.auto_approve ? 'auto-approve on' : 'manual approval', tone: 'accent' }
    case 'stage':
      return { title: `Stage: ${stageLabel(e.stage)}`, tone: 'accent' }
    case 'task_specs':
      return {
        title: `Formulated ${e.tasks.length} task${e.tasks.length === 1 ? '' : 's'}`,
        detail: e.tasks.map((t) => t.id).join(', ') + (e.warnings?.length ? ` · ${e.warnings.length} warning(s) repaired` : ''),
        tone: 'teal',
      }
    case 'discovery':
      return { title: `Profiled streams for ${e.task_id}`, detail: `kept ${e.kept} of ${e.scanned} scanned`, tone: 'neutral' }
    case 'design_attempt':
      return { title: `Writing prepare.py (attempt ${e.attempt})`, detail: e.task_id, tone: 'neutral' }
    case 'tool_call':
      return { title: `Tool: ${e.tool}`, detail: e.task_id, tone: 'neutral', noisy: true }
    case 'validation':
      return {
        title: `Validation ${e.ok ? 'passed' : 'failed'} (attempt ${e.attempt})`,
        detail: e.ok ? e.task_id : `${e.task_id}: ${e.messages[0] ?? 'see checks'}`,
        tone: e.ok ? 'ok' : 'bad',
      }
    case 'task_dropped':
      return { title: `Dropped ${e.task_id}`, detail: e.reason, tone: 'bad' }
    case 'approval_required':
      return { title: 'Waiting for human approval', detail: `${Object.keys(e.tasks).length} task(s) to review`, tone: 'warn' }
    case 'approved':
      return { title: 'Plan approved', detail: e.tasks.map((t) => t.id).join(', ') || 'no tasks kept', tone: 'ok' }
    case 'rejected':
      return { title: 'Plan rejected', detail: e.reason || undefined, tone: 'bad' }
    case 'exploration':
      return { title: `Explored data for ${e.task_id}`, detail: 'notes.md written', tone: 'neutral' }
    case 'plan':
      return { title: e.revision ? `Revised research plan (${e.task_id})` : `Research plan (${e.task_id})`, tone: 'neutral' }
    case 'proposal':
      return { title: `Proposed ${e.exp_id}`, detail: e.description || e.hypothesis, tone: 'accent' }
    case 'proposal_invalid':
      return { title: `Unusable proposal (${e.task_id})`, detail: e.error.split('\n')[0], tone: 'warn' }
    case 'repair':
      return { title: `Repairing code (attempt ${e.attempt})`, detail: `${e.task_id} · ${e.step}`, tone: 'warn' }
    case 'fix':
      return { title: `Fixing ${e.of}`, detail: e.description, tone: 'warn' }
    case 'progress': {
      const parts = Object.entries(e)
        .filter(([k]) => !['type', 'ts', 'run_id', 'task_id', 'line'].includes(k))
        .map(([k, v]) => `${k}=${typeof v === 'number' ? fmt(v) : String(v)}`)
      return { title: `Progress (${e.task_id})`, detail: parts.join(' · ') || (e.line as string | undefined), tone: 'neutral', noisy: true }
    }
    case 'experiment':
      return {
        title: `${e.exp_id} → ${e.status}`,
        detail: `${e.task_id} · score ${fmt(e.score)}${e.description ? ` · ${e.description}` : ''}`,
        tone: e.status === 'keep' ? 'ok' : e.status === 'discard' ? 'neutral' : 'bad',
      }
    case 'research_done':
      return { title: `Research finished (${e.task_id})`, detail: `${e.n_exp} experiments · ${e.reason}`, tone: 'teal' }
    case 'full_train_done':
      return { title: `Full train done (${e.task_id})`, detail: `score ${fmt(e.score)}`, tone: 'ok' }
    case 'full_train_failed':
      return { title: `Full train failed (${e.task_id})`, detail: e.error, tone: 'bad' }
    case 'final_test':
      return {
        title: `Final gate ${e.passed ? 'passed' : 'failed'} (${e.task_id})`,
        detail: e.failed_kpis.length ? `failed: ${e.failed_kpis.join(', ')}` : undefined,
        tone: e.passed ? 'ok' : 'bad',
      }
    case 'report':
      return { title: 'Final report written', tone: 'teal' }
    case 'deployed':
      return { title: `Deployed ${e.deployment.id}`, detail: e.deployment.task_ids.join(', '), tone: 'teal' }
    case 'run_finished':
      return { title: 'Run finished', detail: e.status ? `status: ${e.status}` : undefined, tone: e.status === 'done' ? 'ok' : 'neutral' }
    case 'run_failed':
      return { title: 'Run failed', detail: e.error, tone: 'bad' }
    default:
      return { title: raw.type, tone: 'neutral' }
  }
}

export function eventTaskId(e: AnyEvent): string | undefined {
  const t = (e as { task_id?: unknown }).task_id
  return typeof t === 'string' ? t : undefined
}
