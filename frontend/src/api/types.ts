// Mirrors src/core/contracts.py and the payloads in src/api/*.py. Keep in sync by hand
// (or regenerate from http://localhost:8000/openapi.json, which only covers request bodies).

export type Modality = 'video' | 'image' | 'vibration' | 'pressure' | 'temperature' | 'plc' | 'log' | 'other'
export type PredictionFormat = 'scalar' | 'binary_score' | 'class_probs' | 'boxes' | 'masks'

export interface StreamInfo {
  id: string
  name: string
  modality: Modality
  unit: string | null
  source: string
  sample_hz: number | null
  description: string
}

export interface LabelSource {
  id: string
  name: string
  description: string
  kind: 'events' | 'annotations' | 'table'
}

export interface KPI {
  metric: string
  op: '<=' | '>='
  threshold: number
}

export interface TaskSpec {
  id: string
  name: string
  rationale: string
  framing: string
  input_streams: string[]
  label_sources: string[]
  target_definition: string
  prediction_format: PredictionFormat
  primary_metric: string
  metrics: string[]
  kpis: KPI[]
  split_strategy: string
  time_budget_s: number
}

export interface StreamProfile {
  stream_id: string
  relevance: number
  keep: boolean
  reason: string
}

export type ExperimentStatus = 'keep' | 'discard' | 'crash' | 'timeout' | 'invalid_output'

/** ExperimentResult. `score` is LOWER = BETTER (contracts.py / ml/scoring.py). */
export interface ExperimentResult {
  exp_id: string
  task_id: string
  status: ExperimentStatus
  score: number | null
  metrics: Record<string, number>
  kpis_passed: Record<string, boolean>
  train_seconds: number
  peak_mem_gb: number | null
  description: string
  hypothesis: string
  log_tail?: string
  diff?: string
}

export interface FinalReport {
  task_id: string
  exp_id: string
  test_metrics: Record<string, number>
  passed: boolean
  failed_kpis: string[]
}

export interface Deployment {
  id: string
  run_id: string
  task_ids: string[]
  exp_ids: string[]
  package_paths: Record<string, string>
  endpoint: string
  status: 'live' | 'stopped'
}

// ---- /health

export interface Health {
  ok: boolean
  ml_mock: boolean
  llm_mock: boolean
  llm_model: string
}

// ---- /api/runs

/** `running` | `awaiting_approval` | `done` | `failed` | `rejected` | `no_tasks` (strings set by the backend). */
export type RunStatus = string

export interface RunSummary {
  run_id: string
  goal: string
  status: RunStatus
  error: string | null
  /** "recorded": read back from runs/{id}/events.jsonl by a later API process (replay only, no live control) */
  source?: 'live' | 'recorded'
}

/** OuterState as JSON (src/agent/state.py). Only updated after each pipeline node finishes. */
export interface RunState {
  run_id?: string
  goal?: string
  auto_approve?: boolean
  status?: string
  spec?: { goal: string; tasks: TaskSpec[] }
  streams?: StreamInfo[]
  label_sources?: LabelSource[]
  profiles?: Record<string, StreamProfile[]>
  workspaces?: Record<string, { run_id: string; task_id: string; path: string; spec: TaskSpec; frozen: boolean }>
  dropped?: Record<string, string>
  best?: Record<string, ExperimentResult>
  full?: Record<string, ExperimentResult>
  reports?: Record<string, FinalReport>
  summary?: string
  deployment?: Deployment
}

export interface RunDetail extends RunSummary {
  state: RunState
}

export interface StartRunBody {
  goal: string
  auto_approve: boolean
}

/** POST /api/runs/{id}/approve. Only kpis / time_budget_s / primary_metric / metrics of edited tasks apply. */
export interface ApprovalBody {
  approved: boolean
  tasks: Array<Pick<TaskSpec, 'id'> & Partial<Pick<TaskSpec, 'kpis' | 'time_budget_s' | 'primary_metric' | 'metrics'>>>
  drop: string[]
  reason: string
}

// ---- /api/streams

export interface Catalog {
  streams: StreamInfo[]
  label_sources: LabelSource[]
}

/** Simulator output: numeric streams give a number, image streams give {frame_id}. */
export interface PreviewPoint {
  cycle: number
  value: number | { frame_id: number } | unknown
}

// ---- /api/deployments

/** One task's prediction; keys depend on the prediction format (see ml/package.py Predictor). */
export type Prediction = Record<string, unknown> & {
  alarm?: boolean
  latency_ms?: number
  /** boxes format: [x1, y1, x2, y2, score, class] in image pixels */
  boxes?: number[][]
  /** Live feed of a real deployment (api/deployments.py): the held-out test sample it replayed... */
  sample_id?: string
  /** ...its true label (boxes: [x1, y1, x2, y2, class] per defect), and whether the alarm was right */
  truth?: unknown
  outcome?: Outcome
  /** camera tasks: GET this for the image behind the prediction */
  frame_url?: string
}

export type Outcome = 'caught' | 'missed' | 'false_alarm' | 'ok'

export interface LiveTick {
  t: number
  cycle: number
  predictions: Record<string, Prediction>
  kpi_tracker: Record<string, { alarms: number; ticks: number; alarm_rate: number }>
}
