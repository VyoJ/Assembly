import type {
  ApprovalBody,
  Catalog,
  Deployment,
  ExperimentResult,
  Health,
  Prediction,
  PreviewPoint,
  RunDetail,
  RunSummary,
  StartRunBody,
} from './types'

/** Empty = same origin (the Vite proxy forwards /api and /health to the backend). */
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

export class ApiError extends Error {
  readonly status: number
  readonly detail: string

  constructor(status: number, detail: string) {
    super(status === 0 ? detail : `${status}: ${detail}`)
    this.status = status
    this.detail = detail
  }

  /** The backend could not be reached at all (down, proxy error). */
  get unreachable(): boolean {
    return this.status === 0 || this.status === 502 || this.status === 503 || this.status === 504
  }
}

export function url(path: string): string {
  return `${API_BASE}${path}`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url(path), {
      ...init,
      headers: { Accept: 'application/json', ...(init?.body ? { 'Content-Type': 'application/json' } : {}), ...init?.headers },
    })
  } catch {
    throw new ApiError(0, 'Backend unreachable')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail ?? body)
    } catch {
      /* non-JSON error body (e.g. proxy error page) */
    }
    throw new ApiError(res.status, detail || 'Request failed')
  }
  return res.json() as Promise<T>
}

const enc = encodeURIComponent

export const api = {
  health: () => request<Health>('/health'),

  listRuns: () => request<RunSummary[]>('/api/runs'),
  getRun: (id: string) => request<RunDetail>(`/api/runs/${enc(id)}`),
  startRun: (body: StartRunBody) => request<RunSummary>('/api/runs', { method: 'POST', body: JSON.stringify(body) }),
  approve: (id: string, body: ApprovalBody) =>
    request<{ ok: boolean }>(`/api/runs/${enc(id)}/approve`, { method: 'POST', body: JSON.stringify(body) }),
  experiments: (id: string, taskId: string) =>
    request<ExperimentResult[]>(`/api/runs/${enc(id)}/tasks/${enc(taskId)}/experiments`),
  runEventsPath: (id: string) => `/api/runs/${enc(id)}/events`,

  catalog: () => request<Catalog>('/api/streams'),
  preview: (streamId: string, n = 50) => request<PreviewPoint[]>(`/api/streams/${enc(streamId)}/preview?n=${n}`),

  deployments: () => request<Deployment[]>('/api/deployments'),
  predict: (depId: string, sample: Record<string, unknown>, taskId?: string) =>
    request<Record<string, Prediction>>(
      `/api/deployments/${enc(depId)}/predict${taskId ? `?task_id=${enc(taskId)}` : ''}`,
      { method: 'POST', body: JSON.stringify(sample) },
    ),
  livePath: (depId: string, hz: number) => `/api/deployments/${enc(depId)}/live?hz=${hz}`,
}
