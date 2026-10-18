import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Run ids look like 20261018-124800-9d65 (agent/graph.py new_run_id). Returns null for other shapes. */
export function runIdDate(runId: string): Date | null {
  const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})-/.exec(runId)
  if (!m) return null
  const [, y, mo, d, h, mi, s] = m.map(Number)
  return new Date(y, mo - 1, d, h, mi, s)
}

export function timeAgo(date: Date | number | null | undefined, now = Date.now()): string {
  if (date == null) return '—'
  const t = typeof date === 'number' ? date : date.getTime()
  const s = Math.round((now - t) / 1000)
  if (s < 5) return 'just now'
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return new Date(t).toLocaleDateString()
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—'
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  if (m < 60) return `${m}m ${s}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

export function formatClock(tsSeconds: number): string {
  return new Date(tsSeconds * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

/** Compact number formatting that never invents precision. */
export function fmt(value: unknown, digits = 4): string {
  if (typeof value !== 'number') return value == null ? '—' : String(value)
  if (!Number.isFinite(value)) return String(value)
  if (value !== 0 && (Math.abs(value) >= 1e5 || Math.abs(value) < 1e-3)) return value.toExponential(2)
  return Number(value.toPrecision(digits)).toString()
}

export function kpiLabel(k: { metric: string; op: string; threshold: number }): string {
  return `${k.metric} ${k.op} ${k.threshold}`
}
