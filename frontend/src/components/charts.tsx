// Shared Recharts styling: recessive grid/axes, 2px lines, text in text tokens.
import type { ReactNode } from 'react'
import { fmt } from '@/lib/utils'

interface TooltipRow {
  label: string
  value: ReactNode
}

/** Tooltip body used by every chart (pass via `content`). */
export function ChartTooltip({ title, rows }: { title: ReactNode; rows: TooltipRow[] }) {
  return (
    <div className="max-w-xs rounded-lg border border-line-strong bg-surface-2 px-3 py-2 text-xs shadow-xl">
      <div className="mb-1 font-medium text-fg">{title}</div>
      {rows.map((r) => (
        <div key={r.label} className="flex justify-between gap-4">
          <span className="text-fg-muted">{r.label}</span>
          <span className="font-mono text-fg tabular">{typeof r.value === 'number' ? fmt(r.value) : r.value}</span>
        </div>
      ))}
    </div>
  )
}

export function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] text-fg-muted">
      <span className="size-2 rounded-full" style={{ background: color }} />
      {label}
    </span>
  )
}
