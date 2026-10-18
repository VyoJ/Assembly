import { useMemo, useState } from 'react'
import { Download, ScrollText } from 'lucide-react'
import type { AnyEvent } from '@/api/events'
import { cn, formatClock } from '@/lib/utils'
import { Button, Card, Mono, inputClass } from '@/components/ui/primitives'
import { EmptyState } from '@/components/ui/feedback'
import { CodeBlock } from '@/components/ui/content'
import { describe, eventTaskId } from './describe'

export function EventLogTab({ events, runId }: { events: AnyEvent[]; runId: string }) {
  const [type, setType] = useState('')
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState<number | null>(null)
  const types = useMemo(() => [...new Set(events.map((e) => e.type))].sort(), [events])
  const rows = useMemo(() => {
    const q = query.toLowerCase()
    return events
      .map((e, i) => ({ e, i }))
      .filter(({ e }) => (!type || e.type === type) && (!q || JSON.stringify(e).toLowerCase().includes(q)))
      .reverse()
  }, [events, type, query])

  const download = () => {
    const blob = new Blob([events.map((e) => JSON.stringify(e)).join('\n') + '\n'], { type: 'application/x-ndjson' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${runId}-events.jsonl`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <Card>
      <div className="flex flex-col gap-2 border-b border-line p-3 sm:flex-row sm:items-center">
        <select value={type} onChange={(e) => setType(e.target.value)} className={cn(inputClass, 'sm:w-48')} aria-label="Event type">
          <option value="">All types ({events.length})</option>
          {types.map((t) => (
            <option key={t} value={t}>
              {t} ({events.filter((e) => e.type === t).length})
            </option>
          ))}
        </select>
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search payloads…" className={inputClass} />
        <Button size="md" onClick={download} disabled={!events.length}>
          <Download className="size-3.5" /> JSONL
        </Button>
      </div>
      {rows.length === 0 ? (
        <EmptyState icon={<ScrollText className="size-5" />} title={events.length ? 'No matching events' : 'No events received yet'} />
      ) : (
        <div className="max-h-[70vh] divide-y divide-line overflow-y-auto">
          {rows.slice(0, 500).map(({ e, i }) => {
            const line = describe(e)
            return (
              <div key={i}>
                <button
                  onClick={() => setOpen(open === i ? null : i)}
                  className="grid w-full grid-cols-[64px_minmax(0,140px)_1fr] items-baseline gap-3 px-3 py-2 text-left text-xs hover:bg-surface-2"
                >
                  <Mono className="text-[11px] text-fg-subtle">{formatClock(e.ts)}</Mono>
                  <Mono className="truncate text-accent">{e.type}</Mono>
                  <span className="min-w-0 truncate text-fg-muted">
                    {eventTaskId(e) && <Mono className="mr-2 text-fg-subtle">{eventTaskId(e)}</Mono>}
                    {line.detail ?? line.title}
                  </span>
                </button>
                {open === i && (
                  <div className="px-3 pb-3">
                    <CodeBlock>{JSON.stringify(e, null, 2)}</CodeBlock>
                  </div>
                )}
              </div>
            )
          })}
          {rows.length > 500 && <div className="px-3 py-2 text-center text-[11px] text-fg-subtle">Showing the latest 500 of {rows.length}. Download for the full log.</div>}
        </div>
      )}
    </Card>
  )
}
