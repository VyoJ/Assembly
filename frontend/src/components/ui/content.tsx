import { Fragment, useState, type ReactNode } from 'react'
import { Check, ChevronRight, Copy } from 'lucide-react'
import { cn } from '@/lib/utils'

/** Monospace block for logs, code and diffs. `diff` colours +/- lines. */
export function CodeBlock({
  children,
  className,
  diff,
  maxHeight = 'max-h-80',
}: {
  children: string
  className?: string
  diff?: boolean
  maxHeight?: string
}) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(children)
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch {
      /* clipboard unavailable */
    }
  }
  return (
    <div className={cn('group relative rounded-lg border border-line bg-canvas', className)}>
      <button
        onClick={copy}
        className="absolute top-2 right-2 rounded-md border border-line bg-surface-2 p-1 text-fg-subtle opacity-0 transition-opacity group-hover:opacity-100 hover:text-fg"
        aria-label="Copy"
      >
        {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
      </button>
      <pre className={cn('overflow-auto p-3 font-mono text-[11.5px] leading-relaxed text-fg-muted', maxHeight)}>
        {diff
          ? children.split('\n').map((line, i) => (
              <div
                key={i}
                className={cn(
                  line.startsWith('+') && !line.startsWith('+++') && 'bg-ok-soft text-ok',
                  line.startsWith('-') && !line.startsWith('---') && 'bg-bad-soft text-bad',
                  line.startsWith('@@') && 'text-accent',
                )}
              >
                {line || ' '}
              </div>
            ))
          : children}
      </pre>
    </div>
  )
}

function inline(text: string): ReactNode[] {
  // `code` and **bold** only; everything else stays literal text.
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).map((part, i) => {
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2)
      return (
        <code key={i} className="rounded bg-surface-3 px-1 py-px font-mono text-[0.9em] text-fg">
          {part.slice(1, -1)}
        </code>
      )
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4)
      return (
        <strong key={i} className="font-medium text-fg">
          {part.slice(2, -2)}
        </strong>
      )
    return <Fragment key={i}>{part}</Fragment>
  })
}

/** Tiny Markdown renderer for agent-written text (data cards, plans, notes, reports): headings, lists, code fences. */
export function Markdown({ text, className }: { text: string; className?: string }) {
  const blocks: ReactNode[] = []
  const lines = text.split('\n')
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (line.startsWith('```')) {
      const body: string[] = []
      i++
      while (i < lines.length && !lines[i].startsWith('```')) body.push(lines[i++])
      i++
      blocks.push(<CodeBlock key={blocks.length} className="my-2">{body.join('\n')}</CodeBlock>)
      continue
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(line)
    if (h) {
      const level = h[1].length
      blocks.push(
        <div
          key={blocks.length}
          className={cn('font-medium text-fg', level <= 1 ? 'mt-3 mb-1.5 text-sm' : 'mt-3 mb-1 text-[13px]')}
        >
          {inline(h[2])}
        </div>,
      )
      i++
      continue
    }
    if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const items: string[] = []
      const ordered = /^\s*\d+\./.test(line)
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*([-*]|\d+\.)\s+/, ''))
      const List = ordered ? 'ol' : 'ul'
      blocks.push(
        <List key={blocks.length} className={cn('my-1.5 space-y-0.5 pl-5', ordered ? 'list-decimal' : 'list-disc', 'marker:text-fg-subtle')}>
          {items.map((it, j) => (
            <li key={j}>{inline(it)}</li>
          ))}
        </List>,
      )
      continue
    }
    if (line.trim() === '') {
      i++
      continue
    }
    const para: string[] = []
    while (i < lines.length && lines[i].trim() !== '' && !/^(#{1,4}\s|```|\s*([-*]|\d+\.)\s)/.test(lines[i])) para.push(lines[i++])
    blocks.push(
      <p key={blocks.length} className="my-1.5">
        {inline(para.join(' '))}
      </p>,
    )
  }
  return <div className={cn('text-[13px] leading-relaxed break-words text-fg-muted', className)}>{blocks}</div>
}

/** Collapsible section with a chevron. */
export function Disclosure({
  summary,
  children,
  defaultOpen = false,
  className,
}: {
  summary: ReactNode
  children: ReactNode
  defaultOpen?: boolean
  className?: string
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={className}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1.5 text-left text-xs text-fg-muted hover:text-fg"
        aria-expanded={open}
      >
        <ChevronRight className={cn('size-3.5 shrink-0 transition-transform', open && 'rotate-90')} />
        {summary}
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  )
}

export function KeyValue({ items, className }: { items: [ReactNode, ReactNode][]; className?: string }) {
  return (
    <dl className={cn('grid grid-cols-[minmax(0,auto)_1fr] gap-x-4 gap-y-1.5 text-xs', className)}>
      {items.map(([k, v], i) => (
        <Fragment key={i}>
          <dt className="text-fg-subtle">{k}</dt>
          <dd className="min-w-0 break-words text-fg">{v}</dd>
        </Fragment>
      ))}
    </dl>
  )
}

export interface TabItem<T extends string> {
  id: T
  label: ReactNode
  badge?: ReactNode
}

export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
  className,
}: {
  tabs: TabItem<T>[]
  value: T
  onChange: (id: T) => void
  className?: string
}) {
  return (
    <div role="tablist" className={cn('flex gap-1 overflow-x-auto border-b border-line', className)}>
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={value === t.id}
          onClick={() => onChange(t.id)}
          className={cn(
            '-mb-px flex shrink-0 items-center gap-2 border-b-2 px-3 py-2.5 text-sm transition-colors',
            value === t.id ? 'border-accent text-fg' : 'border-transparent text-fg-muted hover:text-fg',
          )}
        >
          {t.label}
          {t.badge}
        </button>
      ))}
    </div>
  )
}

/** Validator / design messages can list hundreds of columns: show the start, the rest on demand. */
const LONG_TEXT_CHARS = 360

export function LongText({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  if (text.length <= LONG_TEXT_CHARS) return <>{text}</>
  return (
    <>
      <span className="break-words">{open ? text : `${text.slice(0, LONG_TEXT_CHARS)}…`}</span>{' '}
      <button onClick={() => setOpen((o) => !o)} className="text-[11px] underline underline-offset-2 opacity-80 hover:opacity-100">
        {open ? 'show less' : `show all (${text.length.toLocaleString()} characters)`}
      </button>
    </>
  )
}
