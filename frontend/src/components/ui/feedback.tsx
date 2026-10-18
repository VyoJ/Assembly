import type { ReactNode } from 'react'
import { AlertTriangle, PlugZap, RotateCw } from 'lucide-react'
import type { ApiError } from '@/api/client'
import { cn } from '@/lib/utils'
import { Button } from './primitives'

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon?: ReactNode
  title: ReactNode
  description?: ReactNode
  action?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex flex-col items-center justify-center px-6 py-12 text-center', className)}>
      {icon && (
        <div className="mb-3 flex size-10 items-center justify-center rounded-lg border border-line bg-surface-2 text-fg-subtle">
          {icon}
        </div>
      )}
      <div className="text-sm font-medium text-fg">{title}</div>
      {description && <div className="mt-1 max-w-sm text-xs leading-relaxed text-fg-muted">{description}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function ErrorState({
  error,
  onRetry,
  title,
  className,
}: {
  error: ApiError | Error | string
  onRetry?: () => void
  title?: string
  className?: string
}) {
  const unreachable = typeof error === 'object' && 'unreachable' in error && error.unreachable
  const message = typeof error === 'string' ? error : error.message
  return (
    <div className={cn('flex flex-col items-center justify-center px-6 py-10 text-center', className)}>
      <div
        className={cn(
          'mb-3 flex size-10 items-center justify-center rounded-lg border',
          unreachable ? 'border-warn/30 bg-warn-soft text-warn' : 'border-bad/30 bg-bad-soft text-bad',
        )}
      >
        {unreachable ? <PlugZap className="size-5" /> : <AlertTriangle className="size-5" />}
      </div>
      <div className="text-sm font-medium text-fg">
        {title ?? (unreachable ? 'Backend unavailable' : 'Request failed')}
      </div>
      <div className="mt-1 max-w-md text-xs leading-relaxed text-fg-muted">
        {unreachable ? (
          <>
            Can't reach the Assembly API. Start it with{' '}
            <code className="font-mono text-fg">uv run uvicorn api.main:app --port 8000</code>.
          </>
        ) : (
          message
        )}
      </div>
      {onRetry && (
        <Button size="sm" className="mt-4" onClick={onRetry}>
          <RotateCw className="size-3.5" /> Retry
        </Button>
      )}
    </div>
  )
}

export function Callout({
  tone = 'neutral',
  icon,
  title,
  children,
  className,
}: {
  tone?: 'neutral' | 'warn' | 'bad' | 'ok' | 'accent'
  icon?: ReactNode
  title?: ReactNode
  children?: ReactNode
  className?: string
}) {
  const tones = {
    neutral: 'border-line bg-surface-2 text-fg-muted',
    warn: 'border-warn/25 bg-warn-soft text-warn',
    bad: 'border-bad/25 bg-bad-soft text-bad',
    ok: 'border-ok/25 bg-ok-soft text-ok',
    accent: 'border-accent/25 bg-accent-soft text-accent',
  }
  return (
    <div className={cn('flex gap-2.5 rounded-lg border px-3 py-2.5 text-xs leading-relaxed', tones[tone], className)}>
      {icon && <div className="mt-px shrink-0">{icon}</div>}
      <div className="min-w-0">
        {title && <div className="font-medium">{title}</div>}
        {children && <div className={cn(title && 'mt-0.5', 'text-fg-muted')}>{children}</div>}
      </div>
    </div>
  )
}
