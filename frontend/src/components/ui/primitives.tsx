import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from 'react'
import { Link, type LinkProps } from 'react-router'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { buttonClasses, type ButtonSize, type ButtonVariant } from './buttonStyles'

// ---- Button

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  loading?: boolean
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', loading, className, children, disabled, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={buttonClasses(variant, size, cn('disabled:pointer-events-none disabled:opacity-50', className))}
      {...props}
    >
      {loading && <Loader2 className="size-3.5 animate-spin" />}
      {children}
    </button>
  )
})

/** A router link styled as a button (avoids nesting <button> inside <a>). */
export function ButtonLink({
  variant,
  size,
  className,
  ...props
}: LinkProps & { variant?: ButtonVariant; size?: ButtonSize }) {
  return <Link className={buttonClasses(variant, size, className)} {...props} />
}

// ---- Card

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('rounded-xl border border-line bg-surface', className)} {...props} />
}

export function CardHeader({
  title,
  description,
  actions,
  icon,
  className,
}: {
  title: ReactNode
  description?: ReactNode
  actions?: ReactNode
  icon?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex items-start justify-between gap-3 border-b border-line px-4 py-3', className)}>
      <div className="flex min-w-0 items-start gap-2.5">
        {icon && <div className="mt-0.5 text-fg-subtle">{icon}</div>}
        <div className="min-w-0">
          <h3 className="text-sm font-medium text-fg">{title}</h3>
          {description && <p className="mt-0.5 text-xs text-fg-muted">{description}</p>}
        </div>
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  )
}

// ---- Badge

export type Tone = 'neutral' | 'accent' | 'teal' | 'ok' | 'warn' | 'bad'

const toneClasses: Record<Tone, string> = {
  neutral: 'bg-surface-3 text-fg-muted border-line-strong',
  accent: 'bg-accent-soft text-accent border-accent/25',
  teal: 'bg-teal-soft text-teal border-teal/25',
  ok: 'bg-ok-soft text-ok border-ok/25',
  warn: 'bg-warn-soft text-warn border-warn/25',
  bad: 'bg-bad-soft text-bad border-bad/25',
}

export function Badge({
  tone = 'neutral',
  className,
  children,
  dot,
  pulse,
}: {
  tone?: Tone
  className?: string
  children: ReactNode
  dot?: boolean
  pulse?: boolean
}) {
  return (
    <span
      className={cn(
        'inline-flex h-5 shrink-0 items-center gap-1.5 rounded-full border px-2 text-[11px] font-medium whitespace-nowrap',
        toneClasses[tone],
        className,
      )}
    >
      {dot && <span className={cn('size-1.5 rounded-full bg-current', pulse && 'animate-pulse-dot')} />}
      {children}
    </span>
  )
}

// ---- Misc

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded border border-line-strong bg-surface-3 px-1.5 py-0.5 font-mono text-[10px] text-fg-muted">
      {children}
    </kbd>
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-md bg-surface-3', className)} />
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn('size-4 animate-spin text-fg-subtle', className)} />
}

export function Mono({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return <span className={cn('font-mono text-[12px]', className)} {...props} />
}

export function Label({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('text-[11px] font-medium tracking-wide text-fg-subtle uppercase', className)}>{children}</div>
}

export function Stat({ label, value, sub }: { label: ReactNode; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="min-w-0">
      <Label>{label}</Label>
      <div className="mt-1 truncate text-lg font-medium tabular text-fg">{value}</div>
      {sub && <div className="mt-0.5 truncate text-xs text-fg-muted">{sub}</div>}
    </div>
  )
}

export const inputClass =
  'w-full rounded-md border border-line-strong bg-surface-2 px-3 py-2 text-sm text-fg placeholder:text-fg-subtle ' +
  'transition-colors hover:border-fg-subtle/40 focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20 ' +
  'disabled:opacity-60'
