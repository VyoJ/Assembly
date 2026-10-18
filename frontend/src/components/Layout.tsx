import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router'
import { Boxes, Database, Info, LayoutDashboard, Loader2, Menu, Plus, Rocket, X } from 'lucide-react'
import { useHealth } from '@/hooks/health'
import { openWelcome, WelcomeDialog } from './WelcomeDialog'
import { cn } from '@/lib/utils'
import { Badge } from './ui/primitives'

const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/runs/new', label: 'New run', icon: Plus },
  { to: '/deployments', label: 'Deployments', icon: Rocket },
  { to: '/data', label: 'Data sources', icon: Database },
]

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex size-7 items-center justify-center rounded-lg border border-line-strong bg-surface-3">
        <Boxes className="size-4 text-teal" />
      </div>
      <div className="leading-tight">
        <div className="text-sm font-semibold tracking-tight">Assembly</div>
        <div className="text-[10px] text-fg-subtle">Autonomous ML engineer</div>
      </div>
    </div>
  )
}

function BackendStatus() {
  const { data, error, loading } = useHealth()
  if (loading && !data) return <div className="text-xs text-fg-subtle">Connecting (may be cold starting)…</div>
  if (error && !data)
    return (
      <Badge tone="bad" dot>
        API offline
      </Badge>
    )
  if (!data) return null
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Badge tone={error ? 'warn' : 'ok'} dot pulse={!error}>
          {error ? 'API unstable' : 'API online'}
        </Badge>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {data.ml_mock && <Badge tone="warn">ML mock</Badge>}
        {data.llm_mock && <Badge tone="warn">LLM mock</Badge>}
      </div>
      {!data.llm_mock && data.llm_model && (
        <div className="truncate font-mono text-[11px] text-fg-subtle" title={data.llm_model}>
          {data.llm_model}
        </div>
      )}
    </div>
  )
}

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="space-y-0.5">
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          onClick={onNavigate}
          className={({ isActive }) =>
            cn(
              'flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors',
              isActive ? 'bg-surface-3 text-fg' : 'text-fg-muted hover:bg-surface-2 hover:text-fg',
            )
          }
        >
          <Icon className="size-4" />
          {label}
        </NavLink>
      ))}
      <button
        onClick={() => {
          onNavigate?.()
          openWelcome()
        }}
        className="flex w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm text-fg-muted transition-colors hover:bg-surface-2 hover:text-fg"
      >
        <Info className="size-4" />
        About this demo
      </button>
    </nav>
  )
}

/** The hosted backend scales to zero when idle (Modal): the first request after a pause waits for a container
 * to start. After a short grace period with no /health answer, say so instead of looking broken. */
const COLD_START_AFTER_MS = 2500

function useWaking(hasData: boolean): boolean {
  const [slow, setSlow] = useState(false)
  useEffect(() => {
    if (hasData) {
      setSlow(false)
      return
    }
    const t = setTimeout(() => setSlow(true), COLD_START_AFTER_MS)
    return () => clearTimeout(t)
  }, [hasData])
  return slow && !hasData
}

export function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()
  const health = useHealth()
  const waking = useWaking(!!health.data)
  const offline = !!health.error?.unreachable && !health.loading && !waking

  useEffect(() => setMobileOpen(false), [location.pathname])

  return (
    <div className="min-h-dvh lg:grid lg:grid-cols-[232px_1fr]">
      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-dvh flex-col border-r border-line bg-surface/40 px-3 py-4 lg:flex">
        <div className="px-1.5">
          <Logo />
        </div>
        <div className="mt-6">
          <NavItems />
        </div>
        <div className="mt-auto rounded-lg border border-line bg-surface p-3">
          <div className="mb-2 text-[11px] font-medium tracking-wide text-fg-subtle uppercase">Backend</div>
          <BackendStatus />
        </div>
      </aside>

      {/* Mobile top bar */}
      <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-line bg-canvas/90 px-4 backdrop-blur lg:hidden">
        <Logo />
        <button
          className="rounded-md p-2 text-fg-muted hover:bg-surface-3 hover:text-fg"
          onClick={() => setMobileOpen((o) => !o)}
          aria-label="Menu"
        >
          {mobileOpen ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </header>
      {mobileOpen && (
        <div className="fixed inset-x-0 top-14 z-20 border-b border-line bg-surface p-3 shadow-2xl lg:hidden">
          <NavItems onNavigate={() => setMobileOpen(false)} />
          <div className="mt-3 border-t border-line pt-3">
            <BackendStatus />
          </div>
        </div>
      )}

      <WelcomeDialog />
      <main className="min-w-0">
        {waking && (
          <div
            role="status"
            className="flex items-center gap-2 border-b border-accent/25 bg-accent-soft px-4 py-2 text-xs text-accent lg:px-8"
          >
            <Loader2 className="size-3.5 shrink-0 animate-spin" />
            <span>
              Waking up the backend. It scales to zero when nobody is using it, so the first request after a pause starts
              a fresh container (usually under a minute). Recorded runs and deployments reload automatically.
            </span>
          </div>
        )}
        {offline && (
          <div className="border-b border-warn/25 bg-warn-soft px-4 py-2 text-xs text-warn lg:px-8">
            The Assembly API is unreachable. Data on screen may be stale; reconnecting automatically.
          </div>
        )}
        <Outlet />
      </main>
    </div>
  )
}

export function PageHeader({
  title,
  description,
  actions,
  eyebrow,
}: {
  title: React.ReactNode
  description?: React.ReactNode
  actions?: React.ReactNode
  eyebrow?: React.ReactNode
}) {
  return (
    <div className="flex flex-col gap-4 border-b border-line px-4 py-5 sm:flex-row sm:items-end sm:justify-between lg:px-8 lg:py-6">
      <div className="min-w-0">
        {eyebrow && <div className="mb-1.5 text-xs text-fg-subtle">{eyebrow}</div>}
        <h1 className="text-xl font-semibold tracking-tight text-fg">{title}</h1>
        {description && <p className="mt-1 max-w-2xl text-sm text-fg-muted">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

export function PageBody({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn('px-4 py-6 lg:px-8', className)}>{children}</div>
}
