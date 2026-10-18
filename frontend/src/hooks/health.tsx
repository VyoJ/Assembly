import { createContext, useContext, type ReactNode } from 'react'
import { api } from '@/api/client'
import type { Health } from '@/api/types'
import { useApi, type ApiState } from './useApi'

const HealthContext = createContext<ApiState<Health> | null>(null)

export function HealthProvider({ children }: { children: ReactNode }) {
  const health = useApi(api.health, ['health'], { intervalMs: 10_000 })
  return <HealthContext.Provider value={health}>{children}</HealthContext.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useHealth(): ApiState<Health> {
  const ctx = useContext(HealthContext)
  if (!ctx) throw new Error('useHealth outside HealthProvider')
  return ctx
}
