import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '@/api/client'

export interface ApiState<T> {
  data: T | undefined
  error: ApiError | undefined
  loading: boolean
  /** Refetch now; resolves when done. */
  refresh: () => Promise<void>
}

/**
 * Fetch on mount / when `key` changes, optionally polling every `intervalMs` while `poll` is true.
 * Keeps the last good data on a failed refresh so the UI doesn't flash empty during a blip.
 */
export function useApi<T>(
  fetcher: (() => Promise<T>) | null,
  key: unknown[],
  { intervalMs, poll = true }: { intervalMs?: number; poll?: boolean } = {},
): ApiState<T> {
  const [data, setData] = useState<T>()
  const [error, setError] = useState<ApiError>()
  const [loading, setLoading] = useState(!!fetcher)
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher
  const seq = useRef(0)

  const refresh = useCallback(async () => {
    const f = fetcherRef.current
    if (!f) return
    const mine = ++seq.current
    try {
      const value = await f()
      if (mine !== seq.current) return
      setData(value)
      setError(undefined)
    } catch (e) {
      if (mine !== seq.current) return
      setError(e instanceof ApiError ? e : new ApiError(0, String(e)))
    } finally {
      if (mine === seq.current) setLoading(false)
    }
  }, [])

  const keyStr = JSON.stringify(key)
  useEffect(() => {
    setData(undefined)
    setError(undefined)
    setLoading(!!fetcherRef.current)
    void refresh()
    const counter = seq
    return () => {
      counter.current++ // drop in-flight responses for the old key
    }
  }, [keyStr, refresh])

  useEffect(() => {
    if (!intervalMs || !poll) return
    const id = setInterval(() => {
      if (document.visibilityState === 'visible') void refresh()
    }, intervalMs)
    return () => clearInterval(id)
  }, [intervalMs, poll, refresh, keyStr])

  return { data, error, loading, refresh }
}
