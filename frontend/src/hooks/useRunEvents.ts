import { useEffect, useRef, useState } from 'react'
import { api, url } from '@/api/client'
import type { AnyEvent } from '@/api/events'
import { connectSse, mergeReplayed, type SseStatus } from '@/api/sse'

export interface RunEventsState {
  events: AnyEvent[]
  status: SseStatus
  detail?: string
  /** Bumps whenever new events land; cheap dependency for effects. */
  version: number
}

const FINAL_TYPES = new Set(['run_finished', 'run_failed'])

/**
 * Subscribes to GET /api/runs/{id}/events. The backend replays the whole history on every connection and
 * events carry no id, so after a reconnect the first N events are the ones we already have: they are
 * skipped by position (and checked by timestamp; a mismatch resets the log). Updates are batched per frame
 * because the replay can deliver hundreds of events at once.
 */
export function useRunEvents(runId: string | undefined): RunEventsState {
  const [state, setState] = useState<RunEventsState>({ events: [], status: 'connecting', version: 0 })
  const eventsRef = useRef<AnyEvent[]>([])

  useEffect(() => {
    if (!runId) return
    eventsRef.current = []
    setState({ events: [], status: 'connecting', version: 0 })

    let cursor = 0 // position within the current connection's stream
    let finished = false
    let frame = 0
    const flush = () => {
      frame = 0
      setState((s) => ({ ...s, events: eventsRef.current.slice(), version: s.version + 1 }))
    }
    const scheduleFlush = () => {
      if (!frame) frame = requestAnimationFrame(flush)
    }

    const stop = connectSse(url(api.runEventsPath(runId)), {
      onOpen: () => {
        cursor = 0
      },
      onMessage: ({ data }) => {
        let ev: AnyEvent
        try {
          ev = JSON.parse(data)
        } catch {
          return
        }
        const merged = mergeReplayed(eventsRef.current, cursor, ev)
        cursor = merged.cursor
        if (!merged.appended) return // replayed duplicate
        if (FINAL_TYPES.has(ev.type)) finished = true
        scheduleFlush()
        // Nothing follows run_finished / run_failed; release the connection (browsers cap them per origin).
        if (finished) queueMicrotask(stop)
      },
      onStatus: (status, detail) =>
        setState((s) => ({ ...s, status: finished && status === 'closed' ? 'closed' : status, detail })),
      shouldReconnect: () => !finished,
    })

    return () => {
      cancelAnimationFrame(frame)
      stop()
    }
  }, [runId])

  return state
}
