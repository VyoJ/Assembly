// Minimal SSE client over fetch. Unlike EventSource it delivers every named event (the backend uses the
// event type as the SSE `event` name, so EventSource would need a listener per type), surfaces HTTP status
// codes (a 404 means the run is gone, e.g. after a backend restart) and lets us control reconnects.

export interface SseMessage {
  event: string
  data: string
}

/** Incremental parser for the text/event-stream format. Feed it chunks; it returns complete messages. */
export class SseParser {
  private buffer = ''
  private event = ''
  private data: string[] = []

  push(chunk: string): SseMessage[] {
    this.buffer += chunk
    const out: SseMessage[] = []
    let nl: number
    while ((nl = this.buffer.search(/\r\n|\r|\n/)) !== -1) {
      const line = this.buffer.slice(0, nl)
      const sepLen = this.buffer.startsWith('\r\n', nl) ? 2 : 1
      this.buffer = this.buffer.slice(nl + sepLen)
      if (line === '') {
        if (this.data.length) out.push({ event: this.event || 'message', data: this.data.join('\n') })
        this.event = ''
        this.data = []
        continue
      }
      if (line.startsWith(':')) continue // comment / keep-alive ping
      const colon = line.indexOf(':')
      const field = colon === -1 ? line : line.slice(0, colon)
      let value = colon === -1 ? '' : line.slice(colon + 1)
      if (value.startsWith(' ')) value = value.slice(1)
      if (field === 'event') this.event = value
      else if (field === 'data') this.data.push(value)
    }
    return out
  }
}

export type SseStatus = 'connecting' | 'open' | 'reconnecting' | 'closed' | 'error'

export interface SseOptions {
  onOpen?: () => void
  onMessage: (msg: SseMessage) => void
  onStatus?: (status: SseStatus, detail?: string) => void
  /** Return false to stop reconnecting after the stream ends cleanly (e.g. the run is finished). */
  shouldReconnect?: () => boolean
  maxBackoffMs?: number
}

/** Connects and keeps reconnecting with backoff until stopped. Returns a stop function. */
export function connectSse(url: string, opts: SseOptions): () => void {
  const controller = new AbortController()
  let stopped = false
  let attempt = 0
  let timer: ReturnType<typeof setTimeout> | undefined
  const maxBackoff = opts.maxBackoffMs ?? 10_000

  const schedule = (detail?: string) => {
    if (stopped) return
    if (opts.shouldReconnect && !opts.shouldReconnect()) {
      opts.onStatus?.('closed', detail)
      return
    }
    attempt += 1
    const delay = Math.min(maxBackoff, 500 * 2 ** Math.min(attempt, 5))
    opts.onStatus?.('reconnecting', detail)
    timer = setTimeout(run, delay)
  }

  async function run() {
    if (stopped) return
    opts.onStatus?.(attempt === 0 ? 'connecting' : 'reconnecting')
    let res: Response
    try {
      res = await fetch(url, { headers: { Accept: 'text/event-stream' }, signal: controller.signal, cache: 'no-store' })
    } catch {
      if (!stopped) schedule('Backend unreachable')
      return
    }
    if (res.status === 404) {
      opts.onStatus?.('error', 'Not found on the backend (it may have restarted; state is in memory).')
      return
    }
    if (!res.ok || !res.body) {
      schedule(`HTTP ${res.status}`)
      return
    }
    attempt = 0
    opts.onStatus?.('open')
    opts.onOpen?.()
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
    const parser = new SseParser()
    try {
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        for (const msg of parser.push(value)) opts.onMessage(msg)
      }
    } catch {
      /* aborted or network drop */
    }
    if (!stopped) schedule('Stream ended')
  }

  run()
  return () => {
    stopped = true
    clearTimeout(timer)
    controller.abort()
    opts.onStatus?.('closed')
  }
}

/**
 * Merge one event from a (re)connected stream that replays full history from the start. `cursor` is how many
 * events this connection has delivered so far. Already-known events are skipped by position (checked by
 * ts + type); on a mismatch the local log is truncated to what the server agrees with. Mutates `known`.
 * Returns the new cursor and whether `ev` was appended.
 */
export function mergeReplayed<E extends { ts: number; type: string }>(
  known: E[],
  cursor: number,
  ev: E,
): { cursor: number; appended: boolean } {
  if (cursor < known.length) {
    if (known[cursor].ts === ev.ts && known[cursor].type === ev.type) return { cursor: cursor + 1, appended: false }
    known.length = cursor
  }
  known.push(ev)
  return { cursor: cursor + 1, appended: true }
}
