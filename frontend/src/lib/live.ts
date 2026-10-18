import type { LiveTick, Outcome } from '@/api/types'

export type OutcomeTally = Record<Outcome, number> & { total: number }

/** Count alarm outcomes for one task over the live ticks (real deployments only; simulator ticks have none). */
export function tallyOutcomes(ticks: LiveTick[], taskId: string): OutcomeTally {
  const t: OutcomeTally = { caught: 0, missed: 0, false_alarm: 0, ok: 0, total: 0 }
  for (const tick of ticks) {
    const o = tick.predictions[taskId]?.outcome
    if (!o || !(o in t)) continue
    t[o] += 1
    t.total += 1
  }
  return t
}

/** Live recall / false-alarm rate from a tally (undefined when there is nothing to divide by yet). */
export function liveRates(t: OutcomeTally): { recall?: number; falseAlarmRate?: number } {
  const positives = t.caught + t.missed
  const negatives = t.false_alarm + t.ok
  return {
    recall: positives ? t.caught / positives : undefined,
    falseAlarmRate: negatives ? t.false_alarm / negatives : undefined,
  }
}

/** True if the feed replays held-out test samples (real deployment) rather than the plant simulator. */
export function isReplayFeed(ticks: LiveTick[]): boolean {
  return ticks.some((t) => Object.values(t.predictions).some((p) => p?.sample_id !== undefined))
}
