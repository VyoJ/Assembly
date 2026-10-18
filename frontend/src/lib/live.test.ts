import { describe, expect, it } from 'vitest'
import type { LiveTick } from '@/api/types'
import { isReplayFeed, liveRates, tallyOutcomes } from './live'

const tick = (preds: LiveTick['predictions']): LiveTick => ({ t: 0, cycle: 1, predictions: preds, kpi_tracker: {} })

describe('live feed', () => {
  const ticks = [
    tick({ cam: { alarm: true, outcome: 'caught', sample_id: 'test-0' } }),
    tick({ cam: { alarm: false, outcome: 'missed', sample_id: 'test-1' } }),
    tick({ cam: { alarm: true, outcome: 'false_alarm', sample_id: 'test-2' } }),
    tick({ cam: { alarm: false, outcome: 'ok', sample_id: 'test-3' } }),
    tick({ cam: { alarm: false, outcome: 'ok', sample_id: 'test-4' } }),
  ]

  it('tallies outcomes per task and derives live rates', () => {
    const t = tallyOutcomes(ticks, 'cam')
    expect(t).toEqual({ caught: 1, missed: 1, false_alarm: 1, ok: 2, total: 5 })
    expect(liveRates(t)).toEqual({ recall: 0.5, falseAlarmRate: 1 / 3 })
    expect(liveRates(tallyOutcomes([], 'cam'))).toEqual({ recall: undefined, falseAlarmRate: undefined })
  })

  it('tells replayed test samples from simulator ticks', () => {
    expect(isReplayFeed(ticks)).toBe(true)
    expect(isReplayFeed([tick({ press: { value: 80, alarm: false } })])).toBe(false)
  })
})
