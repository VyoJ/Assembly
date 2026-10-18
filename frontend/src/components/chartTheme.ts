// Chart colours, matching the tokens in index.css (Recharts needs literal values).
export const CHART = {
  grid: '#23272d',
  axis: '#6b727d',
  line: '#3b9eff',
  teal: '#2dd4bf',
  ok: '#3fb950',
  bad: '#f85149',
  warn: '#e3a008',
  neutral: '#6b727d',
  surface: '#111316',
}

export const axisProps = {
  stroke: CHART.axis,
  tick: { fill: CHART.axis, fontSize: 11 },
  tickLine: false,
  axisLine: { stroke: CHART.grid },
} as const
