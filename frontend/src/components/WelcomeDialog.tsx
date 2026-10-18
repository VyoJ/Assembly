import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { Camera, ChevronLeft, ChevronRight, Cpu, Database, ExternalLink, Heart, History, Play, Rocket, ShieldCheck, Waves, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from './ui/primitives'

/** Shown once per browser on first visit; reopened from the sidebar ("About this demo"). */
const SEEN_KEY = 'assembly.welcome.v1'
const OPEN_EVENT = 'assembly:welcome'
const DEMO_VIDEO = '/Assembly-Demo.mp4'

// eslint-disable-next-line react-refresh/only-export-components
export function openWelcome() {
  window.dispatchEvent(new Event(OPEN_EVENT))
}

function seen(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) === '1'
  } catch {
    return false // private mode / blocked storage: just show it
  }
}

function markSeen() {
  try {
    localStorage.setItem(SEEN_KEY, '1')
  } catch {
    /* not persisted: it shows again next visit, which is fine */
  }
}

const DATASETS: { icon: typeof Database; name: string; href: string; what: ReactNode }[] = [
  {
    icon: Cpu,
    name: 'NASA C-MAPSS turbofan degradation',
    href: 'https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data',
    what: <>Sensors of 100 engines run to failure, as <b>Press 3</b>, plus a maintenance log of when each failed.</>,
  },
  {
    icon: Waves,
    name: 'CWRU Bearing Data Center',
    href: 'https://engineering.case.edu/bearingdatacenter',
    what: <>Vibration recordings of healthy and faulty motor bearings, as <b>Motor 4</b>.</>,
  },
  {
    icon: Camera,
    name: 'MVTec AD: metal nut',
    href: 'https://www.mvtec.com/company/research/datasets/mvtec-ad',
    what: <>Images of good and defective metal nuts with defect masks, as the <b>Line 3 camera</b> and its QC inspections.</>,
  },
]

type Step = { eyebrow: string; title: string; body: ReactNode }

export function WelcomeDialog() {
  const [open, setOpen] = useState(() => !seen())
  const [step, setStep] = useState(0)
  const [video, setVideo] = useState(false)
  const navigate = useNavigate()
  const closeRef = useRef<HTMLButtonElement>(null)

  const close = () => {
    markSeen()
    setOpen(false)
    setVideo(false)
    setStep(0)
  }
  const go = (to: string) => {
    close()
    navigate(to)
  }

  const steps: Step[] = [
    {
      eyebrow: 'Live demo',
      title: 'Welcome to Assembly',
      body: (
        <>
          <p className="text-base leading-relaxed text-fg">
            An autonomous ML engineer for factories. Give it a goal in plain language; it finds the right plant data, builds the dataset,
            trains and tests models, and deploys the ones that meet your targets.
          </p>
          {video ? (
            <video src={DEMO_VIDEO} controls autoPlay playsInline preload="metadata" className="w-full rounded-lg border border-line bg-black" />
          ) : (
            <Button variant="primary" size="lg" onClick={() => setVideo(true)}>
              <Play className="size-4" /> Watch the demo video
            </Button>
          )}
        </>
      ),
    },
    {
      eyebrow: '1 · The data',
      title: 'Real data sources, already connected',
      body: (
        <>
          <p>Three public industrial datasets make up the plant:</p>
          <ul className="space-y-2">
            {DATASETS.map(({ icon: Icon, name, href, what }) => (
              <li key={name} className="flex gap-2.5 rounded-lg border border-line bg-surface-2 p-3">
                <Icon className="mt-0.5 size-4 shrink-0 text-teal" />
                <div>
                  <a href={href} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-fg underline-offset-2 hover:underline">
                    {name} <ExternalLink className="size-3" />
                  </a>
                  <div className="mt-0.5 text-xs leading-relaxed">{what}</div>
                </div>
              </li>
            ))}
          </ul>
          <p className="text-xs">Plus three generated distractor streams, to see whether the agent ignores what's irrelevant.</p>
        </>
      ),
    },
    {
      eyebrow: '2 · Explore',
      title: 'See what it built',
      body: (
        <>
          <p className="flex gap-2.5">
            <History className="mt-1 size-4 shrink-0 text-teal" />
            <span>
              <b className="text-fg">Overview:</b> full runs we did with a real LLM. See how it framed the goal, explored the data, wrote and fixed its
              own code, and which models passed the held-out test.
            </span>
          </p>
          <p className="flex gap-2.5">
            <Rocket className="mt-1 size-4 shrink-0 text-teal" />
            <span>
              <b className="text-fg">Deployments:</b> run the trained models live on test samples they never saw, including the camera model drawing
              defect boxes.
            </span>
          </p>
        </>
      ),
    },
    {
      eyebrow: '3 · Your plant',
      title: 'Bring your own data',
      body: (
        <>
          <p className="flex gap-2.5">
            <Database className="mt-1 size-4 shrink-0 text-teal" />
            <span>
              Adding a source is <b className="text-fg">one config entry</b>: historian or SCADA exports, sensor recordings, camera images or video.
              The logs a plant already keeps (maintenance, QC inspections) become the labels. No hand-built training sets.
            </span>
          </p>
          <p className="flex gap-2.5">
            <ShieldCheck className="mt-1 size-4 shrink-0 text-teal" />
            <span>
              Assembly can be <b className="text-fg">self-hosted</b>, so no data has to leave the factory's systems.
            </span>
          </p>
        </>
      ),
    },
    {
      eyebrow: '4 · Thanks',
      title: 'Built on AMD',
      body: (
        <>
          <p className="flex gap-2.5">
            <Heart className="mt-1 size-4 shrink-0 text-bad" />
            <span>
              Thank you to <b className="text-fg">AMD</b> for the <b className="text-fg">Token Factory</b>: every decision and line of code the agent
              writes comes from LLMs served on AMD GPUs.
            </span>
          </p>
          <p className="flex gap-2.5">
            <Cpu className="mt-1 size-4 shrink-0 text-fg-subtle" />
            <span>
              This demo runs entirely on free resources, so the backend sleeps when idle: the first load can take a little while, and a new run
              takes 30 to 45 minutes.
            </span>
          </p>
        </>
      ),
    },
  ]
  const last = step === steps.length - 1
  const current = steps[step]

  useEffect(() => {
    const reopen = () => setOpen(true)
    window.addEventListener(OPEN_EVENT, reopen)
    return () => window.removeEventListener(OPEN_EVENT, reopen)
  }, [])

  useEffect(() => {
    if (!open) return
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
      if (e.key === 'ArrowRight') setStep((i) => Math.min(i + 1, steps.length - 1))
      if (e.key === 'ArrowLeft') setStep((i) => Math.max(i - 1, 0))
    }
    document.addEventListener('keydown', onKey)
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = overflow
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  useEffect(() => setVideo(false), [step]) // leaving the first page stops the video

  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="welcome-title"
        className="flex max-h-[92dvh] w-full max-w-xl flex-col overflow-hidden rounded-xl border border-line-strong bg-surface shadow-2xl sm:h-[520px]"
      >
        <div className="flex items-start justify-between gap-4 px-6 pt-5">
          <div>
            <div className="text-[11px] font-medium tracking-wide text-teal uppercase">{current.eyebrow}</div>
            <h2 id="welcome-title" className="mt-1 text-xl font-semibold tracking-tight text-fg">
              {current.title}
            </h2>
          </div>
          <button ref={closeRef} onClick={close} aria-label="Close" className="rounded-md p-1.5 text-fg-subtle hover:bg-surface-3 hover:text-fg">
            <X className="size-4" />
          </button>
        </div>

        {/* fixed panel height on wider screens, so Back / Next don't move between pages */}
        <div className="min-h-[220px] flex-1 space-y-4 overflow-y-auto px-6 py-5 text-sm leading-relaxed text-fg-muted">{current.body}</div>

        <div className="flex items-center justify-between gap-3 border-t border-line px-6 py-3">
          <div className="flex items-center gap-1.5" aria-label={`Page ${step + 1} of ${steps.length}`}>
            {steps.map((s, i) => (
              <button
                key={s.title}
                onClick={() => setStep(i)}
                aria-label={`Go to page ${i + 1}: ${s.title}`}
                className={cn('h-1.5 rounded-full transition-all', i === step ? 'w-5 bg-accent' : 'w-1.5 bg-line-strong hover:bg-fg-subtle')}
              />
            ))}
          </div>
          <div className="flex items-center gap-2">
            {step === 0 ? (
              <Button variant="ghost" onClick={close}>
                Skip
              </Button>
            ) : (
              <Button variant="ghost" onClick={() => setStep(step - 1)}>
                <ChevronLeft className="size-3.5" /> Back
              </Button>
            )}
            {last ? (
              <>
                <Button variant="secondary" onClick={() => go('/deployments')}>
                  <Rocket className="size-3.5" /> Deployments
                </Button>
                <Button variant="primary" onClick={() => go('/')}>
                  Explore the runs
                </Button>
              </>
            ) : (
              <Button variant="primary" onClick={() => setStep(step + 1)}>
                Next <ChevronRight className="size-3.5" />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
