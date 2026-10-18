import { lazy, StrictMode, Suspense, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router'
import './index.css'
import { Layout } from './components/Layout'
import { HealthProvider } from './hooks/health'
import { OverviewPage } from './pages/Overview'
import { NewRunPage } from './pages/NewRun'
import { NotFoundPage } from './pages/NotFound'
import { Spinner } from './components/ui/primitives'

// Chart-heavy pages load on demand (keeps recharts out of the initial bundle).
const RunPage = lazy(() => import('./pages/Run').then((m) => ({ default: m.RunPage })))
const DeploymentsPage = lazy(() => import('./pages/Deployments').then((m) => ({ default: m.DeploymentsPage })))
const DataSourcesPage = lazy(() => import('./pages/DataSources').then((m) => ({ default: m.DataSourcesPage })))

const page = (node: ReactNode) => (
  <Suspense
    fallback={
      <div className="flex justify-center py-32">
        <Spinner />
      </div>
    }
  >
    {node}
  </Suspense>
)
const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: '/', element: <OverviewPage /> },
      { path: '/runs/new', element: <NewRunPage /> },
      { path: '/runs/:runId', element: page(<RunPage />) },
      { path: '/deployments', element: page(<DeploymentsPage />) },
      { path: '/deployments/:depId', element: page(<DeploymentsPage />) },
      { path: '/data', element: page(<DataSourcesPage />) },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
])

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HealthProvider>
      <RouterProvider router={router} />
    </HealthProvider>
  </StrictMode>,
)
