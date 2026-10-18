import { defineConfig, loadEnv, type ProxyOptions } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath, URL } from 'node:url'

// The dev server proxies /api and /health to the FastAPI backend so the browser stays same-origin
// (no CORS, and SSE streams pass straight through). Override the target with ASSEMBLY_API_TARGET.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.ASSEMBLY_API_TARGET || 'http://localhost:8000'
  const proxy: ProxyOptions = {
    target,
    changeOrigin: true,
    // Vite answers 500 when the backend is down; send 502 so the UI can tell "unreachable" from a server error.
    configure: (p) =>
      p.on('error', (_err, _req, res) => {
        if ('writeHead' in res && !res.headersSent) {
          res.writeHead(502, { 'Content-Type': 'application/json' })
          res.end(JSON.stringify({ detail: `Assembly API unreachable at ${target}` }))
        }
      }),
  }
  return {
    plugins: [react(), tailwindcss()],
    resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
    server: { port: 5173, proxy: { '/api': proxy, '/health': proxy } },
    preview: { port: 4173, proxy: { '/api': proxy, '/health': proxy } },
    test: { environment: 'node' },
  }
})
