/// <reference types="node" />
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// Dev server proxies /api to the FastAPI layer shipped by
// ``booth_retriever.web`` (default port 8000 for ``uvicorn ... --reload``).
// In production the static build is meant to be served alongside the API by
// any HTTP server, so the /api prefix is preserved at build time as well.

// Multi-page build: the UI is split across two top-level HTML entries,
// ``index.html`` (curate) and ``ask.html``. Vite's dev server picks these
// up automatically, but the production build needs them listed explicitly
// in ``rollupOptions.input`` or only ``index.html`` ends up in ``dist/``.
const here = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.BOOTH_API_URL ?? "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    target: "es2022",
    rollupOptions: {
      input: {
        main: resolve(here, "index.html"),
        ask: resolve(here, "ask.html"),
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["tests/**/*.test.ts", "src/**/*.test.ts"],
  },
});
