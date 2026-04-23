/// <reference types="node" />
import { defineConfig } from "vitest/config";

// Dev server proxies /api to the FastAPI layer shipped by
// ``booth_retriever.web`` (default port 8000 for ``uvicorn ... --reload``).
// In production the static build is meant to be served alongside the API by
// any HTTP server, so the /api prefix is preserved at build time as well.
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
  },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["tests/**/*.test.ts", "src/**/*.test.ts"],
  },
});
