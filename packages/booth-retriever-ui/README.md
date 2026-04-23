# booth-retriever-ui

Static web UI for the [`booth-retriever`](../booth-retriever) curation
workflow. Plain HTML + CSS + TypeScript, built with Vite. No React / Vue /
Svelte — the page is simple enough that a framework is noise.

The finished build is a single-page app meant to be pointed at by any
dashboard host that can render an iframe (NeoDash, Grafana text panel,
internal admin portal, …). It never talks to Neo4j directly — all mutations
go through the FastAPI layer in
[`booth_retriever.web`](../booth-retriever/src/booth_retriever/web/api.py).

## Features (MVP)

- Stats tiles (total + counts per status).
- Pending queries list with status filter.
- Detail panel with Cypher textarea, parameters input, and
  Approve / Save edits / Reject buttons.
- Inline surfacing of the server's `422` verifier error next to the
  textarea, so the curator can fix it and retry without leaving the page.
- Helpful / Not helpful feedback buttons on queries that already have a
  FewShot.

Not in MVP (see the plan under `.cursor/plans/`):

- CodeMirror syntax highlighting.
- Ask / Test pages (maps to `BOOTHRetriever.query`).
- Auth.

## Install

```bash
cd packages/booth-retriever-ui
npm install
```

## Dev workflow

Start the FastAPI backend in one terminal:

```bash
# from the repo root, with your Python venv active:
pip install -e "packages/booth-retriever[web]"
uvicorn booth_retriever.web:app --reload    # -> http://localhost:8000
```

…and the Vite dev server in another:

```bash
cd packages/booth-retriever-ui
npm run dev                                  # -> http://localhost:5173
```

`vite.config.ts` proxies `/api/*` from the dev server to `http://localhost:8000`,
so the browser never deals with CORS in development. Override the proxy
target by setting `BOOTH_API_URL` before `npm run dev`, e.g.:

```bash
BOOTH_API_URL=http://staging.example.com npm run dev
```

## Production build

```bash
npm run build        # type-checks, then emits static assets to ./dist
npm run preview      # local smoke-test of the built bundle
```

`dist/` contains plain `index.html`, a CSS file and a single JS chunk. You
can serve it from anything that serves static files — a CDN, an nginx
sidecar, or FastAPI itself:

```python
from fastapi.staticfiles import StaticFiles
from booth_retriever.web import app

app.mount(
    "/",
    StaticFiles(directory="packages/booth-retriever-ui/dist", html=True),
    name="ui",
)
```

At that point a single `uvicorn booth_retriever.web:app` serves both the API
and the curator UI.

## Tests

Vitest unit tests live under `tests/` and exercise the `fetch` wrappers in
`src/api.ts` against a stubbed `globalThis.fetch` — both happy paths and
the 404 / 422 / 500 error shapes.

```bash
npm test            # one-shot run
npm run test:watch  # vitest in watch mode
```

Type-checking only (no bundling):

```bash
npm run typecheck
```

## Layout

```
packages/booth-retriever-ui/
  index.html        # single-page shell
  package.json
  tsconfig.json
  vite.config.ts    # /api dev proxy + vitest config
  public/
    brand/          # Neo4j brand assets (copied from neo4j-branding/)
      css/          # palettes.css, fonts.css, theme-dark.css
      typography/fonts/   # Public Sans + SyneNeo WOFF2 subset
      logos/        # official SVG logos (do not edit)
  src/
    main.ts         # bootstraps DOM wiring
    api.ts          # typed fetch wrappers, one per route
    render.ts       # pure DOM render functions
    types.ts        # mirrors of the Pydantic schemas
    styles.css      # aliases Neo4j brand tokens onto app-scoped vars
  tests/
    api.test.ts     # Vitest suite for api.ts
```

## Branding

The UI conforms to the Neo4j brand system:

- `public/brand/css/` contains the canonical palette, font, and dark-theme
  token stylesheets from the `neo4j-branding/` package. They are linked
  directly from `index.html`; `src/styles.css` only aliases the tokens,
  never hardcodes hex values.
- Typography is Public Sans (body) and SyneNeo (display / app title only),
  loaded as WOFF2 from `public/brand/typography/fonts/`.
- The header shows the official Neo4j wordmark (white variant, per
  `LOGOS.md` — white logos on dark backgrounds). The logo is never
  recoloured, stretched, or cropped.
- The favicon uses the full-colour Neo4j monogram.
- Dark theme is forced via `data-theme="dark"` on `<html>` so the page
  renders identically inside a NeoDash iframe regardless of the host's
  colour-scheme preference.

To update brand assets, re-copy from the source `neo4j-branding/` package
into `public/brand/`. **Never edit the files under `public/brand/` directly.**

## Embedding in a dashboard

The page is self-contained and has no routing. Host it on any URL reachable
by the dashboard, then embed:

```html
<iframe
  src="https://booth-curator.example.com/"
  style="width:100%;height:100vh;border:0"
></iframe>
```

The only runtime configuration is where `/api` points; in embedded
deployments it is usually easiest to serve the built assets from the same
origin as FastAPI (see the `StaticFiles` recipe above) so the iframe needs
no cross-origin configuration.
