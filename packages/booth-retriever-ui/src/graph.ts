// Neo4j Visualization Library (NVL) helpers, framework-free.
//
// Exposes a single `mountGraph` that instantiates an NVL graph into a sized
// container and auto-fits it, plus a `GraphPanel` slide-over used by the Ask
// page. The Curate page mounts the same graph inline for "process" queries.
//
// Layout note: we use the `d3Force` layout (main-thread d3-force) rather than
// the default `forceDirected` (CoseBilkent web-worker) layout. The latter is
// deprecated for small datasets and, when its worker fails to load, leaves
// every node stacked at the origin — the graph then looks like a single node
// until you drag it apart. d3Force positions nodes reliably without a worker.

import { NVL, d3ForceLayoutType } from "@neo4j-nvl/base";
import type { Node as NvlNode, Relationship as NvlRel } from "@neo4j-nvl/base";
import {
  DragNodeInteraction,
  PanInteraction,
  ZoomInteraction,
} from "@neo4j-nvl/interaction-handlers";
import type { GraphPayload } from "./types.js";

// Caption -> brand-ish colour. Falls back to a neutral grey for labels we
// don't recognise. Picked from the Neo4j palette family used elsewhere.
const LABEL_COLORS: Record<string, string> = {
  UserQuestion: "#68bdf6",
  Query: "#6dce9e",
  FewShot: "#ff756e",
  CypherAttempt: "#de9bf9",
  Response: "#fb95af",
  Step: "#f79767",
  Agent: "#ffc454",
  Tool: "#569480",
  DataProduct: "#c990c0",
  System: "#4c8eda",
};

function nodeColor(labels: string[]): string {
  for (const label of labels) {
    if (LABEL_COLORS[label]) return LABEL_COLORS[label];
  }
  return "#a5abb6";
}

export interface GraphHandle {
  destroy(): void;
}

/**
 * Instantiate an NVL graph into `frame` and auto-fit it once the layout
 * settles. Returns a handle whose `destroy()` tears the instance down.
 *
 * `frame` must already be visible and have a non-zero size.
 */
export function mountGraph(frame: HTMLElement, payload: GraphPayload): GraphHandle {
  const nodes: NvlNode[] = payload.nodes.map((n) => ({
    id: n.id,
    caption: n.caption,
    color: nodeColor(n.labels),
  }));
  const rels: NvlRel[] = payload.relationships.map((r) => ({
    id: r.id,
    from: r.from,
    to: r.to,
    caption: r.caption,
  }));

  const nodeIds = nodes.map((n) => n.id);
  let nvl: NVL | null = null;

  const fit = (): void => {
    if (!nvl || nodeIds.length === 0) return;
    nvl.fit(nodeIds);
  };

  nvl = new NVL(
    frame,
    nodes,
    rels,
    { layout: d3ForceLayoutType, initialZoom: 0.75 },
    { onLayoutDone: fit },
  );

  // `@neo4j-nvl/base` is render-only — wheel-zoom, background-drag pan, and
  // node dragging come from the interaction-handlers companion. Without these
  // the graph is a static image you can't navigate.
  const interactions = [
    new ZoomInteraction(nvl),
    new PanInteraction(nvl),
    new DragNodeInteraction(nvl),
  ];

  // Belt-and-braces: re-fit a couple of times in case onLayoutDone fired
  // before the container had its final size (e.g. just-revealed panel).
  const timers = [
    window.setTimeout(fit, 150),
    window.setTimeout(fit, 600),
  ];

  return {
    destroy(): void {
      for (const t of timers) window.clearTimeout(t);
      for (const interaction of interactions) interaction.destroy();
      if (nvl) {
        nvl.destroy();
        nvl = null;
      }
    },
  };
}

interface PanelElements {
  panel: HTMLElement;
  frame: HTMLElement;
  empty: HTMLElement;
  closeBtn: HTMLButtonElement;
}

/** Right-side slide-over graph used on the Ask page. */
export class GraphPanel {
  private handle: GraphHandle | null = null;
  private readonly els: PanelElements;

  constructor(els: PanelElements) {
    this.els = els;
    this.els.closeBtn.addEventListener("click", () => this.close());
  }

  show(payload: GraphPayload): void {
    this.els.panel.classList.remove("hidden");
    this.els.panel.setAttribute("aria-hidden", "false");

    this.destroyGraph();
    if (payload.nodes.length === 0) {
      this.els.empty.classList.remove("hidden");
      return;
    }
    this.els.empty.classList.add("hidden");
    // Mount after the reveal so the frame has its final size.
    requestAnimationFrame(() => {
      this.handle = mountGraph(this.els.frame, payload);
    });
  }

  close(): void {
    this.els.panel.classList.add("hidden");
    this.els.panel.setAttribute("aria-hidden", "true");
    this.destroyGraph();
  }

  private destroyGraph(): void {
    if (this.handle) {
      this.handle.destroy();
      this.handle = null;
    }
  }
}

function mustGet<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing required element #${id}`);
  return node as T;
}

/** Wire up the Ask-page panel against the elements declared in `ask.html`. */
export function createGraphPanel(): GraphPanel {
  return new GraphPanel({
    panel: mustGet<HTMLElement>("graph-panel"),
    frame: mustGet<HTMLElement>("graph-frame"),
    empty: mustGet<HTMLElement>("graph-empty"),
    closeBtn: mustGet<HTMLButtonElement>("graph-close"),
  });
}
