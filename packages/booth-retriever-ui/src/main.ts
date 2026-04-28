// Entrypoint for the curate page. Wires DOM elements declared in
// ``index.html`` to the ``api.ts`` fetch wrappers and the ``render.ts``
// view functions.
//
// State is intentionally trivial (a single selected query id, a single
// status filter); anything more elaborate justifies reaching for a
// framework, at which point the plan's "simplest thing that works"
// premise has gone out the window.

import {
  approveQuery,
  deleteQuery,
  editQuery,
  fetchQueries,
  fetchQuery,
  fetchStats,
  rejectQuery,
} from "./api.js";
import {
  renderDetail,
  renderQueryList,
  renderStats,
  showDetailError,
  showToast,
} from "./render.js";
import type { PendingQuery, QueryDetail } from "./types.js";
import { ApiError } from "./types.js";

interface AppElements {
  statsTiles: HTMLElement;
  queryList: HTMLElement;
  detailPane: HTMLElement;
  statusFilter: HTMLSelectElement;
  refreshBtn: HTMLButtonElement;
}

const state = {
  selectedId: null as string | null,
  statusFilter: "pending_approval" as string,
};

function mustGet<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing required element #${id}`);
  return node as T;
}

function bindElements(): AppElements {
  return {
    statsTiles: mustGet<HTMLElement>("stats-tiles"),
    queryList: mustGet<HTMLElement>("query-list"),
    detailPane: mustGet<HTMLElement>("detail-pane"),
    statusFilter: mustGet<HTMLSelectElement>("status-filter"),
    refreshBtn: mustGet<HTMLButtonElement>("refresh-btn"),
  };
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

async function loadStats(els: AppElements): Promise<void> {
  try {
    const stats = await fetchStats();
    renderStats(els.statsTiles, stats);
  } catch (err) {
    els.statsTiles.replaceChildren();
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = `Stats failed: ${describeError(err)}`;
    els.statsTiles.appendChild(p);
  }
}

async function loadQueries(els: AppElements): Promise<PendingQuery[]> {
  try {
    const rows = await fetchQueries(
      state.statusFilter ? { status: state.statusFilter } : {},
    );
    renderQueryList(els.queryList, rows, {
      selectedId: state.selectedId,
      onSelect: (id) => void selectQuery(els, id),
    });
    return rows;
  } catch (err) {
    els.queryList.replaceChildren();
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = `Failed to load queries: ${describeError(err)}`;
    els.queryList.appendChild(li);
    return [];
  }
}

async function selectQuery(els: AppElements, id: string): Promise<void> {
  state.selectedId = id;

  // Reflect selection in the list without a full re-fetch.
  for (const li of els.queryList.querySelectorAll<HTMLElement>(".query-row")) {
    const isSelected = li.dataset.queryId === id;
    li.classList.toggle("selected", isSelected);
    li.setAttribute("aria-selected", String(isSelected));
  }

  try {
    const detail = await fetchQuery(id);
    renderDetail(els.detailPane, detail, {
      onApprove: (cypher, params) => void approve(els, detail, cypher, params),
      onEdit: (cypher, params) => void edit(els, detail, cypher, params),
      onReject: (reason) => void reject(els, detail, reason),
      onDelete: () => void remove(els, detail),
    });
  } catch (err) {
    els.detailPane.replaceChildren();
    const p = document.createElement("p");
    p.className = "warn";
    p.textContent = `Failed to load query ${id}: ${describeError(err)}`;
    els.detailPane.appendChild(p);
  }
}

async function approve(
  els: AppElements,
  detail: QueryDetail,
  cypherTemplate: string,
  parameters: string[],
): Promise<void> {
  try {
    const result = await approveQuery(detail.query_id, {
      cypher_template: cypherTemplate,
      parameters,
    });
    showToast(
      `Approved (fewshot ${result.fewshot_was_new ? "created" : "updated"})`,
      "info",
    );
    await refreshAll(els);
  } catch (err) {
    handleMutationError(els, err);
  }
}

async function edit(
  els: AppElements,
  detail: QueryDetail,
  cypherTemplate: string,
  parameters: string[],
): Promise<void> {
  try {
    await editQuery(detail.query_id, {
      cypher_template: cypherTemplate,
      parameters,
    });
    showToast("Saved edits", "info");
    await refreshAll(els);
  } catch (err) {
    handleMutationError(els, err);
  }
}

async function reject(
  els: AppElements,
  detail: QueryDetail,
  reason: string | null,
): Promise<void> {
  try {
    await rejectQuery(detail.query_id, { reason });
    showToast("Rejected", "info");
    clearDetail(els);
    await refreshAll(els);
  } catch (err) {
    handleMutationError(els, err);
  }
}

async function remove(els: AppElements, detail: QueryDetail): Promise<void> {
  try {
    await deleteQuery(detail.query_id);
    showToast("Query deleted", "info");
    clearDetail(els);
    await refreshAll(els);
  } catch (err) {
    handleMutationError(els, err);
  }
}

function clearDetail(els: AppElements): void {
  state.selectedId = null;
  els.detailPane.replaceChildren();
  const p = document.createElement("p");
  p.className = "muted";
  p.textContent = "Select a query on the left to curate it.";
  els.detailPane.appendChild(p);
}

async function refreshAll(els: AppElements): Promise<void> {
  await Promise.all([loadStats(els), loadQueries(els)]);
  if (state.selectedId) {
    await selectQuery(els, state.selectedId);
  }
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.detail;
  if (err instanceof Error) return err.message;
  return String(err);
}

function handleMutationError(els: AppElements, err: unknown): void {
  if (err instanceof ApiError && err.status === 422) {
    showDetailError(els.detailPane, err.detail);
    showToast(err.detail, "error");
    return;
  }
  showToast(describeError(err), "error");
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

function main(): void {
  const els = bindElements();

  els.statusFilter.addEventListener("change", () => {
    state.statusFilter = els.statusFilter.value;
    void loadQueries(els);
  });

  els.refreshBtn.addEventListener("click", () => {
    void refreshAll(els);
  });

  void refreshAll(els);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main);
} else {
  main();
}
