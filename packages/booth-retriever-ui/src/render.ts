// Pure-function DOM renderers for the curate page.
//
// Keep these free of ``fetch`` / side effects: they take already-fetched
// data (or callbacks) and return DOM nodes or mutate an existing container.
// ``main.ts`` wires them to the API and to event handlers.

import type {
  AskResponse,
  PendingQuery,
  QueryDetail,
  StatsResponse,
} from "./types.js";

/** Display order of status counters in the top bar. */
const STATUS_ORDER = [
  "pending_approval",
  "approved",
  "rejected",
  "declined",
  "needs_review",
] as const;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  opts: {
    className?: string;
    text?: string;
    attrs?: Record<string, string>;
  } = {},
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (opts.className) node.className = opts.className;
  if (opts.text !== undefined) node.textContent = opts.text;
  if (opts.attrs) {
    for (const [k, v] of Object.entries(opts.attrs)) node.setAttribute(k, v);
  }
  return node;
}

function truncate(s: string, n = 90): string {
  return s.length > n ? `${s.slice(0, n - 1)}\u2026` : s;
}

// ---------------------------------------------------------------------------
// Stats tiles
// ---------------------------------------------------------------------------

export function renderStats(container: HTMLElement, stats: StatsResponse): void {
  container.replaceChildren();

  const total = el("div", { className: "tile tile-total" });
  total.append(
    el("span", { className: "tile-label", text: "total" }),
    el("span", { className: "tile-value", text: String(stats.total) }),
  );
  container.appendChild(total);

  // Known statuses first (in a stable order), then any extras the server
  // surprises us with. Missing known statuses render as zero.
  const seen = new Set<string>();
  for (const key of STATUS_ORDER) {
    seen.add(key);
    container.appendChild(buildTile(key, stats.counts[key] ?? 0));
  }
  for (const [key, value] of Object.entries(stats.counts)) {
    if (!seen.has(key)) container.appendChild(buildTile(key, value));
  }
}

function buildTile(status: string, count: number): HTMLElement {
  const tile = el("div", { className: `tile tile-${status}` });
  tile.append(
    el("span", { className: "tile-label", text: status }),
    el("span", { className: "tile-value", text: String(count) }),
  );
  return tile;
}

// ---------------------------------------------------------------------------
// Queries list
// ---------------------------------------------------------------------------

export interface QueryListOptions {
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function renderQueryList(
  container: HTMLElement,
  rows: PendingQuery[],
  opts: QueryListOptions,
): void {
  container.replaceChildren();
  if (rows.length === 0) {
    const empty = el("li", {
      className: "muted empty",
      text: "No queries for this filter.",
    });
    container.appendChild(empty);
    return;
  }
  for (const row of rows) {
    container.appendChild(buildQueryRow(row, opts));
  }
}

function buildQueryRow(row: PendingQuery, opts: QueryListOptions): HTMLElement {
  const isSelected = row.query_id === opts.selectedId;
  const li = el("li", {
    className: `query-row${isSelected ? " selected" : ""}`,
    attrs: {
      role: "option",
      tabindex: "0",
      "data-query-id": row.query_id,
      "aria-selected": String(isSelected),
    },
  });

  const top = el("div", { className: "query-row-top" });
  top.append(
    el("span", { className: `badge badge-${row.status}`, text: row.status }),
    el("span", {
      className: "risk",
      text: row.risk_level ? `risk: ${row.risk_level}` : "risk: n/a",
    }),
    ...(row.has_fewshot
      ? [el("span", { className: "badge badge-fewshot", text: "fewshot" })]
      : []),
  );

  const text = el("p", {
    className: "query-row-text",
    text: truncate(row.query_text, 120),
  });

  const meta = el("p", {
    className: "query-row-meta muted",
    text: row.timestamp,
  });

  li.append(top, text, meta);
  li.addEventListener("click", () => opts.onSelect(row.query_id));
  li.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      opts.onSelect(row.query_id);
    }
  });
  return li;
}

// ---------------------------------------------------------------------------
// Detail pane
// ---------------------------------------------------------------------------

export interface DetailCallbacks {
  onApprove: (cypherTemplate: string, parameters: string[]) => void;
  onEdit: (cypherTemplate: string, parameters: string[]) => void;
  onReject: (reason: string | null) => void;
  onFeedback: (helpful: boolean) => void;
}

export function renderDetail(
  container: HTMLElement,
  detail: QueryDetail,
  callbacks: DetailCallbacks,
): void {
  container.replaceChildren();

  const header = el("header", { className: "detail-header" });
  header.append(
    el("h2", { text: "Query detail" }),
    el("span", { className: `badge badge-${detail.status}`, text: detail.status }),
    el("span", {
      className: "muted",
      text: detail.risk_level ? `risk: ${detail.risk_level}` : "risk: n/a",
    }),
  );
  container.appendChild(header);

  container.appendChild(
    labeled("Question", el("p", { className: "query-text", text: detail.query_text })),
  );

  if (detail.rejection_reason) {
    container.appendChild(
      labeled(
        "Previous rejection reason",
        el("p", { className: "warn", text: detail.rejection_reason }),
      ),
    );
  }
  if (detail.user_feedback) {
    container.appendChild(
      labeled(
        "User feedback",
        el("p", { className: "muted", text: detail.user_feedback }),
      ),
    );
  }

  // ---- Cypher textarea -----------------------------------------------------
  const cypherLabel = el("label", {
    className: "field-label",
    text: "Cypher template",
    attrs: { for: "cypher-input" },
  });
  const cypher = el("textarea", {
    className: "cypher-input",
    attrs: {
      id: "cypher-input",
      rows: "10",
      spellcheck: "false",
      placeholder: "MATCH (n) RETURN n LIMIT 10",
    },
  });
  cypher.value = detail.fewshot_cypher ?? "";

  const cypherError = el("p", {
    className: "field-error hidden",
    attrs: { role: "alert" },
  });

  // ---- Parameters input ----------------------------------------------------
  const paramsLabel = el("label", {
    className: "field-label",
    text: "Parameters (comma-separated)",
    attrs: { for: "params-input" },
  });
  const params = el("input", {
    className: "params-input",
    attrs: {
      id: "params-input",
      type: "text",
      placeholder: "tenant, user_id",
    },
  });
  params.value = detail.fewshot_parameters.join(", ");

  // ---- Action buttons ------------------------------------------------------
  const actions = el("div", { className: "actions" });
  const approveBtn = el("button", {
    className: "primary",
    text: "Approve",
    attrs: { type: "button" },
  });
  const editBtn = el("button", {
    className: "secondary",
    text: "Save edits",
    attrs: { type: "button" },
  });
  const rejectBtn = el("button", {
    className: "danger",
    text: "Reject",
    attrs: { type: "button" },
  });

  const updateDisabled = () => {
    const hasCypher = cypher.value.trim().length > 0;
    approveBtn.disabled = !hasCypher;
    editBtn.disabled = !hasCypher;
  };
  cypher.addEventListener("input", () => {
    cypherError.classList.add("hidden");
    cypherError.textContent = "";
    updateDisabled();
  });
  updateDisabled();

  const parseParams = (): string[] =>
    params.value
      .split(",")
      .map((p) => p.trim())
      .filter((p) => p.length > 0);

  approveBtn.addEventListener("click", () => {
    callbacks.onApprove(cypher.value.trim(), parseParams());
  });
  editBtn.addEventListener("click", () => {
    callbacks.onEdit(cypher.value.trim(), parseParams());
  });
  rejectBtn.addEventListener("click", () => {
    const reason = window.prompt("Rejection reason (optional):", "");
    callbacks.onReject(reason && reason.trim().length > 0 ? reason.trim() : null);
  });

  actions.append(approveBtn, editBtn, rejectBtn);

  // ---- Feedback sub-bar (only meaningful once a fewshot exists) -----------
  const feedbackBar = el("div", { className: "feedback-bar" });
  feedbackBar.append(
    el("span", { className: "muted", text: "Mark this fewshot:" }),
    (() => {
      const b = el("button", {
        text: "Helpful",
        className: "link",
        attrs: { type: "button" },
      });
      b.addEventListener("click", () => callbacks.onFeedback(true));
      return b;
    })(),
    (() => {
      const b = el("button", {
        text: "Not helpful",
        className: "link",
        attrs: { type: "button" },
      });
      b.addEventListener("click", () => callbacks.onFeedback(false));
      return b;
    })(),
  );

  container.append(
    cypherLabel,
    cypher,
    cypherError,
    paramsLabel,
    params,
    actions,
  );
  if (detail.fewshot_cypher) {
    container.appendChild(feedbackBar);
  }
}

/** Surface a verification error (422) inline under the cypher textarea. */
export function showDetailError(container: HTMLElement, message: string): void {
  const errorNode = container.querySelector<HTMLElement>(".field-error");
  if (!errorNode) return;
  errorNode.textContent = message;
  errorNode.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Ask / answer card
// ---------------------------------------------------------------------------

export interface AnswerCallbacks {
  onFeedback: (helpful: boolean) => void;
}

/**
 * Paint a single-turn response from ``POST /api/ask`` into ``container``.
 *
 * The UI mirrors the Streamlit reference app: a status banner derived
 * from ``resp.success`` / ``resp.declined``, the answer text, an
 * expandable metadata block, and (when feedback is possible) a pair of
 * Helpful / Not helpful buttons. Feedback is disabled for declined
 * responses or when the server didn't surface a ``query_id``.
 */
export function renderAnswer(
  container: HTMLElement,
  resp: AskResponse,
  callbacks: AnswerCallbacks,
): void {
  container.replaceChildren();
  container.classList.remove("hidden");

  const kind = answerKind(resp);
  const banner = el("div", {
    className: `answer-banner answer-banner-${kind}`,
    attrs: { role: "status" },
  });
  banner.append(
    el("span", { className: "answer-kind", text: kindLabel(kind) }),
    ...(resp.similar_match
      ? [el("span", { className: "badge badge-fewshot", text: "cache hit" })]
      : []),
    ...(resp.high_risk
      ? [el("span", { className: "badge badge-rejected", text: "high risk" })]
      : []),
  );
  container.appendChild(banner);

  container.appendChild(
    el("p", { className: `answer-text answer-text-${kind}`, text: resp.answer }),
  );

  // Metadata — collapsed by default; mirrors Streamlit's st.json(...) expander.
  const details = el("details", { className: "answer-meta" });
  details.appendChild(el("summary", { text: "Response metadata" }));
  const pre = el("pre", { className: "answer-meta-json" });
  pre.textContent = JSON.stringify(
    {
      query_id: resp.query_id,
      similar_match: resp.similar_match,
      high_risk: resp.high_risk,
      declined: resp.declined,
      tool_used: resp.tool_used,
      cypher_used: resp.cypher_used,
      error_message: resp.error_message,
    },
    null,
    2,
  );
  details.appendChild(pre);
  container.appendChild(details);

  // Feedback bar — only meaningful when the server actually persisted a
  // Query node we can attach feedback to.
  const canFeedback = Boolean(resp.query_id) && !resp.declined;
  const feedbackBar = el("div", { className: "feedback-bar" });
  feedbackBar.append(
    el("span", {
      className: "muted",
      text: canFeedback
        ? "Was this answer helpful?"
        : "Feedback unavailable for this response.",
    }),
  );
  if (canFeedback) {
    const helpful = el("button", {
      text: "Helpful",
      className: "primary",
      attrs: { type: "button" },
    });
    const notHelpful = el("button", {
      text: "Not helpful",
      className: "secondary",
      attrs: { type: "button" },
    });
    const disableAfter = () => {
      helpful.disabled = true;
      notHelpful.disabled = true;
    };
    helpful.addEventListener("click", () => {
      disableAfter();
      callbacks.onFeedback(true);
    });
    notHelpful.addEventListener("click", () => {
      disableAfter();
      callbacks.onFeedback(false);
    });
    feedbackBar.append(helpful, notHelpful);
  }
  container.appendChild(feedbackBar);
}

type AnswerKind = "success" | "declined" | "error" | "info";

function answerKind(resp: AskResponse): AnswerKind {
  if (resp.declined) return "declined";
  if (resp.success) return "success";
  if (resp.error_message) return "error";
  return "info";
}

function kindLabel(kind: AnswerKind): string {
  switch (kind) {
    case "success":
      return "Answer";
    case "declined":
      return "Declined";
    case "error":
      return "Error";
    default:
      return "Info";
  }
}

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------

export function showToast(message: string, kind: "info" | "error" = "info"): void {
  const host = document.getElementById("toast-container");
  if (!host) return;
  const toast = el("div", { className: `toast toast-${kind}`, text: message });
  host.appendChild(toast);
  setTimeout(() => toast.classList.add("visible"), 10);
  setTimeout(() => {
    toast.classList.remove("visible");
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function labeled(labelText: string, node: HTMLElement): HTMLElement {
  const wrap = el("div", { className: "labeled" });
  wrap.append(el("span", { className: "field-label", text: labelText }), node);
  return wrap;
}
