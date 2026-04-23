// Entrypoint for the Ask page. Mirrors the shape of ``main.ts`` — no
// framework, just DOM wiring — and deliberately keeps the UX single-turn
// (the latest answer replaces the previous one, matching the Streamlit
// reference app in packages/booth-retriever/examples/streamlit_app/).
//
// Contrast with main.ts (the curator page):
//   - no list view, no periodic refresh;
//   - the only mutable state is which ``query_id`` we should send feedback
//     against; everything else lives in the DOM.

import { askQuestion, submitFeedback } from "./api.js";
import { renderAnswer, showToast } from "./render.js";
import { ApiError } from "./types.js";

interface AskElements {
  form: HTMLFormElement;
  input: HTMLTextAreaElement;
  risk: HTMLInputElement;
  submit: HTMLButtonElement;
  answerCard: HTMLElement;
}

const state = {
  lastQueryId: null as string | null,
};

function mustGet<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing required element #${id}`);
  return node as T;
}

function bindElements(): AskElements {
  return {
    form: mustGet<HTMLFormElement>("ask-form"),
    input: mustGet<HTMLTextAreaElement>("ask-input"),
    risk: mustGet<HTMLInputElement>("ask-risk"),
    submit: mustGet<HTMLButtonElement>("ask-submit"),
    answerCard: mustGet<HTMLElement>("answer-card"),
  };
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

async function onSubmit(els: AskElements, ev: SubmitEvent): Promise<void> {
  ev.preventDefault();
  const query = els.input.value.trim();
  if (!query) {
    showToast("Type a question first.", "error");
    els.input.focus();
    return;
  }

  els.submit.disabled = true;
  const originalLabel = els.submit.textContent;
  els.submit.textContent = "Asking\u2026";

  try {
    const resp = await askQuestion({
      query_text: query,
      is_high_risk: els.risk.checked,
    });
    // Feedback targets the query that just came back — stash the id for
    // the onFeedback callback (closure would work too, but we also want
    // to null it out on reset / re-ask).
    state.lastQueryId = resp.declined ? null : resp.query_id;
    renderAnswer(els.answerCard, resp, {
      onFeedback: (helpful) => void sendFeedback(helpful),
    });
    els.answerCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    els.answerCard.classList.add("hidden");
    els.answerCard.replaceChildren();
    showToast(describeError(err), "error");
  } finally {
    els.submit.disabled = false;
    els.submit.textContent = originalLabel ?? "Ask";
  }
}

async function sendFeedback(helpful: boolean): Promise<void> {
  const queryId = state.lastQueryId;
  if (!queryId) {
    // Shouldn't happen — renderAnswer hides the buttons in this case —
    // but guard anyway so a stray click doesn't produce a confusing error.
    return;
  }
  try {
    await submitFeedback(queryId, { helpful });
    showToast(
      helpful
        ? "Marked helpful. Query is now pending curator approval."
        : "Marked for human review.",
      "info",
    );
  } catch (err) {
    showToast(describeError(err), "error");
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

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

function main(): void {
  const els = bindElements();
  els.form.addEventListener("submit", (ev) => void onSubmit(els, ev));

  // Ctrl/Cmd+Enter inside the textarea submits — saves a trip to the button
  // for keyboard-first users.
  els.input.addEventListener("keydown", (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") {
      ev.preventDefault();
      els.form.requestSubmit();
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main);
} else {
  main();
}
