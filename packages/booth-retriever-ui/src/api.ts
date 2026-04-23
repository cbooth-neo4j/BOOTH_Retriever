// Typed ``fetch`` wrappers over the FastAPI routes in
// ``booth_retriever.web.api``. One function per endpoint, each returning a
// plain object (or throwing ``ApiError`` on failure).
//
// Why this lives here and not in each DOM handler:
//   - single place to handle response parsing / error shape normalisation;
//   - testable in isolation with a mocked ``fetch`` (see ``api.test.ts``).

import type {
  ApprovalResult,
  ApproveRequest,
  EditRequest,
  FeedbackRequest,
  PendingQuery,
  QueryDetail,
  RejectRequest,
  StatsResponse,
} from "./types.js";
import { ApiError } from "./types.js";

async function handle<T>(resp: Response): Promise<T> {
  if (resp.ok) {
    if (resp.status === 204) return undefined as T;
    return (await resp.json()) as T;
  }
  let detail = resp.statusText || "Request failed";
  try {
    const payload = await resp.json();
    if (payload && typeof payload.detail === "string") detail = payload.detail;
    else if (payload) detail = JSON.stringify(payload);
  } catch {
    // Non-JSON body — keep the statusText fallback.
  }
  throw new ApiError(resp.status, detail);
}

function jsonPost<B>(body: B): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// ---------------------------------------------------------------------------

export async function fetchStats(): Promise<StatsResponse> {
  const resp = await fetch("/api/stats");
  return handle<StatsResponse>(resp);
}

export async function fetchQueries(opts: {
  status?: string;
  limit?: number;
} = {}): Promise<PendingQuery[]> {
  const params = new URLSearchParams();
  if (opts.status) params.set("status", opts.status);
  if (opts.limit !== undefined) params.set("limit", String(opts.limit));
  const qs = params.toString();
  const url = qs ? `/api/queries?${qs}` : "/api/queries";
  const resp = await fetch(url);
  return handle<PendingQuery[]>(resp);
}

export async function fetchQuery(id: string): Promise<QueryDetail> {
  const resp = await fetch(`/api/queries/${encodeURIComponent(id)}`);
  return handle<QueryDetail>(resp);
}

export async function approveQuery(
  id: string,
  body: ApproveRequest,
): Promise<ApprovalResult> {
  const resp = await fetch(
    `/api/queries/${encodeURIComponent(id)}/approve`,
    jsonPost(body),
  );
  return handle<ApprovalResult>(resp);
}

export async function editQuery(id: string, body: EditRequest): Promise<void> {
  const resp = await fetch(
    `/api/queries/${encodeURIComponent(id)}/edit`,
    jsonPost(body),
  );
  await handle<void>(resp);
}

export async function rejectQuery(
  id: string,
  body: RejectRequest = {},
): Promise<void> {
  const resp = await fetch(
    `/api/queries/${encodeURIComponent(id)}/reject`,
    jsonPost(body),
  );
  await handle<void>(resp);
}

export async function submitFeedback(
  id: string,
  body: FeedbackRequest,
): Promise<void> {
  const resp = await fetch(
    `/api/queries/${encodeURIComponent(id)}/feedback`,
    jsonPost(body),
  );
  await handle<void>(resp);
}
