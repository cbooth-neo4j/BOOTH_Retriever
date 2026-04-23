// TypeScript mirrors of the Pydantic schemas exposed by
// ``booth_retriever.web.api``. Kept deliberately small — only the fields
// the curate page renders or posts.

export interface StatsResponse {
  total: number;
  counts: Record<string, number>;
}

export interface PendingQuery {
  query_id: string;
  query_text: string;
  status: string;
  risk_level: string | null;
  timestamp: string;
  user_feedback: string | null;
  has_fewshot: boolean;
}

export interface QueryDetail extends PendingQuery {
  rejection_reason: string | null;
  fewshot_cypher: string | null;
  fewshot_parameters: string[];
}

export interface ApprovalResult {
  query_id: string;
  fewshot_id: string;
  fewshot_was_new: boolean;
}

export interface ApproveRequest {
  cypher_template: string;
  parameters?: string[];
  category?: string | null;
}

export interface EditRequest {
  cypher_template: string;
  parameters?: string[];
}

export interface RejectRequest {
  reason?: string | null;
}

export interface FeedbackRequest {
  helpful: boolean;
}

export interface AskRequest {
  query_text: string;
  is_high_risk?: boolean;
}

/**
 * Flattened ``BOOTHResponse`` returned by ``POST /api/ask``. Matches
 * ``_response_to_dict`` in ``booth_retriever/web/api.py``; ``raw_data`` is
 * intentionally excluded server-side because FewShot Cypher can return
 * projections that aren't JSON-serialisable.
 */
export interface AskResponse {
  success: boolean;
  answer: string;
  query_id: string | null;
  similar_match: boolean;
  high_risk: boolean;
  declined: boolean;
  cypher_used: string | null;
  tool_used: string | null;
  error_message: string | null;
  pending_feedback: boolean;
}

/**
 * Thrown by ``api.ts`` wrappers when the server returns a non-2xx response.
 * ``status`` is the HTTP status code; ``detail`` is whatever string the
 * FastAPI error handler surfaced (see ``_map_curator_value_error`` in
 * ``booth_retriever/web/api.py``).
 */
export class ApiError extends Error {
  public readonly status: number;
  public readonly detail: string;

  constructor(status: number, detail: string) {
    super(`HTTP ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}
