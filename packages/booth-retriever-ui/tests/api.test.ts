// Vitest unit tests for the fetch wrappers in ``src/api.ts``.
//
// Covers:
//   - happy-path JSON parsing and 204 No Content returns;
//   - URL construction (query strings, id escaping);
//   - 404 / 422 error shape translation into ``ApiError``.
//
// We stub ``globalThis.fetch`` per-test instead of wiring a real server —
// the Python-side tests already cover the HTTP contract end-to-end.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  approveQuery,
  askQuestion,
  editQuery,
  fetchGraph,
  fetchQueries,
  fetchQuery,
  fetchStats,
  rejectQuery,
  submitFeedback,
} from "../src/api.js";
import { ApiError } from "../src/types.js";

interface StubCall {
  url: string;
  init: RequestInit | undefined;
}

function stubFetch(responder: (call: StubCall) => Response | Promise<Response>): {
  calls: StubCall[];
  spy: ReturnType<typeof vi.fn>;
} {
  const calls: StubCall[] = [];
  const spy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const call: StubCall = { url, init };
    calls.push(call);
    return responder(call);
  });
  globalThis.fetch = spy as unknown as typeof fetch;
  return { calls, spy };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function emptyResponse(status: number): Response {
  return new Response(null, { status });
}

// ---------------------------------------------------------------------------

describe("api.ts happy paths", () => {
  const originalFetch = globalThis.fetch;
  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("fetchStats parses the JSON payload", async () => {
    const { calls } = stubFetch(() =>
      jsonResponse({ total: 4, counts: { approved: 3, rejected: 1 } }),
    );
    const stats = await fetchStats();
    expect(stats.total).toBe(4);
    expect(stats.counts.approved).toBe(3);
    expect(calls[0]?.url).toBe("/api/stats");
  });

  it("fetchQueries defaults to the pending list", async () => {
    const { calls } = stubFetch(() => jsonResponse([]));
    await fetchQueries();
    expect(calls[0]?.url).toBe("/api/queries");
  });

  it("fetchQueries builds a query string from status and limit", async () => {
    const { calls } = stubFetch(() => jsonResponse([]));
    await fetchQueries({ status: "approved", limit: 25 });
    expect(calls[0]?.url).toBe("/api/queries?status=approved&limit=25");
  });

  it("fetchQuery URL-encodes the id", async () => {
    const { calls } = stubFetch(() =>
      jsonResponse({
        query_id: "id with space",
        query_text: "q",
        status: "approved",
        risk_level: null,
        timestamp: "t",
        user_feedback: null,
        has_fewshot: false,
        rejection_reason: null,
        fewshot_cypher: null,
        fewshot_parameters: [],
      }),
    );
    await fetchQuery("id with space");
    expect(calls[0]?.url).toBe("/api/queries/id%20with%20space");
  });

  it("approveQuery posts JSON and returns the result", async () => {
    const { calls } = stubFetch(() =>
      jsonResponse({ query_id: "q1", fewshot_id: "fs-1", fewshot_was_new: true }),
    );
    const result = await approveQuery("q1", {
      cypher_template: "RETURN 1",
      parameters: ["x"],
    });
    expect(result.fewshot_id).toBe("fs-1");
    expect(result.fewshot_was_new).toBe(true);

    const call = calls[0];
    expect(call?.url).toBe("/api/queries/q1/approve");
    expect(call?.init?.method).toBe("POST");
    expect(call?.init?.body).toBe(
      JSON.stringify({ cypher_template: "RETURN 1", parameters: ["x"] }),
    );
    const headers = new Headers(call?.init?.headers);
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("editQuery resolves on 204", async () => {
    stubFetch(() => emptyResponse(204));
    await expect(
      editQuery("q1", { cypher_template: "RETURN 1" }),
    ).resolves.toBeUndefined();
  });

  it("rejectQuery sends the provided reason", async () => {
    const { calls } = stubFetch(() => emptyResponse(204));
    await rejectQuery("q1", { reason: "off-topic" });
    expect(calls[0]?.init?.body).toBe(JSON.stringify({ reason: "off-topic" }));
  });

  it("rejectQuery works without a reason", async () => {
    const { calls } = stubFetch(() => emptyResponse(204));
    await rejectQuery("q1");
    expect(calls[0]?.init?.body).toBe("{}");
  });

  it("submitFeedback posts the helpful flag", async () => {
    const { calls } = stubFetch(() => emptyResponse(204));
    await submitFeedback("q1", { helpful: true });
    expect(calls[0]?.init?.body).toBe(JSON.stringify({ helpful: true }));
  });

  it("fetchGraph requests the graph endpoint and parses the payload", async () => {
    const { calls } = stubFetch(() =>
      jsonResponse({
        nodes: [{ id: "1", caption: "q", labels: ["Query"], properties: {} }],
        relationships: [],
      }),
    );
    const graph = await fetchGraph("q1");
    expect(graph.nodes).toHaveLength(1);
    expect(graph.nodes[0]?.id).toBe("1");
    expect(calls[0]?.url).toBe("/api/queries/q1/graph");
  });

  it("askQuestion posts the question and parses the response", async () => {
    const { calls } = stubFetch(() =>
      jsonResponse({
        success: true,
        answer: "42",
        query_id: "q-123",
        similar_match: true,
        high_risk: false,
        declined: false,
        cypher_used: "MATCH (n) RETURN count(n)",
        tool_used: "cache_hit",
        error_message: null,
        pending_feedback: true,
      }),
    );

    const resp = await askQuestion({
      query_text: "How many nodes?",
      is_high_risk: false,
    });

    expect(resp.answer).toBe("42");
    expect(resp.query_id).toBe("q-123");
    expect(resp.similar_match).toBe(true);
    expect(resp.tool_used).toBe("cache_hit");

    const call = calls[0];
    expect(call?.url).toBe("/api/ask");
    expect(call?.init?.method).toBe("POST");
    expect(call?.init?.body).toBe(
      JSON.stringify({ query_text: "How many nodes?", is_high_risk: false }),
    );
    const headers = new Headers(call?.init?.headers);
    expect(headers.get("Content-Type")).toBe("application/json");
  });
});

// ---------------------------------------------------------------------------

describe("api.ts error handling", () => {
  const originalFetch = globalThis.fetch;
  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("404 throws ApiError with the server detail", async () => {
    stubFetch(() =>
      jsonResponse({ detail: "No query with id 'nope'" }, 404),
    );
    await expect(fetchQuery("nope")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      detail: "No query with id 'nope'",
    });
  });

  it("422 throws ApiError carrying the verification message", async () => {
    stubFetch(() =>
      jsonResponse(
        { detail: "cypher_template failed verification: missing RETURN" },
        422,
      ),
    );
    try {
      await approveQuery("q1", { cypher_template: "MATCH (n)" });
      throw new Error("expected approveQuery to reject");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.status).toBe(422);
      expect(apiErr.detail).toContain("failed verification");
    }
  });

  it("non-JSON error body falls back to statusText", async () => {
    stubFetch(
      () => new Response("boom", { status: 500, statusText: "Server Error" }),
    );
    await expect(fetchStats()).rejects.toMatchObject({
      status: 500,
      detail: "Server Error",
    });
  });
});

// ---------------------------------------------------------------------------

describe("api.ts wiring", () => {
  const originalFetch = globalThis.fetch;
  beforeEach(() => {
    // Make sure each test starts with a clean fetch reference.
    globalThis.fetch = originalFetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("approveQuery throws a clear error if fetch itself rejects", async () => {
    globalThis.fetch = vi
      .fn()
      .mockRejectedValue(new Error("network down")) as unknown as typeof fetch;
    await expect(
      approveQuery("q1", { cypher_template: "RETURN 1" }),
    ).rejects.toThrow("network down");
  });
});
