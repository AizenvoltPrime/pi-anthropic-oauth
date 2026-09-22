import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { streamAnthropicOAuth } from "../.test-dist/stream.js";

/**
 * Claude Opus 5.5 as pi 0.87.1 ships it (providers/data/anthropic.json). The devDependency pins an
 * older pi-ai that has no entry for it, so the fields this provider reads are copied here.
 */
const OPUS_55 = {
  id: "claude-opus-5-5",
  name: "Claude Opus 5.5",
  api: "anthropic-messages",
  provider: "anthropic",
  reasoning: true,
  input: ["text", "image"],
  cost: { input: 4, output: 20, cacheRead: 0.2, cacheWrite: 5 },
  contextWindow: 1000000,
  maxTokens: 128000,
  thinkingLevelMap: {
    off: null,
    minimal: null,
    low: "low",
    medium: "medium",
    high: "high",
    xhigh: "xhigh",
    max: "max",
  },
  compat: {
    supportsMidConvoEffort: true,
    supportsMidConvoSystemMessages: true,
    supportsMidConvoToolChanges: true,
    forceAdaptiveThinking: true,
    supportsTemperature: false,
    supportsStrictTools: true,
  },
};

/** Sonnet 5 as shipped: adaptive thinking it can turn off, and no managed effort. */
const SONNET_5 = {
  id: "claude-sonnet-5",
  name: "Claude Sonnet 5",
  api: "anthropic-messages",
  provider: "anthropic",
  reasoning: true,
  input: ["text", "image"],
  cost: { input: 2, output: 10, cacheRead: 0.2, cacheWrite: 2.5 },
  contextWindow: 1000000,
  maxTokens: 128000,
  thinkingLevelMap: { minimal: null, low: "low", medium: "medium", high: "high", xhigh: "xhigh", max: "max" },
  compat: { forceAdaptiveThinking: true },
};

const CONTEXT = {
  messages: [
    { role: "system", content: "SYSTEM", timestamp: 0 },
    { role: "user", content: [{ type: "text", text: "hello" }], timestamp: 1 },
  ],
};

/** Serve one request: record it, answer with `respond`, return what was sent and what came back. */
async function exchange(model, options, respond) {
  let body;
  let headers;
  const server = http.createServer((req, res) => {
    headers = req.headers;
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      respond(res);
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const result = await streamAnthropicOAuth(
      { ...model, baseUrl: `http://127.0.0.1:${server.address().port}` },
      CONTEXT,
      { apiKey: "sk-ant-oat01-test", ...options },
    ).result();
    return { body, headers, result };
  } finally {
    server.close();
  }
}

function reject(res) {
  res.writeHead(400, { "content-type": "application/json" });
  res.end(JSON.stringify({ type: "error", error: { type: "invalid_request_error", message: "captured" } }));
}

function sse(events) {
  return (res) => {
    res.writeHead(200, { "content-type": "text/event-stream" });
    for (const event of events) res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
    res.end();
  };
}

for (const level of ["low", "medium", "high", "xhigh", "max"]) {
  test(`Opus 5.5 at ${level} sends that effort, not a collapsed one`, async () => {
    const { body } = await exchange(OPUS_55, { reasoning: level }, reject);
    assert.deepEqual(body.output_config, { effort: level });
    assert.equal(body.thinking.type, "adaptive");
    assert.equal(body.thinking.display, "summarized");
  });
}

test("Opus 5.5 with no level requested still thinks, at high", async () => {
  const { body } = await exchange(OPUS_55, {}, reject);
  assert.equal(body.thinking.type, "adaptive", "the model rejects thinking.type disabled");
  assert.deepEqual(body.output_config, { effort: "high" });
});

// Anthropic returns 400 when a thinking block is replayed after the prompt prefix changed, and this
// provider rewrites the prefix on every mid-session prompt or tool change.
test("Opus 5.5 asks Anthropic to drop a thinking block whose prefix changed", async () => {
  const { body, headers } = await exchange(OPUS_55, { reasoning: "high" }, reject);
  assert.deepEqual(body.thinking.block_binding, { prefix_mismatch_behavior: "drop_block" });
  assert.match(headers["anthropic-beta"], /thinking-binding-controls-2026-08-01/);
  assert.match(headers["anthropic-beta"], /claude-code-20250219/, "Claude Code betas stay");
});

test("Opus 5.5 gets its full output limit, since thinking counts toward it", async () => {
  const { body } = await exchange(OPUS_55, { reasoning: "max" }, reject);
  assert.equal(body.max_tokens, 128000);
});

test("an explicit output limit still wins", async () => {
  const { body } = await exchange(OPUS_55, { reasoning: "high", maxTokens: 1 }, reject);
  assert.equal(body.max_tokens, 1);
});

test("Sonnet 5 with thinking off sends disabled, because omitting it leaves thinking on", async () => {
  const { body, headers } = await exchange(SONNET_5, {}, reject);
  assert.deepEqual(body.thinking, { type: "disabled" });
  assert.equal(body.output_config, undefined);
  assert.doesNotMatch(headers["anthropic-beta"], /thinking-binding-controls/, "only managed-effort models bind blocks");
});

test("Sonnet 5 at xhigh sends xhigh", async () => {
  const { body } = await exchange(SONNET_5, { reasoning: "xhigh" }, reject);
  assert.equal(body.thinking.type, "adaptive");
  assert.deepEqual(body.output_config, { effort: "xhigh" });
});

test("a refusal surfaces as an error carrying Anthropic's explanation", async () => {
  const { result } = await exchange(
    OPUS_55,
    { reasoning: "high" },
    sse([
      {
        type: "message_start",
        message: {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-opus-5-5",
          content: [],
          stop_reason: null,
          usage: { input_tokens: 10, output_tokens: 0 },
        },
      },
      {
        type: "message_delta",
        delta: {
          stop_reason: "refusal",
          stop_details: { type: "refusal", category: "reasoning_extraction", explanation: "Declined: reasoning extraction." },
        },
        usage: { output_tokens: 3 },
      },
      { type: "message_stop" },
    ]),
  );
  assert.equal(result.stopReason, "error");
  assert.equal(result.errorMessage, "Declined: reasoning extraction.");
  assert.equal(result.rawStopReason, "refusal");
});
