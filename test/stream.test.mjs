import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { streamAnthropicOAuth } from "../.test-dist/stream.js";

/**
 * Since pi 0.86.0 a provider receives a `TranscriptContext`: `{ messages }` and nothing else. The
 * prompt and the tool declarations live on the transcript's system messages, so a provider that reads
 * `context.systemPrompt` / `context.tools` silently sends no prompt and no tools.
 *
 * These tests drive the provider against a capture server and assert on the request body, because
 * that body is the only place the omission is observable.
 */

const TOOLS = [
  {
    name: "bash",
    description: "Execute a shell command",
    parameters: {
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
    },
  },
  {
    name: "SaveMemory",
    description: "Persist a memory",
    parameters: {
      type: "object",
      properties: { content: { type: "string" } },
      required: ["content"],
    },
  },
];

const MODEL = {
  id: "claude-opus-5",
  name: "Claude Opus 5",
  api: "anthropic-messages",
  provider: "anthropic",
  reasoning: true,
  input: ["text"],
  cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
  contextWindow: 1000000,
  maxTokens: 128000,
  compat: { forceAdaptiveThinking: true },
};

/** Run one request against a throwaway server and return the JSON body the provider sent. */
async function capture(context) {
  let body;
  const server = http.createServer((req, res) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      res.writeHead(400, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          type: "error",
          error: { type: "invalid_request_error", message: "captured" },
        }),
      );
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const model = { ...MODEL, baseUrl: `http://127.0.0.1:${server.address().port}` };
  try {
    await streamAnthropicOAuth(model, context, {
      apiKey: "sk-ant-oat01-test",
      reasoning: "high",
    }).result();
  } finally {
    server.close();
  }
  return body;
}

/** A transcript the way the agent loop builds one: tools declared on the leading system message. */
function transcript({ tools = TOOLS, sections, content = "" } = {}) {
  return {
    messages: [
      {
        role: "system",
        content,
        ...(sections ? { sections } : {}),
        ...(tools.length > 0 ? { toolsAdded: tools } : {}),
        timestamp: 0,
      },
      { role: "user", content: [{ type: "text", text: "run something" }], timestamp: 1 },
    ],
  };
}

test("declares the transcript's tools on the request", async () => {
  const body = await capture(transcript());
  assert.ok(body.tools, "request carried no tools array");
  assert.deepEqual(
    body.tools.map((tool) => tool.name),
    ["Bash", "mcp__pi__SaveMemory"],
    "OAuth requests use Claude Code casing for known names and namespace the rest",
  );
  assert.deepEqual(body.tools[0].input_schema.required, ["command"]);
});

test("sends the transcript's system prompt, not just the Claude Code identity", async () => {
  const body = await capture(
    transcript({
      sections: { preamble: "PREAMBLE TEXT", damocles_tone: "<damocles_tone>\nBe concise.\n</damocles_tone>" },
    }),
  );
  const text = body.system.map((block) => block.text).join("\n\n");
  assert.match(text, /You are Claude Code/, "identity block must stay first for allowance billing");
  assert.match(text, /PREAMBLE TEXT/);
  assert.match(text, /Be concise\./);
});

test("replays later section patches into the prompt it sends", async () => {
  const context = transcript({ sections: { preamble: "V1", extra: "KEEP" } });
  context.messages.push({
    role: "system",
    content: "",
    sections: { preamble: "V2", extra: null },
    timestamp: 2,
  });
  context.messages.push({ role: "user", content: [{ type: "text", text: "again" }], timestamp: 3 });

  const body = await capture(context);
  const text = body.system.map((block) => block.text).join("\n\n");
  assert.match(text, /V2/, "a patched section must reach the model");
  assert.doesNotMatch(text, /V1/, "the superseded value must not be sent");
  assert.doesNotMatch(text, /KEEP/, "a section removed by a null patch must not be sent");
  assert.ok(
    body.messages.every((message) => message.role !== "system"),
    "system messages are collapsed into params.system, never sent as conversation turns",
  );
});

test("withdraws tools removed by a later system message", async () => {
  const context = transcript();
  context.messages.push({
    role: "system",
    content: "",
    toolsRemoved: [{ name: "bash" }],
    timestamp: 2,
  });
  context.messages.push({ role: "user", content: [{ type: "text", text: "again" }], timestamp: 3 });

  const body = await capture(context);
  assert.deepEqual(
    body.tools.map((tool) => tool.name),
    ["mcp__pi__SaveMemory"],
  );
});

test("omits the tools array when the transcript declares none", async () => {
  const body = await capture(transcript({ tools: [] }));
  assert.equal(body.tools, undefined);
});
