import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_MANIFEST = resolve(ROOT, "teacher", "manifest.jsonl");
const RUNS_ROOT = resolve(ROOT, "teacher", "runs");

function readFlag(argv, index, name) {
  const arg = argv[index];
  const next = argv[index + 1];
  const prefix = `--${name}=`;

  if (arg === `--${name}` && next !== undefined) {
    return { value: next, advance: 1 };
  }
  if (typeof arg === "string" && arg.startsWith(prefix)) {
    return { value: arg.slice(prefix.length), advance: 0 };
  }
  return null;
}

function parseArgs(argv) {
  const result = {
    manifest: DEFAULT_MANIFEST,
    outputDir: null,
    baseUrl: process.env.BTL_TEACHER_BASE_URL?.trim() || "https://api.deepseek.com",
    apiKey: process.env.BTL_TEACHER_API_KEY?.trim() || "",
    model: process.env.BTL_TEACHER_MODEL?.trim() || "deepseek-v4-flash",
    limit: null,
    variants: new Set(["minimal", "verbose", "negative"]),
    depths: null,
    dryRun: false,
    delayMs: Number(process.env.BTL_TEACHER_DELAY_MS || 0),
    maxTokens: Number(process.env.BTL_TEACHER_MAX_TOKENS || 1024),
    temperature: null,
    thinking: process.env.BTL_TEACHER_THINKING?.trim() || "disabled",
    concurrency: Number(process.env.BTL_TEACHER_CONCURRENCY || 4),
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--dry-run") {
      result.dryRun = true;
      continue;
    }
    const manifest = readFlag(argv, i, "manifest");
    if (manifest) {
      result.manifest = resolve(manifest.value);
      i += manifest.advance;
      continue;
    }
    const outputDir = readFlag(argv, i, "output-dir");
    if (outputDir) {
      result.outputDir = resolve(outputDir.value);
      i += outputDir.advance;
      continue;
    }
    const baseUrl = readFlag(argv, i, "base-url");
    if (baseUrl) {
      result.baseUrl = baseUrl.value;
      i += baseUrl.advance;
      continue;
    }
    const apiKey = readFlag(argv, i, "api-key");
    if (apiKey) {
      result.apiKey = apiKey.value;
      i += apiKey.advance;
      continue;
    }
    const model = readFlag(argv, i, "model");
    if (model) {
      result.model = model.value;
      i += model.advance;
      continue;
    }
    const limit = readFlag(argv, i, "limit");
    if (limit) {
      const value = Number(limit.value);
      if (Number.isInteger(value) && value > 0) result.limit = value;
      i += limit.advance;
      continue;
    }
    const depths = readFlag(argv, i, "depths");
    if (depths) {
      const items = depths.value
        .split(",")
        .map((value) => Number(value.trim()))
        .filter((value) => Number.isInteger(value) && value > 0);
      if (items.length > 0) result.depths = new Set(items);
      i += depths.advance;
      continue;
    }
    const variants = readFlag(argv, i, "variants");
    if (variants) {
      const items = variants.value
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean);
      if (items.length > 0) result.variants = new Set(items);
      i += variants.advance;
      continue;
    }
    const delay = readFlag(argv, i, "delay-ms");
    if (delay) {
      const value = Number(delay.value);
      if (Number.isFinite(value) && value >= 0) result.delayMs = value;
      i += delay.advance;
      continue;
    }
    const maxTokens = readFlag(argv, i, "max-tokens");
    if (maxTokens) {
      const value = Number(maxTokens.value);
      if (Number.isInteger(value) && value > 0) result.maxTokens = value;
      i += maxTokens.advance;
      continue;
    }
    const temperature = readFlag(argv, i, "temperature");
    if (temperature) {
      const value = Number(temperature.value);
      if (Number.isFinite(value)) result.temperature = value;
      i += temperature.advance;
      continue;
    }
    const thinking = readFlag(argv, i, "thinking");
    if (thinking) {
      result.thinking = thinking.value;
      i += thinking.advance;
      continue;
    }
    const concurrency = readFlag(argv, i, "concurrency");
    if (concurrency) {
      const value = Number(concurrency.value);
      if (Number.isInteger(value) && value > 0) result.concurrency = value;
      i += concurrency.advance;
      continue;
    }
  }

  return result;
}

function readJsonl(path) {
  const text = readFileSync(path, "utf-8");
  if (!text.trim()) return [];
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`Failed to parse ${path} at line ${index + 1}: ${error.message}`);
      }
    });
}

function writeJsonl(path, rows) {
  writeFileSync(path, rows.map((row) => JSON.stringify(row)).join("\n"));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildEndpoint(baseUrl, pathname) {
  const normalized = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  return new URL(pathname, normalized).toString();
}

function buildMessages(job) {
  if (Array.isArray(job.messages) && job.messages.length >= 2) {
    return job.messages;
  }
  const user = [
    `Variant: ${job.variant}`,
    `Depth: ${job.depth}`,
    `Source row: ${job.source_index + 1}`,
    `Task family: ${job.family}`,
    `User task: ${job.utterance}`,
    `Reference chain:`,
    "```json",
    JSON.stringify(job.reference_chain ?? [], null, 2),
    "```",
    "Instructions:",
    `- ${
      job.variant === "minimal"
        ? "Produce the shortest valid reasoning wrapper and preserve the reference chain exactly."
        : job.variant === "verbose"
        ? "Produce a longer but still correct reasoning wrapper and preserve the reference chain exactly."
        : "Produce a plausible near-miss trace with exactly one controlled structural mistake."
    }`,
  ].join("\n");

  return [
    {
      role: "system",
      content: [
        "You are the BTL-1 teacher model.",
        "Return exactly one assistant message.",
        "The assistant message must be exactly: <reasoning>...</reasoning> followed by a JSON array of tool calls.",
        "Do not add markdown, labels, or commentary outside the assistant message.",
        "Keep the JSON valid and preserve the requested style exactly.",
      ].join("\n"),
    },
    { role: "user", content: user },
  ];
}

function extractTrace(text) {
  if (typeof text !== "string") {
    return { ok: false, reasoning: "", payload: null, error: "missing text" };
  }

  const match = text.trim().match(/^<reasoning>([\s\S]*?)<\/reasoning>\s*([\s\S]*)$/);
  if (!match) {
    return { ok: false, reasoning: "", payload: null, error: "missing reasoning block" };
  }

  const reasoning = match[1].trim();
  const payloadText = match[2].trim();
  try {
    const payload = JSON.parse(payloadText);
    return { ok: true, reasoning, payload, error: "" };
  } catch (error) {
    return { ok: false, reasoning, payload: null, error: error.message };
  }
}

async function fetchCompletion(job, config) {
  const temperature = config.temperature ?? (job.variant === "verbose" ? 0.4 : 0.2);
  const body = {
    model: config.model,
    messages: buildMessages(job),
    max_tokens: config.maxTokens,
    thinking: { type: config.thinking },
  };
  if (config.thinking === "disabled") {
    body.temperature = temperature;
  } else {
    body.reasoning_effort = "high";
  }

  const response = await fetch(buildEndpoint(config.baseUrl, "chat/completions"), {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify(body),
  });

  const text = await response.text();
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    throw new Error(`Non-JSON response (${response.status}): ${text.slice(0, 500)}`);
  }

  if (!response.ok) {
    const message = payload?.error?.message || payload?.message || `HTTP ${response.status}`;
    throw new Error(message);
  }

  const choice = payload?.choices?.[0];
  const content = choice?.message?.content ?? "";
  return {
    raw: payload,
    content,
    finish_reason: choice?.finish_reason ?? "",
  };
}

function ensureOutputDir(baseDir) {
  mkdirSync(baseDir, { recursive: true });
  mkdirSync(join(baseDir, "raw"), { recursive: true });
  mkdirSync(join(baseDir, "completed"), { recursive: true });
  mkdirSync(join(baseDir, "failed"), { recursive: true });
}

async function main() {
  const config = parseArgs(process.argv.slice(2));
  if (!existsSync(config.manifest)) {
    throw new Error(`Manifest not found: ${config.manifest}`);
  }
  if (!config.dryRun && !config.apiKey) {
    throw new Error("BTL_TEACHER_API_KEY is required unless --dry-run is set.");
  }
  if (!config.model && !config.dryRun) {
    throw new Error("BTL_TEACHER_MODEL is required unless --dry-run is set.");
  }

  const jobs = readJsonl(config.manifest).filter((job) => {
    if (config.depths && !config.depths.has(Number(job.depth))) return false;
    if (config.variants && !config.variants.has(String(job.variant))) return false;
    return true;
  });

  const selected = config.limit ? jobs.slice(0, config.limit) : jobs;
  const runId = new Date().toISOString().replace(/[:.]/g, "-");
  const outputDir = config.outputDir || join(RUNS_ROOT, runId);
  if (!config.dryRun) {
    ensureOutputDir(outputDir);
  }

  const rawRows = [];
  const completedRows = [];
  const failedRows = [];
  let nextIndex = 0;
  async function worker(workerId) {
    while (true) {
      const index = nextIndex++;
      if (index >= selected.length) return;
      const job = selected[index];
      const baseRow = {
        id: job.id,
        depth: job.depth,
        variant: job.variant,
        source_index: job.source_index,
        source_kind: job.source_kind,
        family: job.family,
        utterance: job.utterance,
        reference_chain: job.reference_chain,
      };

      try {
        let completion;
        if (config.dryRun) {
          completion = {
            content: "<reasoning>Dry run</reasoning>\n[]",
            raw: null,
            finish_reason: "dry_run",
          };
        } else {
          completion = await fetchCompletion(job, config);
        }

        const extracted = extractTrace(completion.content);
        const rawRow = {
          ...baseRow,
          request: {
            model: config.model || "dry-run",
            base_url: config.baseUrl,
            thinking: config.thinking,
            temperature: config.temperature ?? (job.variant === "verbose" ? 0.4 : 0.2),
            max_tokens: config.maxTokens,
            messages: buildMessages(job),
          },
          response: {
            finish_reason: completion.finish_reason,
            raw: completion.raw,
            assistant_content: completion.content,
            parsed_ok: extracted.ok,
            parse_error: extracted.error,
          },
        };
        rawRows.push(rawRow);

        if (extracted.ok) {
          completedRows.push({
            ...baseRow,
            messages: [...buildMessages(job), { role: "assistant", content: completion.content }],
            assistant_content: completion.content,
            reasoning: extracted.reasoning,
            trace: extracted.payload,
          });
        } else {
          failedRows.push({
            ...baseRow,
            assistant_content: completion.content,
            parse_error: extracted.error,
          });
        }
      } catch (error) {
        failedRows.push({
          ...baseRow,
          error: error instanceof Error ? error.message : String(error),
        });
      }

      if (config.delayMs > 0) {
        await sleep(config.delayMs);
      }

      if ((index + 1) % 25 === 0 || index + 1 === selected.length) {
        console.log(`teacher run: ${index + 1}/${selected.length}`);
      }
    }
  }

  const workerCount = Math.min(config.concurrency, selected.length);
  await Promise.all(Array.from({ length: workerCount }, (_, i) => worker(i + 1).catch((error) => {
    failedRows.push({
      id: `worker-${i + 1}`,
      error: error instanceof Error ? error.message : String(error),
    });
  })));

  if (!config.dryRun) {
    writeJsonl(join(outputDir, "raw.jsonl"), rawRows);
    writeJsonl(join(outputDir, "completed.jsonl"), completedRows);
    writeJsonl(join(outputDir, "failed.jsonl"), failedRows);
    writeFileSync(
      join(outputDir, "summary.json"),
      JSON.stringify(
        {
          manifest: config.manifest,
          base_url: config.baseUrl,
          model: config.model || "dry-run",
          requested: selected.length,
          completed: completedRows.length,
          failed: failedRows.length,
          raw_rows: rawRows.length,
        },
        null,
        2,
      ),
    );
  }

  console.log(`teacher run output -> ${outputDir}`);
  console.log(`  completed: ${completedRows.length}`);
  console.log(`  failed: ${failedRows.length}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
