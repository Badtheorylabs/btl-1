import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SPEC_PATH = resolve(ROOT, "specs", "depth-1.json");
const OUTPUT_DIR = resolve(ROOT, "depth-1");

function pick(arr, rng) {
  return arr[Math.floor(rng.random() * arr.length)];
}

function uniquePlaceholders(template) {
  return [...new Set([...template.matchAll(/\{(\w+)\}/g)].map((match) => match[1]))];
}

function fillTemplate(template, vars) {
  return template.replace(/\{(\w+)\}/g, (_, key) => {
    if (vars[key] === undefined) return `{${key}}`;
    return String(vars[key]);
  });
}

function buildSystemPrompt() {
  return [
    "You are generating training traces for BTL-1, a local coding agent.",
    "Output a JSON array of tool calls for the given user request.",
    "Each tool call has: tool name, params object, and unique id.",
    "Use exactly one tool call. No reasoning tags. No extra text.",
    "Use the shortest correct parameter set. Skip optional params unless the input requires them.",
  ].join("\n");
}

class SeededRng {
  constructor(seed) {
    this.seed = seed;
    this.state = seed;
  }
  random() {
    this.state = (this.state * 1664525 + 1013904223) >>> 0;
    return (this.state >>> 0) / 4294967296;
  }
}

function buildVars(template, pools, rng) {
  const vars = {};
  for (const key of uniquePlaceholders(template)) {
    const pool = pools[key];
    if (Array.isArray(pool) && pool.length > 0) {
      vars[key] = pick(pool, rng);
    }
  }
  return vars;
}

function buildFileSearchParams(vars) {
  const params = {};
  if (vars.filename !== undefined) params.filename = vars.filename;
  if (vars.ext !== undefined) params.ext = vars.ext;
  if (vars.time !== undefined) params.time = vars.time;
  if (vars.folder !== undefined) params.folder = vars.folder;
  return params;
}

function buildReadFileParams(vars) {
  const params = {};
  const pathValue = vars.path || (vars.filename ? `${vars.filename}${vars.ext ? `.${vars.ext}` : ""}` : undefined);
  if (pathValue !== undefined) params.path = pathValue;
  if (vars.line_spec !== undefined) params.line_spec = vars.line_spec;
  return params;
}

function buildWriteFileParams(vars) {
  const params = {};
  const pathValue = vars.path || (vars.filename ? `${vars.filename}${vars.ext ? `.${vars.ext}` : ""}` : "output.txt");
  params.path = pathValue;
  if (vars.content !== undefined) params.content = vars.content;
  else if (vars.content_phrase !== undefined) params.content = vars.content_phrase;
  return params;
}

function buildEmailParams(template, vars) {
  const params = {};
  params.to = vars.contact ?? vars.to ?? "team@example.com";

  if (vars.subject !== undefined) {
    params.subject = vars.subject;
  } else if (vars.attachment_type !== undefined) {
    params.subject = vars.attachment_type;
  } else if (vars.filename !== undefined) {
    params.subject = vars.filename;
  } else {
    params.subject = "update";
  }

  if (vars.body !== undefined) {
    params.body = vars.body;
  } else if (vars.attachment_type !== undefined) {
    params.body = `Please see the ${String(vars.attachment_type).toLowerCase()} attached.`;
  } else if (vars.content !== undefined) {
    params.body = vars.content;
  } else {
    params.body = `Please review ${params.subject}.`;
  }

  const lowerTemplate = template.toLowerCase();
  if (/(attach|forward|send the .* to)/.test(lowerTemplate)) {
    if (vars.filename && vars.ext) {
      params.attachments = [`${vars.filename}.${vars.ext}`];
    } else if (vars.attachment_type !== undefined) {
      params.attachments = [String(vars.attachment_type).toLowerCase()];
    }
  }

  return params;
}

function buildWebSearchParams(vars) {
  let query = vars.topic ?? vars.query ?? "research";
  if (vars.detail !== undefined) query = `${vars.detail} about ${query}`;
  if (vars.time_range !== undefined) query = `${query} ${vars.time_range}`;
  if (vars.timeframe !== undefined) query = `${query} ${vars.timeframe}`;
  return { query };
}

function buildBrowseParams(vars) {
  return {
    url: vars.url ?? vars.url_target ?? "the first result",
  };
}

function buildClipboardParams(template, vars) {
  const lowerTemplate = template.toLowerCase();
  if (/(read|check|what did i copy|paste the current|paste clipboard)/.test(lowerTemplate)) {
    return { action: "read" };
  }
  return {
    action: "write",
    content: vars.content ?? vars.clip_content ?? "clipboard content",
  };
}

function buildShellCommandParams(vars) {
  return {
    command: vars.command ?? "echo hello",
  };
}

function buildNotesParams(template, vars) {
  const lowerTemplate = template.toLowerCase();
  let action = "write";
  if (/(show me|what is on|what is in|read|open|display|fetch)/.test(lowerTemplate)) {
    action = "read";
  } else if (/append/.test(lowerTemplate)) {
    action = "append";
  }

  const params = {
    action,
    title: vars.note_title ?? vars.title ?? "notes",
  };

  if (action !== "read") {
    params.content = vars.content_phrase ?? vars.content ?? "note entry";
  }

  return params;
}

function buildReasoningParams(vars) {
  const params = {
    question: vars.question ?? vars.reasoning_q ?? "What should I reason about?",
  };
  if (vars.context !== undefined || vars.reasoning_ctx !== undefined) {
    params.context = vars.context ?? vars.reasoning_ctx;
  }
  return params;
}

function buildParams(toolDef, template, vars) {
  switch (toolDef.tool) {
    case "file_search":
      return buildFileSearchParams(vars);
    case "read_file":
      return buildReadFileParams(vars);
    case "write_file":
      return buildWriteFileParams(vars);
    case "email":
      return buildEmailParams(template, vars);
    case "web_search":
      return buildWebSearchParams(vars);
    case "browse":
      return buildBrowseParams(vars);
    case "clipboard":
      return buildClipboardParams(template, vars);
    case "shell_command":
      return buildShellCommandParams(vars);
    case "notes":
      return buildNotesParams(template, vars);
    case "reasoning":
      return buildReasoningParams(vars);
    default:
      return {};
  }
}

function buildReasoning(tool, params) {
  switch (tool) {
    case "file_search":
      return `The user wants me to find ${params.filename ?? "the file"}. I'll do this directly.`;
    case "read_file":
      return `The user wants me to read ${params.path}. I'll do this directly.`;
    case "write_file":
      return `The user wants me to write to ${params.path}. I'll do this directly.`;
    case "email":
      return `The user wants me to email ${params.to}. I'll do this directly.`;
    case "web_search":
      return `The user wants me to search for ${params.query}. I'll do this directly.`;
    case "browse":
      return `The user wants me to open ${params.url}. I'll do this directly.`;
    case "clipboard":
      return `The user wants me to ${params.action} the clipboard. I'll do this directly.`;
    case "shell_command":
      return `The user wants me to run ${params.command}. I'll do this directly.`;
    case "notes":
      return `The user wants me to ${params.action} my ${params.title} notes. I'll do this directly.`;
    case "reasoning":
      return `The user wants me to reason about ${params.question}. I'll do this directly.`;
    default:
      return `The user wants me to execute ${tool}. I'll do this directly.`;
  }
}

function generateToolTask(toolDef, rng) {
  const template = pick(toolDef.utterances, rng);
  const vars = buildVars(template, toolDef.pool, rng);
  const utterance = fillTemplate(template, vars);
  const expectedParams = buildParams(toolDef, template, vars);
  const reasoning = buildReasoning(toolDef.tool, expectedParams);

  return {
    kind: "depth-1",
    tool: toolDef.tool,
    template,
    vars,
    utterance,
    expected: [{
      tool: toolDef.tool,
      params: expectedParams,
      id: "step_1",
    }],
    system_prompt: buildSystemPrompt(),
    reasoning,
  };
}

function shiftUtterance(base, toolDef, rng) {
  let altTemplate = pick(toolDef.utterances, rng);
  if (toolDef.utterances.length > 1) {
    let guard = 0;
    while (altTemplate === base.template && guard < 10) {
      altTemplate = pick(toolDef.utterances, rng);
      guard += 1;
    }
  }
  return fillTemplate(altTemplate, base.vars);
}

function main() {
  const spec = JSON.parse(readFileSync(SPEC_PATH, "utf-8"));
  const rng = new SeededRng(42);

  const numPerTool = spec.generation?.per_tool ?? 200;
  const numShift = spec.generation?.per_tool_shift ?? 50;

  const tasks = [];
  const shiftTasks = [];

  for (const toolDef of spec.tools) {
    const baseRows = [];
    for (let i = 0; i < numPerTool; i++) {
      const base = generateToolTask(toolDef, rng);
      tasks.push(base);
      baseRows.push(base);
    }
    for (let i = 0; i < numShift; i++) {
      const base = baseRows[i % baseRows.length];
      shiftTasks.push({
        ...base,
        utterance: shiftUtterance(base, toolDef, rng),
        shift_type: "surface",
        original_utterance: base.utterance,
      });
    }
  }

  mkdirSync(OUTPUT_DIR, { recursive: true });

  const trainPath = resolve(OUTPUT_DIR, "train.jsonl");
  const shiftPath = resolve(OUTPUT_DIR, "shift.jsonl");

  writeFileSync(trainPath, tasks.map((t) => JSON.stringify({
    kind: t.kind,
    tool: t.tool,
    utterance: t.utterance,
    expected: t.expected,
    system_prompt: t.system_prompt,
  })).join("\n"));
  writeFileSync(shiftPath, shiftTasks.map((t) => JSON.stringify({
    kind: t.kind,
    tool: t.tool,
    utterance: t.utterance,
    expected: t.expected,
    system_prompt: t.system_prompt,
    shift_type: t.shift_type,
    original_utterance: t.original_utterance,
  })).join("\n"));

  console.log(`depth-1: ${tasks.length} train, ${shiftTasks.length} shift`);
  console.log(`  train -> ${trainPath}`);
  console.log(`  shift -> ${shiftPath}`);
}

main();
