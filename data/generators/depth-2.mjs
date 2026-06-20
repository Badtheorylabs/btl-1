import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SPEC_PATH = resolve(ROOT, "specs", "depth-2.json");
const OUTPUT_DIR = resolve(ROOT, "depth-2");

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

function fillValue(value, vars) {
  if (typeof value === "string") {
    if (value.startsWith("$")) return value;
    return fillTemplate(value, vars);
  }
  if (Array.isArray(value)) {
    return value.map((item) => fillValue(item, vars));
  }
  if (value && typeof value === "object") {
    const out = {};
    for (const [key, item] of Object.entries(value)) {
      out[key] = fillValue(item, vars);
    }
    return out;
  }
  return value;
}

function buildSystemPrompt() {
  return [
    "You are generating training traces for BTL-1, a local coding agent.",
    "Output a JSON array of tool calls for the given user request.",
    "Each tool call has: tool name, params object, unique id, and depends_on if it references a prior step.",
    "Use $step_N.result to reference outputs from earlier steps.",
    "No reasoning tags. No extra text.",
    "Use the shortest correct tool chain.",
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

function fillOutputSteps(steps, vars) {
  return steps.map((step) => ({
    ...step,
    params: fillValue(step.params, vars),
  }));
}

function generateChainTask(chainDef, pools, rng) {
  const template = pick(chainDef.utterances, rng);
  const vars = buildVars(template, pools, rng);
  const utterance = fillTemplate(template, vars);
  const expected = fillOutputSteps(chainDef.output, vars);

  return {
    kind: "depth-2",
    pattern: chainDef.pattern,
    steps: chainDef.steps,
    template,
    vars,
    utterance,
    expected,
    system_prompt: buildSystemPrompt(),
  };
}

function shiftChainUtterance(base, chainDef, rng) {
  let altTemplate = pick(chainDef.utterances, rng);
  if (chainDef.utterances.length > 1) {
    let guard = 0;
    while (altTemplate === base.template && guard < 10) {
      altTemplate = pick(chainDef.utterances, rng);
      guard += 1;
    }
  }
  return fillTemplate(altTemplate, base.vars);
}

function main() {
  const spec = JSON.parse(readFileSync(SPEC_PATH, "utf-8"));
  const pools = { ...spec.pools };
  const rng = new SeededRng(42);

  const numPerChain = spec.generation?.per_chain ?? 150;
  const numShift = spec.generation?.per_chain_shift ?? 40;

  const tasks = [];
  const shiftTasks = [];

  for (const chainDef of spec.chains) {
    const baseRows = [];
    for (let i = 0; i < numPerChain; i++) {
      const base = generateChainTask(chainDef, pools, rng);
      tasks.push(base);
      baseRows.push(base);
    }
    for (let i = 0; i < numShift; i++) {
      const base = baseRows[i % baseRows.length];
      shiftTasks.push({
        ...base,
        utterance: shiftChainUtterance(base, chainDef, rng),
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
    pattern: t.pattern,
    steps: t.steps,
    utterance: t.utterance,
    expected: t.expected,
    system_prompt: t.system_prompt,
  })).join("\n"));
  writeFileSync(shiftPath, shiftTasks.map((t) => JSON.stringify({
    kind: t.kind,
    pattern: t.pattern,
    steps: t.steps,
    utterance: t.utterance,
    expected: t.expected,
    system_prompt: t.system_prompt,
    shift_type: t.shift_type,
    original_utterance: t.original_utterance,
  })).join("\n"));

  const totalTasks = tasks.length;
  const totalShift = shiftTasks.length;
  const uniquePatterns = new Set(tasks.map((t) => t.pattern)).size;

  console.log(`depth-2: ${totalTasks} train, ${totalShift} shift, ${uniquePatterns} patterns`);
  console.log(`  train -> ${trainPath}`);
  console.log(`  shift -> ${shiftPath}`);
}

main();
