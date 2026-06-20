import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT_DIR = resolve(ROOT, "teacher");

const DEPTH_1_2_VARIANTS = ["minimal", "verbose", "negative"];
const DEPTH_3_VARIANTS = ["minimal", "verbose", "negative", "counterfactual"];

function parseArgs(argv) {
  const result = { depths: [0, 1, 2, 3, 4], limit: null };
  for (const arg of argv) {
    if (arg.startsWith("--depths=")) {
      const value = arg.split("=", 2)[1] || "";
      result.depths = value.split(",").map(Number).filter(n => Number.isInteger(n) && n >= 0);
    }
    if (arg.startsWith("--limit=")) {
      const value = Number(arg.split("=", 2)[1]);
      if (Number.isInteger(value) && value > 0) result.limit = value;
    }
  }
  return result;
}

function readJsonl(path) {
  const text = readFileSync(path, "utf-8");
  if (!text.trim()) return [];
  return text.split(/\r?\n/).map(l => l.trim()).filter(Boolean).map((line, i) => {
    try { return JSON.parse(line); }
    catch (e) { throw new Error(`Failed to parse ${path} at line ${i+1}: ${e.message}`); }
  });
}

function writeJsonl(path, rows) {
  writeFileSync(path, rows.map(r => JSON.stringify(r)).join("\n"));
}

function pad(v, w) { return String(v).padStart(w, "0"); }

function label(row) {
  return row.tool || row.pattern || row.bug_type || row.kind || "task";
}

function buildCodeEditPrompt(row, variant) {
  const lines = [
    `Variant: ${variant}`,
    `Depth: 3`,
    `Task: code edit`,
    `Bug type: ${row.bug_type}`,
    `User request: ${row.utterance}`,
    ``,
    `Buggy code:`,
    "```" + row.language,
    row.buggy_code,
    "```",
    ``,
  ];

  if (variant === "minimal") {
    lines.push(
      "Instructions:",
      "- Output a tool chain that reads the file then writes the fix.",
      "- Use exactly 2 steps: read_file then write_file.",
      "- The fix must be minimal — change only what is needed.",
      "- No extra steps, no commentary."
    );
  } else if (variant === "verbose") {
    lines.push(
      "Instructions:",
      "- Output a tool chain that reads the file, analyzes the bug, then writes the fix.",
      "- Use 3 steps: read_file, reasoning, write_file.",
      "- The reasoning step should explain the causal failure.",
      "- The write_file step should contain the fix."
    );
  } else if (variant === "negative") {
    lines.push(
      "Instructions:",
      "- Output a tool chain with a near-miss wrong fix.",
      `- The correct fix is: ${row.expected_fix}`,
      "- Your trace should use a plausible but wrong fix that looks correct but does not address the causal failure.",
      "- The tool chain format should still be valid."
    );
    if (row.negative_fix) {
      lines.push(`- Example wrong fix (do not use this exact one, but something similarly wrong): ${row.negative_fix}`);
    }
  } else if (variant === "counterfactual") {
    lines.push(
      "Instructions:",
      "- Review the proposed fix and determine whether it fixes the causal failure.",
      "- Output reasoning + a JSON decision: {\"accept\": true/false, \"reason\": \"...\"}",
      `- Proposed fix:\n${row.proposed_fix || "Not provided"}`
    );
  }

  return lines.join("\n");
}

function buildToolChainPrompt(row, depth, variant) {
  const chain = JSON.stringify(row.expected ?? [], null, 2);
  const title = label(row);
  const lines = [
    `Variant: ${variant}`,
    `Depth: ${depth}`,
    `Task family: ${title}`,
    `User task: ${row.utterance}`,
    `Reference chain:`,
    "```json",
    chain,
    "```",
  ];

  if (variant === "minimal") {
    lines.push(
      "Instructions:",
      "- Preserve the reference chain exactly.",
      "- Keep the reasoning wrapper as short as possible.",
      "- No extra steps or commentary."
    );
  } else if (variant === "verbose") {
    lines.push(
      "Instructions:",
      "- Preserve the reference chain exactly.",
      "- Write a longer reasoning wrapper explaining why the chain works.",
      "- No extra steps."
    );
  } else {
    const steps = Array.isArray(row.expected) ? row.expected : [];
    let recipe = "Make exactly one structural mistake.";
    if (steps.length <= 1) {
      recipe = `Use a near-miss sibling tool instead of ${steps[0]?.tool || "the correct tool"}.`;
    } else if (steps.length === 2) {
      recipe = "Swap the step order or break the dependency chain.";
    } else {
      recipe = "Replace the middle step with a sibling tool.";
    }
    lines.push(
      "Instructions:",
      `- ${recipe}`,
      "- Keep the assistant message format valid."
    );
  }

  return lines.join("\n");
}

function buildRepoRepairPrompt(row, variant) {
  const fileList = (row.files || []).map(f => `  ${f.path}: ${f.content.slice(0, 80).replace(/\n/g, ' ')}...`).join("\n");
  return [
    `Variant: ${variant}`,
    `Depth: 4`,
    `Task: repo repair`,
    `Domain: ${row.domain}`,
    `User request: ${row.utterance}`,
    ``,
    `Project files:`,
    fileList,
    ``,
    variant === "minimal"
      ? "Instructions:\n- Navigate the project, find the root cause, apply the minimal fix.\n- Use file_search, read_file, write_file as needed.\n- Fix only what is broken — no refactoring."
      : "Instructions:\n- Navigate the project, find the root cause.\n- Produce a near-miss trace: make a change that looks plausible but does not address the actual cross-file dependency issue.",
  ].join("\n");
}

function buildJob(row, depth, variant, index) {
  const id = `depth-${depth}-${variant}-${pad(index + 1, 4)}`;

  let systemPrompt, userPrompt;

  if (depth === 3) {
    systemPrompt = [
      "You are the BTL-1 teacher model for code edit tasks.",
      "Given buggy code and a user request, produce the correct tool chain (read_file -> write_file for minimal, add reasoning for verbose).",
      "The assistant message must be: <reasoning>...</reasoning> then a JSON array of tool calls.",
      "No markdown, no extra commentary.",
    ].join("\n");
    userPrompt = buildCodeEditPrompt(row, variant);
  } else if (depth === 4) {
    systemPrompt = [
      "You are the BTL-1 teacher model for repo repair tasks.",
      "Given a multi-file project with a bug, produce a tool chain that navigates the project and applies the fix.",
      "The assistant message must be: <reasoning>...</reasoning> then a JSON array of tool calls.",
      "No markdown, no extra commentary.",
    ].join("\n");
    userPrompt = buildRepoRepairPrompt(row, variant);
  } else {
    systemPrompt = [
      "You are the BTL-1 teacher model for tool orchestration tasks.",
      "Convert the source task into a training trace.",
      "The assistant message must be: <reasoning>...</reasoning> followed by a JSON array of tool calls.",
      "No markdown, no extra commentary.",
    ].join("\n");
    userPrompt = buildToolChainPrompt(row, depth, variant);
  }

  return { id, depth, variant, source_index: index, source_kind: row.kind || `depth-${depth}`, family: label(row), utterance: row.utterance, messages: [{ role: "system", content: systemPrompt }, { role: "user", content: userPrompt }] };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(OUTPUT_DIR, { recursive: true });

  const manifest = [];
  const byVariant = { minimal: [], verbose: [], negative: [] };

  for (const depth of args.depths) {
    let sourceRows = [];

    if (depth === 0) {
      for (const variant of ["minimal", "verbose", "negative"]) {
        const path = resolve(ROOT, "coding", `${variant}.jsonl`);
        const rows = readJsonl(path);
        const selected = args.limit ? rows.slice(0, args.limit) : rows;
        for (let i = 0; i < selected.length; i++) {
          const row = selected[i];
          const sysPrompt = row.system_prompt || "You are a code generation teacher. Output only the function implementation.";
          const userPrompt = row.teacher_prompt || row.prompt;
          const id = `coding-${variant}-${pad(i + 1, 4)}`;
          const job = { id, depth: 0, variant, source_index: i, source_kind: "coding", family: row.language, utterance: row.prompt, messages: [{ role: "system", content: sysPrompt }, { role: "user", content: userPrompt }] };
          manifest.push(job);
          if (["minimal", "verbose", "negative"].includes(variant)) {
            byVariant[variant]?.push(job);
          }
        }
      }
    } else if (depth <= 2) {
      const path = resolve(ROOT, `depth-${depth}`, "train.jsonl");
      sourceRows = readJsonl(path).map(r => ({ ...r, _expected: r.expected }));
      const selected = args.limit ? sourceRows.slice(0, args.limit) : sourceRows;
      for (let i = 0; i < selected.length; i++) {
        for (const variant of DEPTH_1_2_VARIANTS) {
          manifest.push(buildJob(selected[i], depth, variant, i));
          byVariant[variant]?.push(buildJob(selected[i], depth, variant, i));
        }
      }
    } else if (depth === 3) {
      for (const variant of DEPTH_3_VARIANTS) {
        const path = resolve(ROOT, `depth-${depth}`, `${variant}.jsonl`);
        const rows = readJsonl(path);
        const selected = args.limit ? rows.slice(0, args.limit) : rows;
        for (let i = 0; i < selected.length; i++) {
          manifest.push(buildJob(selected[i], depth, variant, i));
          if (["minimal", "verbose", "negative"].includes(variant)) {
            byVariant[variant]?.push(buildJob(selected[i], depth, variant, i));
          }
        }
      }
    } else if (depth === 4) {
      for (const variant of ["minimal", "negative"]) {
        const path = resolve(ROOT, `depth-${depth}`, `${variant}.jsonl`);
        const rows = readJsonl(path);
        const selected = args.limit ? rows.slice(0, args.limit) : rows;
        for (let i = 0; i < selected.length; i++) {
          manifest.push(buildJob(selected[i], depth, variant, i));
          if (variant === "minimal" || variant === "negative") {
            byVariant[variant]?.push(buildJob(selected[i], depth, variant, i));
          }
        }
      }
    }
  }

  writeJsonl(resolve(OUTPUT_DIR, "manifest.jsonl"), manifest);
  for (const variant of ["minimal", "verbose", "negative"]) {
    writeJsonl(resolve(OUTPUT_DIR, `${variant}.jsonl`), byVariant[variant]);
  }

  const summary = args.depths.map(d => `depth-${d}`).join(", ");
  console.log(`teacher pipeline built for ${summary}`);
  console.log(`  manifest   -> ${resolve(OUTPUT_DIR, "manifest.jsonl")} (${manifest.length} total jobs)`);
  console.log(`  minimal    -> ${byVariant.minimal.length}`);
  console.log(`  verbose    -> ${byVariant.verbose.length}`);
  console.log(`  negative   -> ${byVariant.negative.length}`);
}

main();
