import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname, extname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { slugify, replaceWords } from "./variant-utils.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SPEC_PATH = resolve(ROOT, "specs", "depth-4.json");
const OUTPUT_DIR = resolve(ROOT, "depth-4");

const DEFAULT_CONTEXTS = [
  "billing",
  "checkout",
  "analytics",
  "support",
  "inventory",
  "platform",
  "onboarding",
  "payments",
  "ops",
  "security",
  "finance",
  "compliance",
];

const SYSTEM_PROMPT = [
  "You are an expert repo repair agent.",
  "Given a multi-file project with a bug, navigate the project, diagnose the failure, and produce the minimal cross-file fix.",
  "Output a JSON array of tool calls.",
  "Use file_search to find files, read_file to inspect them, write_file to make edits.",
  "The fix must address the root cause, not just the symptom.",
  "No reasoning tags. No extra text.",
].join("\n");

const PROJECT_PROFILES = {
  "py-todo-app-01": {
    fileTokens: ["complete_task", "db", "tasks", "database", "todo"],
    domainHint: "web_app",
  },
  "py-cli-tool-01": {
    fileTokens: ["merge_rows", "read_csv", "row_a", "row_b", "rows_a", "rows_b", "merge"],
    domainHint: "cli_tool",
  },
  "py-api-service-01": {
    fileTokens: ["rate_limit", "_requests", "get_data", "server"],
    domainHint: "api_service",
  },
};

function pickContext(spec, index) {
  const contexts = spec.generation?.variant_contexts?.length
    ? spec.generation.variant_contexts
    : DEFAULT_CONTEXTS;
  return contexts[index % contexts.length];
}

function short(text, limit = 90) {
  return String(text).replace(/\s+/g, " ").trim().slice(0, limit);
}

function renamePath(path, suffix) {
  const ext = extname(path);
  const base = basename(path, ext);
  return `${base}_${suffix}${ext}`;
}

function buildProjectMap(project, context, index) {
  const suffix = `${slugify(context)}_${String(index + 1).padStart(3, "0")}`;
  const profile = PROJECT_PROFILES[project.id] || { fileTokens: [], domainHint: project.domain || "repo" };
  const replacements = {};

  for (const token of profile.fileTokens) {
    replacements[token] = `${token}_${suffix}`;
  }

  replacements[project.domain] = `${project.domain}_${suffix}`;
  replacements[profile.domainHint] = `${profile.domainHint}_${suffix}`;

  return { replacements, suffix, profile };
}

function transformFile(file, suffix, replacements) {
  const path = renamePath(file.path, suffix);
  const content = replaceWords(file.content, replacements);
  return { path, content };
}

function buildUserPrompt(project, utterance, transformedFiles, context, variant) {
  const fileList = transformedFiles.map((f) => `  ${f.path}`).join("\n");
  const fileSummaries = transformedFiles
    .map((f) => `  ${f.path}: ${short(f.content)}`)
    .join("\n");

  return [
    `Variant: ${variant}`,
    "Depth: 4",
    "Task: repo repair",
    `Domain: ${context}/${project.domain}`,
    `User request: ${utterance}`,
    "",
    "Project files:",
    fileList,
    "",
    "Project files:",
    fileSummaries,
    "",
    variant === "minimal"
      ? "Instructions:\n- Navigate the project, find the root cause, apply the minimal fix.\n- Use file_search, read_file, write_file as needed.\n- Fix only what is broken - no refactoring."
      : "Instructions:\n- Navigate the project, find the root cause.\n- Produce a near-miss trace: make a change that looks plausible but does not address the actual cross-file dependency issue.",
  ].join("\n");
}

function buildVariantRow(project, variant, context, index) {
  const { replacements, suffix } = buildProjectMap(project, context, index);
  const transformedFiles = project.files.map((file) => transformFile(file, suffix, replacements));
  const expectedFixFile = renamePath(project.expected_fix_file || project.files?.[0]?.path, suffix);
  const expectedFixContent = replaceWords(project.expected_fix_content, replacements);
  const utterance = replaceWords(project.utterance, replacements);
  const domain = `${context}_${project.domain}`;

  return {
    kind: "depth-4",
    variant,
    language: project.language,
    domain,
    utterance,
    files: transformedFiles,
    expected_fix_file: expectedFixFile,
    expected_fix_content: expectedFixContent,
    test: replaceWords(project.test, replacements),
    teacher_prompt: buildUserPrompt(project, utterance, transformedFiles, context, variant),
    system_prompt: SYSTEM_PROMPT,
    causal_note: replaceWords(project.causal_note, replacements),
    base_project_id: project.id,
    variant_context: context,
    variant_index: index,
    variant_id: `${slugify(project.id)}_${slugify(context)}_${String(index + 1).padStart(3, "0")}`,
  };
}

function writeJsonl(path, rows) {
  writeFileSync(path, rows.map((row) => JSON.stringify(row)).join("\n"));
  console.log(`  ${path}: ${rows.length} rows`);
}

function main() {
  const spec = JSON.parse(readFileSync(SPEC_PATH, "utf-8"));
  mkdirSync(OUTPUT_DIR, { recursive: true });

  const perProject = spec.generation?.per_project ?? 8;
  const minimalTasks = [];
  const negativeTasks = [];

  for (const project of spec.projects) {
    for (let index = 0; index < perProject; index++) {
      const context = pickContext(spec, index);
      minimalTasks.push(buildVariantRow(project, "minimal", context, index));
      negativeTasks.push(buildVariantRow(project, "negative", context, index));
    }
  }

  writeJsonl(resolve(OUTPUT_DIR, "minimal.jsonl"), minimalTasks);
  writeJsonl(resolve(OUTPUT_DIR, "negative.jsonl"), negativeTasks);

  const shiftDir = resolve(OUTPUT_DIR, "shifts");
  mkdirSync(shiftDir, { recursive: true });
  writeJsonl(resolve(shiftDir, "shift-domain.jsonl"), minimalTasks.map((row) => ({ ...row, shift: "domain" })));

  console.log("\ndepth-4 totals:");
  console.log(`  ${minimalTasks.length} minimal`);
  console.log(`  ${negativeTasks.length} negative`);
  console.log(`  contexts: ${spec.generation?.variant_contexts?.length || DEFAULT_CONTEXTS.length}`);
  console.log(`  variants per project: ${perProject}`);
}

main();
