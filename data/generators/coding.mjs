import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SPEC_PATH = resolve(ROOT, "specs", "coding.json");
const OUTPUT_DIR = resolve(ROOT, "coding");

function buildMinimalPrompt(task) {
  return [
    `Write a ${task.language} function: ${task.prompt}`,
    "",
    "Return ONLY the function implementation. No imports unless needed. No explanation, no comments, no tests.",
    "Shortest correct solution.",
  ].join("\n");
}

function buildVerbosePrompt(task) {
  return [
    `Write a ${task.language} function: ${task.prompt}`,
    "",
    "Return the function with clear variable names, a short comment explaining the approach, and handle edge cases.",
    "Correct but with more explicit reasoning in comments and variable names.",
  ].join("\n");
}

function buildNegativePrompt(task) {
  return [
    `Write a ${task.language} function: ${task.prompt}`,
    "",
    "Return a function that looks correct at first glance but has a subtle bug.",
    `The bug should be: ${task.negative_reason}`,
    "Make it plausible — someone reviewing quickly might approve it.",
  ].join("\n");
}

function main() {
  const spec = JSON.parse(readFileSync(SPEC_PATH, "utf-8"));
  mkdirSync(OUTPUT_DIR, { recursive: true });

  const minimalTasks = [];
  const verboseTasks = [];
  const negativeTasks = [];
  const shiftTasks = [];

  for (const task of spec.tasks) {
    const base = {
      kind: "coding",
      depth: 0,
      language: task.language,
      prompt: task.prompt,
      minimal_solution: task.minimal_solution,
      verbose_solution: task.verbose_solution,
      negative_solution: task.negative_solution,
      negative_reason: task.negative_reason,
    };

    minimalTasks.push({
      ...base,
      variant: "minimal",
      teacher_prompt: buildMinimalPrompt(task),
      system_prompt: "You are a code generation teacher. Output only the function implementation — no explanations, no tests, no markdown formatting.",
    });

    verboseTasks.push({
      ...base,
      variant: "verbose",
      teacher_prompt: buildVerbosePrompt(task),
      system_prompt: "You are a code generation teacher. Output a well-commented, explicit, correct implementation with edge case handling.",
    });

    negativeTasks.push({
      ...base,
      variant: "negative",
      teacher_prompt: buildNegativePrompt(task),
      system_prompt: "You are a code generation teacher. Output a plausible-looking function with exactly one subtle bug. Make it convincing but wrong in the specified way.",
    });

    shiftTasks.push({
      ...base,
      variant: "minimal",
      shift: "language",
      teacher_prompt: `Write this in ${task.language === "python" ? "JavaScript" : "Python"}: ${task.prompt}`,
    });
  }

  function write(path, rows) {
    writeFileSync(path, rows.map(r => JSON.stringify(r)).join("\n"));
    console.log(`  ${path}: ${rows.length} rows`);
  }

  write(resolve(OUTPUT_DIR, "minimal.jsonl"), minimalTasks);
  write(resolve(OUTPUT_DIR, "verbose.jsonl"), verboseTasks);
  write(resolve(OUTPUT_DIR, "negative.jsonl"), negativeTasks);

  const shiftDir = resolve(OUTPUT_DIR, "shifts");
  mkdirSync(shiftDir, { recursive: true });
  write(resolve(shiftDir, "shift-language.jsonl"), shiftTasks);

  console.log(`\ncoding totals: ${spec.tasks.length} tasks`);
  console.log(`  ${minimalTasks.length} minimal`);
  console.log(`  ${verboseTasks.length} verbose`);
  console.log(`  ${negativeTasks.length} negative`);
  console.log(`  ${shiftTasks.length} shift`);
}

main();
