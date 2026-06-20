import path from "node:path";
import { promises as fs } from "node:fs";
import { buildPlanPrompt, createModelFromEnv } from "./model.js";
import { createExecutionContext, executePlan, parsePlan, readPlanFile } from "./executor.js";

function readArg(flag: string, argv: string[]): string | undefined {
  const idx = argv.indexOf(flag);
  if (idx === -1) return undefined;
  return argv[idx + 1];
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const root = process.env.BTL_PROJECT_ROOT?.trim() || path.resolve(process.cwd(), "..");
  const prompt = readArg("--prompt", argv) || argv.filter((arg) => !arg.startsWith("--")).join(" ");
  const planArg = readArg("--plan", argv);
  const dryRun = argv.includes("--dry-run");
  const model = planArg ? undefined : (argv.includes("--no-model") ? undefined : createModelFromEnv(process.env));
  const ctx = createExecutionContext(root, model);

  let planSource = "";
  if (planArg) {
    planSource = await loadPlanSource(planArg);
  } else {
    if (!prompt.trim()) {
      throw new Error("Pass --prompt, --plan, or a freeform prompt argument.");
    }
    if (!model) {
      throw new Error("A model is required when generating a plan from a prompt.");
    }
    planSource = await model.complete(buildPlanPrompt(prompt));
  }

  const plan = parsePlan(planSource);
  if (dryRun) {
    console.log(JSON.stringify(plan, null, 2));
    return;
  }

  const summary = await executePlan(plan, ctx);
  console.log(JSON.stringify(summary, null, 2));
}

async function loadPlanSource(arg: string): Promise<string> {
  const maybeFile = path.resolve(process.cwd(), arg);
  if (await exists(maybeFile)) return readPlanFile(maybeFile);
  return arg;
}

async function exists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
