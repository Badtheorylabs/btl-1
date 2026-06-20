import path from "node:path";
import { promises as fs } from "node:fs";
import type { ExecutionContext, ReasoningModel, StepResult, ToolStep } from "./types.js";
import { createToolHandlers, resolveToolParams } from "./tools/index.js";

export interface ExecutorSummary {
  results: StepResult[];
}

export function parsePlan(raw: string): ToolStep[] {
  // Strip <reasoning> block if present
  const cleaned = raw.replace(/<reasoning>[\s\S]*?<\/reasoning>\s*/g, "").trim();
  const parsed = JSON.parse(cleaned);
  if (!Array.isArray(parsed)) {
    throw new Error("The model output must be a JSON array.");
  }
  return parsed.map((step, index) => {
    if (!step || typeof step !== "object") {
      throw new Error(`Step ${index + 1} is not an object.`);
    }
    const id = typeof step.id === "string" && step.id.trim() ? step.id : `step_${index + 1}`;
    const tool = step.tool;
    if (typeof tool !== "string") {
      throw new Error(`Step ${id} is missing a tool name.`);
    }
    return {
      id,
      tool: tool as ToolStep["tool"],
      params: step.params && typeof step.params === "object" ? step.params : {},
      depends_on: Array.isArray(step.depends_on) ? step.depends_on.map(String) : [],
    };
  });
}

export function createExecutionContext(projectRoot: string, model?: ReasoningModel): ExecutionContext {
  return {
    cwd: projectRoot,
    projectRoot,
    outboxDir: path.join(projectRoot, ".btl-outbox"),
    notesFile: path.join(projectRoot, ".btl-notes.json"),
    model,
    log: (line) => console.log(line),
  };
}

export async function executePlan(plan: ToolStep[], ctx: ExecutionContext): Promise<ExecutorSummary> {
  const handlers = createToolHandlers();
  const results = new Map<string, StepResult>();

  for (const step of plan) {
    for (const dep of step.depends_on || []) {
      if (!results.has(dep)) {
        throw new Error(`Step ${step.id} depends on missing step ${dep}.`);
      }
    }

    const handler = handlers[step.tool];
    if (!handler) {
      throw new Error(`Unknown tool: ${step.tool}`);
    }

    const params = resolveToolParams(step, results);
    const result = await handler(params, ctx, results);
    results.set(step.id, { id: step.id, tool: step.tool, result });
    ctx.log?.(`[${step.id}] ${step.tool} ok`);
  }

  return { results: [...results.values()] };
}

export async function readPlanFile(filePath: string): Promise<string> {
  return fs.readFile(filePath, "utf8");
}
