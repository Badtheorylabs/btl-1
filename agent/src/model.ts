import { spawn } from "node:child_process";
import { ToolStep, ReasoningModel } from "./types.js";

export interface LlamaConfig {
  binary: string;
  modelPath: string;
  maxTokens: number;
  temperature: number;
  topP: number;
}

export function loadLlamaConfig(env = process.env): LlamaConfig {
  const modelPath = env.BTL_MODEL_PATH?.trim();
  if (!modelPath) {
    throw new Error("BTL_MODEL_PATH is required for local GGUF inference.");
  }

  return {
    binary: env.BTL_MODEL_BIN?.trim() || "llama-cli",
    modelPath,
    maxTokens: Number(env.BTL_MODEL_MAX_TOKENS || 512),
    temperature: Number(env.BTL_MODEL_TEMPERATURE || 0.2),
    topP: Number(env.BTL_MODEL_TOP_P || 0.9),
  };
}

function collect(proc: ReturnType<typeof spawn>): Promise<string> {
  return new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";

    proc.stdout?.on("data", (chunk) => (stdout += String(chunk)));
    proc.stderr?.on("data", (chunk) => (stderr += String(chunk)));
    proc.on("error", reject);
    proc.on("close", (code) => {
      if (code && code !== 0) {
        reject(new Error(stderr.trim() || `Model process exited with code ${code}`));
        return;
      }
      resolve(stdout.trim());
    });
  });
}

export class LlamaCliModel implements ReasoningModel {
  constructor(private readonly config: LlamaConfig) {}

  complete(prompt: string): Promise<string> {
    const args = [
      "-m",
      this.config.modelPath,
      "-p",
      prompt,
      "-n",
      String(this.config.maxTokens),
      "--temp",
      String(this.config.temperature),
      "--top-p",
      String(this.config.topP),
    ];

    return collect(spawn(this.config.binary, args, { stdio: ["ignore", "pipe", "pipe"] }));
  }
}

export function buildPlanPrompt(userPrompt: string): string {
  const system = [
    "You are BTL-1, a local AI agent that executes tool chains.",
    "Given a user request, first think step-by-step inside <reasoning> tags.",
    "Then output a JSON array of tool calls.",
    "Each tool call has: tool name, params object, and optional depends_on array referencing earlier steps by id.",
    "Use $step_{id}.result to reference outputs.",
    "",
    "Available tools:",
    "- file_search: { filename, ext?, time?, folder? }",
    "- read_file: { path }",
    "- write_file: { path, content }",
    "- email: { to, subject, body, attachments? }",
    "- web_search: { query }",
    "- browse: { url }",
    "- clipboard: { action: \"read\" | \"write\", content? }",
    "- shell_command: { command }",
    "- notes: { action: \"read\" | \"write\" | \"append\", title, content? }",
    "- reasoning: { question, context? }",
    "",
    "Rules:",
    "- Use as few steps as possible. Never skip required params.",
    "- For multi-step chains, each step references prior outputs with $step_id.result.",
    "- If no tools apply, output: { \"tool\": \"reasoning\", \"params\": { \"question\": \"...\" } }",
    "- First <reasoning> then the JSON array. No extra text after the JSON.",
  ].join("\n");

  return `${system}\n\nUser request: ${userPrompt}`;
}

export function createModelFromEnv(env = process.env): ReasoningModel {
  return new LlamaCliModel(loadLlamaConfig(env));
}
