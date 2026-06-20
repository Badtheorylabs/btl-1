export type ToolName =
  | "file_search"
  | "read_file"
  | "write_file"
  | "email"
  | "web_search"
  | "browse"
  | "clipboard"
  | "shell_command"
  | "notes"
  | "reasoning";

export interface ToolStep {
  id: string;
  tool: ToolName;
  params?: Record<string, unknown>;
  depends_on?: string[];
}

export interface StepResult {
  id: string;
  tool: ToolName;
  result: unknown;
}

export interface ReasoningModel {
  complete(prompt: string): Promise<string>;
}

export interface ExecutionContext {
  cwd: string;
  projectRoot: string;
  outboxDir: string;
  notesFile: string;
  model?: ReasoningModel;
  log?: (line: string) => void;
}

export type ToolHandler = (
  params: Record<string, unknown>,
  ctx: ExecutionContext,
  results: Map<string, StepResult>,
) => Promise<unknown>;
