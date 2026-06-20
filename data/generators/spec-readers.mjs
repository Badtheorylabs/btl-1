import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { REASONING_TEMPLATES, NEGATIVE_TOOL_SIBLINGS } from "./reasoning-bank.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function readJsonl(path) {
  const text = readFileSync(path, "utf-8");
  if (!text.trim()) return [];
  return text.split(/\r?\n/).map(l => l.trim()).filter(Boolean).map(l => JSON.parse(l));
}

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

const TOOL_LABELS = {
  file_search: "search for files",
  read_file: "read file contents",
  write_file: "write to a file",
  email: "send an email",
  web_search: "search the web",
  browse: "open a web page",
  clipboard: "access the clipboard",
  shell_command: "run a shell command",
  notes: "manage notes",
  reasoning: "reason about a question",
};

/* ───── Reasoning wrapper helpers ───── */
function applyTemplate(depth, variant, vars) {
  const bank = REASONING_TEMPLATES[`depth-${depth}`];
  if (!bank || !bank[variant]) return { text: "Proceed.", id: "fallback" };
  const templates = bank[variant];
  const tpl = pick(templates);
  const text = tpl.fn(vars);
  return { text, id: tpl.id };
}

function pickSibling(tool) {
  const siblings = NEGATIVE_TOOL_SIBLINGS[tool];
  if (!siblings) return tool;
  return pick(siblings);
}

/* ───── Tool chain formatter ───── */
function formatChain(chain) {
  return JSON.stringify(chain);
}

function chainDesc(steps) {
  return steps.map(s => s.tool).join(" → ");
}

/* ───── Depth 0: coding ───── */
export function readCodingSpec() {
  const specPath = resolve(ROOT, "specs", "coding.json");
  const spec = JSON.parse(readFileSync(specPath, "utf-8"));
  const rows = [];

  const sysPrompt = (variant) => {
    if (variant === "minimal") return "You are a code generation teacher. Output only the function implementation — no explanations, no tests, no markdown formatting.";
    if (variant === "verbose") return "You are a code generation teacher. Output a well-commented, explicit, correct implementation with edge case handling.";
    return "You are a code generation teacher. Output a plausible-looking function with exactly one subtle bug. Make it convincing but wrong in the specified way.";
  };

  const userPrompt = (task, variant) => {
    if (variant === "minimal") return `Write a ${task.language} function: ${task.prompt}\n\nReturn ONLY the function implementation. No imports unless needed. No explanation, no comments, no tests. Shortest correct solution.`;
    if (variant === "verbose") return `Write a ${task.language} function: ${task.prompt}\n\nReturn the function with clear variable names, a short comment explaining the approach, and handle edge cases.`;
    return `Write a ${task.language} function: ${task.prompt}\n\nReturn a function that looks correct at first glance but has a subtle bug. The bug should be: ${task.neg_reason}. Make it plausible.`;
  };

  const solution = (task, variant) => {
    if (variant === "minimal") return task.minimal;
    if (variant === "verbose") return task.verbose || task.minimal;
    return task.negative || task.minimal;
  };

  for (const task of spec.tasks) {
    for (const variant of ["minimal", "verbose", "negative"]) {
      const sol = solution(task, variant);
      const { text: reasoningText, id: tempId } = applyTemplate(0, variant, { language: task.language, prompt: task.prompt, approach: "Define function, compute result, return it." });
      const assistantContent = `<reasoning>${reasoningText}</reasoning>\n${sol}`;

      rows.push({
        messages: [
          { role: "system", content: sysPrompt(variant) },
          { role: "user", content: userPrompt(task, variant) },
          { role: "assistant", content: assistantContent },
        ],
        provenance: {
          template_id: `${tempId}`,
          source_depth: 0,
          source_family: task.language,
          variant,
          api_or_template: "template",
          negative_type: variant === "negative" ? task.neg_reason : null,
          task_id: task.id,
        },
      });
    }
  }

  return rows;
}

/* ───── Depth 1: single tool ───── */
export function readDepth1Spec() {
  const rows = readJsonl(resolve(ROOT, "depth-1", "train.jsonl"));
  const result = [];

  const sysPrompt = `You are BTL-1, a local AI agent that executes tool chains. Given a user request, output a JSON array of tool calls. Use as few steps as possible. Use <reasoning> tags first, then the JSON array.`;

  for (const row of rows) {
    for (const variant of ["minimal", "verbose", "negative"]) {
      const tool = row.expected?.[0]?.tool || row.tool;
      const wrongTool = variant === "negative" ? pickSibling(tool) : tool;
      const chain = variant === "negative"
        ? [{ tool: wrongTool, params: { ...row.expected[0].params }, id: "step_1" }]
        : row.expected;

      const { text: reasoningText, id: tempId } = applyTemplate(1, variant, {
        utterance: row.utterance,
        tool_name: variant === "negative" ? wrongTool : tool,
        wrong_tool: wrongTool,
      });

      const assistantContent = `<reasoning>${reasoningText}</reasoning>\n${formatChain(chain)}`;

      result.push({
        messages: [
          { role: "system", content: sysPrompt },
          { role: "user", content: row.utterance },
          { role: "assistant", content: assistantContent },
        ],
        provenance: {
          template_id: `${tempId}`,
          source_depth: 1,
          source_family: tool,
          variant,
          api_or_template: "template",
          negative_type: variant === "negative" ? `wrong_tool:${wrongTool}` : null,
        },
      });
    }
  }

  return result;
}

/* ───── Depth 2: multi-step chains ───── */
export function readDepth2Spec() {
  const rows = readJsonl(resolve(ROOT, "depth-2", "train.jsonl"));
  const result = [];

  const sysPrompt = `You are BTL-1, a local AI agent that executes tool chains. Given a user request, output a JSON array of tool calls with dependencies. Use <reasoning> tags first, then the JSON array.`;

  for (const row of rows) {
    for (const variant of ["minimal", "verbose", "negative"]) {
      const steps = row.expected || [];
      const toolNames = steps.map(s => s.tool);
      const firstStep = steps[0];
      const secondStep = steps[1];

      let chain = steps;
      let wrongStep = "";
      let wrongOrder = "";
      let wrongDesc = "";
      let wrongTool = "";

      if (variant === "negative") {
        if (steps.length >= 2) {
          wrongTool = pickSibling(steps[1].tool);
          const wrongStepObj = { ...steps[1], tool: wrongTool };
          chain = [steps[0], wrongStepObj, ...steps.slice(2)];
          wrongStep = `${wrongTool} instead of ${steps[1].tool}`;
          wrongOrder = `${steps[0].tool} → ${wrongTool} → ${steps.slice(2).map(s => s.tool).join(" → ")}`;
          wrongDesc = chainDesc(chain);
        } else {
          chain = steps;
          wrongStep = `wrong approach`;
        }
      }

      const { text: reasoningText, id: tempId } = applyTemplate(2, variant, {
        utterance: row.utterance,
        chain_desc: chainDesc(steps),
        step_count: steps.length,
        first_step: firstStep ? `${firstStep.tool}(${JSON.stringify(firstStep.params)})` : "",
        second_step: secondStep ? `${secondStep.tool}(${JSON.stringify(secondStep.params)})` : "",
        third_step: steps[2] ? `${steps[2].tool}` : "",
        wrong_step: wrongStep,
        wrong_order: wrongOrder,
        wrong_desc: wrongDesc,
        wrong_tool: wrongTool,
      });

      const assistantContent = `<reasoning>${reasoningText}</reasoning>\n${formatChain(chain)}`;

      result.push({
        messages: [
          { role: "system", content: sysPrompt },
          { role: "user", content: row.utterance },
          { role: "assistant", content: assistantContent },
        ],
        provenance: {
          template_id: `${tempId}`,
          source_depth: 2,
          source_family: row.pattern || toolNames.join("_"),
          variant,
          api_or_template: "template",
          negative_type: variant === "negative" ? wrongStep : null,
        },
      });
    }
  }

  return result;
}

/* ───── Depth 3: code bugs ───── */
export function readDepth3Spec() {
  const result = [];

  const sysPrompt = `You are an expert coding agent that fixes bugs. Given a user request with buggy code, produce a tool chain that reads the file, applies the minimal correct fix, and writes the result. Output a JSON array of tool calls. Use <reasoning> tags first.`;

  for (const variant of ["minimal", "verbose", "negative"]) {
    const rows = readJsonl(resolve(ROOT, "depth-3", `${variant}.jsonl`));
    for (const row of rows) {
      const userPrompt = `Fix the bug described below.\n\nUser: ${row.utterance}\n\nFile contents:\n\`\`\`\n${row.buggy_code}\n\`\`\``;

      const fixedCode = variant === "negative" ? (row.negative_fix || row.buggy_code) : row.expected_fix;

      const fixDesc = variant === "negative"
        ? "near-miss wrong fix"
        : (row.causal_note || `fix the ${row.bug_type}`);

      const chain = [
        { tool: "read_file", params: { path: "buggy_file" }, id: "step_1" },
        { tool: "write_file", params: { path: "buggy_file", content: fixedCode }, depends_on: ["step_1"], id: "step_2" },
      ];

      const { text: reasoningText, id: tempId } = applyTemplate(3, variant, {
        utterance: row.utterance,
        language: row.language,
        bug_type: row.bug_type,
        fix_desc: fixDesc,
        causal_note: row.causal_note || "",
      });

      const assistantContent = `<reasoning>${reasoningText}</reasoning>\n${formatChain(chain)}`;

      result.push({
        messages: [
          { role: "system", content: sysPrompt },
          { role: "user", content: userPrompt },
          { role: "assistant", content: assistantContent },
        ],
        provenance: {
          template_id: `${tempId}`,
          source_depth: 3,
          source_family: row.bug_type,
          variant,
          api_or_template: "template",
          negative_type: variant === "negative" ? `${row.bug_type}:near-miss` : null,
          language: row.language,
          base_bug_id: row.base_bug_id || row.task_id || null,
          variant_context: row.variant_context || null,
          variant_index: row.variant_index ?? null,
          variant_id: row.variant_id || null,
        },
      });
    }
  }

  return result;
}

/* ───── Depth 4: repo repair ───── */
export function readDepth4Spec() {
  const result = [];
  const sysPrompt = `You are an expert coding agent for repo repair. Given a multi-file project with a bug, navigate the project and apply the fix across files. Output a JSON array of tool calls. Use <reasoning> tags first.`;

  for (const variant of ["minimal", "negative"]) {
    const rows = readJsonl(resolve(ROOT, "depth-4", `${variant}.jsonl`));
    for (const row of rows) {
      const fileList = (row.files || []).map(f => `  ${f.path}: ${(f.content || "").slice(0, 80).replace(/\n/g, " ")}`).join("\n");

      const userPrompt = [
        `Task: repo repair`,
        `Domain: ${row.domain}`,
        `User request: ${row.utterance}`,
        ``,
        `Project files:`,
        fileList,
        variant === "minimal"
          ? `\nInstructions:\nNavigate the project, find the root cause, apply the minimal fix.`
          : `\nInstructions:\nProduce a near-miss trace: make a change that looks plausible but does not address the actual cross-file dependency issue.`,
      ].join("\n");

      const chain = (row.expected_chain || [
        { tool: "file_search", params: { filename: "project" }, id: "step_1" },
        { tool: "read_file", params: { path: "$step_1.result.path" }, depends_on: ["step_1"], id: "step_2" },
        { tool: "write_file", params: { path: "$step_2.result", content: "fixed" }, depends_on: ["step_2"], id: "step_3" },
      ]);

      const { text: reasoningText, id: tempId } = applyTemplate(4, variant, {
        utterance: row.utterance,
        bug_type: row.domain || "unknown",
        file_count: (row.files || []).length,
        domain: row.domain,
      });

      const assistantContent = `<reasoning>${reasoningText}</reasoning>\n${formatChain(chain)}`;

      result.push({
        messages: [
          { role: "system", content: sysPrompt },
          { role: "user", content: userPrompt },
          { role: "assistant", content: assistantContent },
        ],
        provenance: {
          template_id: `${tempId}`,
          source_depth: 4,
          source_family: row.domain || "repo",
          variant,
          api_or_template: "template",
          negative_type: variant === "negative" ? "cross-file near-miss" : null,
          base_project_id: row.base_project_id || null,
          variant_context: row.variant_context || null,
          variant_index: row.variant_index ?? null,
          variant_id: row.variant_id || null,
        },
      });
    }
  }

  return result;
}

/* ───── API data reader (existing teacher completions) ───── */
export function readAPIData(completedPath) {
  const rows = readJsonl(completedPath);
  return rows.map((row, i) => {
    const assistantContent = row.assistant_content || row.messages?.find(m => m.role === "assistant")?.content || "";

    return {
      messages: [
        { role: "system", content: row.messages?.[0]?.content || "" },
        { role: "user", content: row.messages?.[1]?.content || "" },
        { role: "assistant", content: assistantContent },
      ],
      provenance: {
        template_id: `api-${row.depth || "?"}-${row.variant || "?"}-${String(i).padStart(6, "0")}`,
        source_depth: row.depth ?? -1,
        source_family: row.family || row.source_kind || "api",
        variant: row.variant || "unknown",
        api_or_template: "api",
        negative_type: row.variant === "negative" ? "api-generated" : null,
        source_id: row.id || null,
      },
    };
  });
}
