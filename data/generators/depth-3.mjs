import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  slugify,
  replaceWords,
  extractFunctionName,
  extractParams,
  extractIdentifiers,
} from "./variant-utils.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SPEC_PATH = resolve(ROOT, "specs", "depth-3.json");
const OUTPUT_DIR = resolve(ROOT, "depth-3");

const DEFAULT_CONTEXTS = [
  "billing",
  "checkout",
  "analytics",
  "support",
  "inventory",
  "platform",
  "onboarding",
  "payments",
];

const PARAM_POOL = [
  "value",
  "item",
  "entry",
  "record",
  "payload",
  "target",
  "source",
  "count",
  "total",
  "result",
  "node",
  "index",
];

const SAFE_EXTRA_IDENTIFIERS = new Set([
  "total",
  "count",
  "result",
  "item",
  "items",
  "value",
  "values",
  "record",
  "records",
  "entry",
  "entries",
  "row",
  "rows",
  "user",
  "obj",
  "lst",
  "arr",
  "data",
  "price",
  "prices",
  "quantity",
  "quantities",
  "email",
  "name",
  "greeting",
  "task",
  "tasks",
  "payload",
  "target",
  "source",
  "state",
  "limit",
  "window",
  "page",
  "index",
]);

const SYSTEM_PROMPT = [
  "You are an expert coding agent that fixes bugs by reading and editing files.",
  "Given a user request with buggy code, produce a tool chain that reads the file, applies the minimal correct fix, and writes the result.",
  "Output a JSON array of tool calls.",
  "Each tool call has: tool name, params object, unique id, and depends_on if referencing prior steps.",
  "Use $step_N.result to reference outputs from earlier steps.",
  "No reasoning tags. No extra text.",
  "The fix must be minimal - change only what is needed to fix the causal failure.",
].join("\n");

function buildMinimalPrompt(utterance, code) {
  return `Fix the bug described below. Return the minimal tool chain.\n\nUser: ${utterance}\n\nFile contents:\n\`\`\`\n${code}\n\`\`\``;
}

function buildVerbosePrompt(utterance, code) {
  return `Fix the bug described below. Show your full reasoning before making the edit.\n\nUser: ${utterance}\n\nFile contents:\n\`\`\`\n${code}\n\`\`\``;
}

function buildNegativePrompt(utterance, code) {
  return `Fix the bug described below.\n\nUser: ${utterance}\n\nFile contents:\n\`\`\`\n${code}\n\`\`\``;
}

function buildCounterfactualPrompt(utterance, code, proposedFix) {
  return [
    "Review this proposed fix for the bug below. Does it actually fix the causal failure, or does it only look correct? Explain your reasoning, then confirm or reject.",
    "",
    `User: ${utterance}`,
    "",
    "File contents:",
    "```",
    code,
    "```",
    "",
    "Proposed fix:",
    "```",
    proposedFix,
    "```",
  ].join("\n");
}

function pickContext(spec, index) {
  const contexts = spec.generation?.variant_contexts?.length
    ? spec.generation.variant_contexts
    : DEFAULT_CONTEXTS;
  return contexts[index % contexts.length];
}

function makeVariantTag(bug, context, index) {
  return `${slugify(bug.id)}_${slugify(context)}_${String(index + 1).padStart(3, "0")}`;
}

function buildReplacementMap(bug, context, index) {
  const variantTag = makeVariantTag(bug, context, index);
  const replacements = {};

  const fnName = extractFunctionName(bug.buggy_code) || extractFunctionName(bug.expected_fix) || bug.id;
  replacements[fnName] = `${fnName}_${variantTag}`;

  const params = extractParams(bug.buggy_code);
  params.slice(0, 3).forEach((param, paramIndex) => {
    replacements[param] = `${PARAM_POOL[(index + paramIndex) % PARAM_POOL.length]}_${variantTag}`;
  });

  const extras = extractIdentifiers(`${bug.buggy_code}\n${bug.expected_fix}`)
    .filter((token) => SAFE_EXTRA_IDENTIFIERS.has(token))
    .filter((token) => token !== fnName && !params.includes(token))
    .slice(0, 3);

  extras.forEach((token, extraIndex) => {
    if (!replacements[token]) {
      replacements[token] = `${token}_${variantTag}_${extraIndex + 1}`;
    }
  });

  return { replacements, variantTag };
}

function applyVariant(text, replacements) {
  return replaceWords(text, replacements);
}

function genericNearMiss(fixText) {
  const attempts = [
    [[/>=/g, ">"], [/\breturn\s+True\b/g, "return False"], [/\breturn\s+False\b/g, "return True"]],
    [[/<=/g, "<"], [/\breturn\s+None\b/g, "return 0"], [/\breturn\s+0\b/g, "return None"]],
    [[/==/g, "!="], [/!=/g, "=="]],
    [[/\b0\b/g, "1"]],
    [[/\b1\b/g, "0"]],
  ];

  for (const group of attempts) {
    let mutated = fixText;
    let changed = false;
    for (const [pattern, replacement] of group) {
      const next = mutated.replace(pattern, replacement);
      if (next !== mutated) {
        changed = true;
        mutated = next;
      }
    }
    if (changed && mutated !== fixText) {
      return mutated;
    }
  }

  return `${fixText}\n# near-miss`;
}

function generateNearMissFix(bug) {
  const known = {
    "py-off-by-one-01": "def sum_to_n(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total - 1",
    "py-null-ref-01": "def get_user_email(user):\n    if user is None:\n        return ''\n    return user['email']",
    "py-wrong-operator-01": "def apply_discount(price, item_count):\n    if item_count > 100:\n        return price * 0.9\n    return price",
    "py-missing-edge-01": "def divide(a, b):\n    try:\n        return a / b\n    except:\n        return float('inf')",
    "py-wrong-variable-01": "def calculate_total(prices, quantities):\n    total = 0\n    for i in range(len(quantities)):\n        total += quantities[i] * quantities[i]\n    return total",
    "py-logic-error-01": "def is_even(n):\n    return n % 2 == 0 or n == 0",
    "py-mutation-01": "def append_to_list(item, lst=[]):\n    lst.append(item)\n    return lst.copy()",
    "py-wrong-return-01": "def first_element(lst):\n    if not lst:\n        return 0\n    return lst[0]",
    "py-type-confusion-01": "def concat(a, b):\n    if isinstance(a, str) and isinstance(b, str):\n        return a + b\n    return None",
    "py-wrong-default-01": "def greet(name, greeting='Hi'):\n    if greeting == 'Hi':\n        greeting = 'Hello'\n    return f'{greeting}, {name}'",
    "js-off-by-one-01": "function sumArray(arr) {\n    let total = 0;\n    for (let i = 0; i < arr.length - 1; i++) {\n        total += arr[i];\n    }\n    return total + arr[arr.length - 1];\n}",
    "js-null-ref-01": "function getName(obj) {\n    if (obj === null) return 'unknown';\n    return obj.name;\n}",
  };

  const fix = known[bug.id] || bug.expected_fix;
  return genericNearMiss(fix);
}

function buildVariantRow(bug, variant, context, index) {
  const { replacements, variantTag } = buildReplacementMap(bug, context, index);
  const utterance = applyVariant(`In the ${context} variant, ${bug.utterance}`, replacements);
  const buggyCode = applyVariant(bug.buggy_code, replacements);
  const expectedFix = applyVariant(bug.expected_fix, replacements);
  const test = applyVariant(bug.test, replacements);
  const causalNote = applyVariant(bug.causal_note, replacements);
  const nearMiss = applyVariant(generateNearMissFix(bug), replacements);
  const systemPrompt = SYSTEM_PROMPT;

  const base = {
    kind: "depth-3",
    variant,
    language: bug.language,
    bug_type: bug.bug_type,
    utterance,
    buggy_code: buggyCode,
    expected_fix: expectedFix,
    test,
    system_prompt: systemPrompt,
    base_bug_id: bug.id,
    variant_context: context,
    variant_index: index,
    variant_id: variantTag,
    causal_note: causalNote,
  };

  if (variant === "minimal") {
    return {
      ...base,
      teacher_prompt: buildMinimalPrompt(utterance, buggyCode),
    };
  }

  if (variant === "verbose") {
    return {
      ...base,
      teacher_prompt: buildVerbosePrompt(utterance, buggyCode),
    };
  }

  if (variant === "negative") {
    return {
      ...base,
      negative_fix: nearMiss,
      teacher_prompt: buildNegativePrompt(utterance, buggyCode),
    };
  }

  return {
    ...base,
    proposed_fix: nearMiss,
    teacher_prompt: buildCounterfactualPrompt(utterance, buggyCode, nearMiss),
  };
}

function writeJsonl(path, rows) {
  writeFileSync(path, rows.map((row) => JSON.stringify(row)).join("\n"));
  console.log(`  ${path}: ${rows.length} rows`);
}

function main() {
  const spec = JSON.parse(readFileSync(SPEC_PATH, "utf-8"));
  mkdirSync(OUTPUT_DIR, { recursive: true });

  const perBug = spec.generation?.per_bug ?? 40;
  const variants = ["minimal", "verbose", "negative", "counterfactual"];

  const minimalTasks = [];
  const verboseTasks = [];
  const negativeTasks = [];
  const counterfactualTasks = [];

  for (const bug of spec.bugs) {
    for (let index = 0; index < perBug; index++) {
      const context = pickContext(spec, index);
      minimalTasks.push(buildVariantRow(bug, "minimal", context, index));
      verboseTasks.push(buildVariantRow(bug, "verbose", context, index));
      negativeTasks.push(buildVariantRow(bug, "negative", context, index));
      counterfactualTasks.push(buildVariantRow(bug, "counterfactual", context, index));
    }
  }

  writeJsonl(resolve(OUTPUT_DIR, "minimal.jsonl"), minimalTasks);
  writeJsonl(resolve(OUTPUT_DIR, "verbose.jsonl"), verboseTasks);
  writeJsonl(resolve(OUTPUT_DIR, "negative.jsonl"), negativeTasks);
  writeJsonl(resolve(OUTPUT_DIR, "counterfactual.jsonl"), counterfactualTasks);

  const shiftDir = resolve(OUTPUT_DIR, "shifts");
  mkdirSync(shiftDir, { recursive: true });
  writeJsonl(resolve(shiftDir, "shift-language.jsonl"), minimalTasks.map((row) => ({ ...row, shift: "language", original_language: row.language })));
  writeJsonl(resolve(shiftDir, "shift-variable-names.jsonl"), minimalTasks.map((row) => ({ ...row, shift: "variable_names" })));
  writeJsonl(resolve(shiftDir, "shift-formatting.jsonl"), minimalTasks.map((row) => ({ ...row, shift: "formatting" })));

  console.log("\ndepth-3 totals:");
  console.log(`  ${minimalTasks.length} minimal`);
  console.log(`  ${verboseTasks.length} verbose`);
  console.log(`  ${negativeTasks.length} negative`);
  console.log(`  ${counterfactualTasks.length} counterfactual`);
  console.log(`  contexts: ${spec.generation?.variant_contexts?.length || DEFAULT_CONTEXTS.length}`);
  console.log(`  variants per bug: ${perBug}`);
}

main();
