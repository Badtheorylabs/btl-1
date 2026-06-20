import { readFileSync, writeFileSync, appendFileSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const SPEC_DIR = resolve(ROOT, 'specs');
const OUT_DIR = resolve(ROOT, 'real-traces');

const DEEPSEEK_API_KEY = process.env.BTL_API_KEY || '';
const KEY_IS_OR = DEEPSEEK_API_KEY.startsWith('sk-or-v1-');
const DEEPSEEK_MODEL = KEY_IS_OR ? 'deepseek/deepseek-chat' : 'deepseek-v4-flash';
const DEEPSEEK_API_URL = KEY_IS_OR
  ? 'https://openrouter.ai/api/v1/chat/completions'
  : 'https://api.deepseek.com/chat/completions';
const OPENAI_API_KEY = process.env.BTL_OPENAI_API_KEY || process.env.OPENAI_API_KEY || '';
const OPENAI_MODEL = process.env.BTL_OPENAI_MODEL || 'gpt-4o-mini';
const NEGATIVE_MODEL = process.env.BTL_NEGATIVE_MODEL || OPENAI_MODEL;
const OPENAI_API_URL = process.env.BTL_OPENAI_API_URL || 'https://api.openai.com/v1/chat/completions';
const TASKS_PER_DEPTH = parseInt(process.env.BTL_TASKS || '2000', 10);
const CONCURRENCY = Math.max(1, parseInt(process.env.BTL_CONCURRENCY || '128', 10));
const MAX_TOKENS = Math.max(64, parseInt(process.env.BTL_MAX_TOKENS || '2048', 10));
const MAX_RETRIES = Math.max(0, parseInt(process.env.BTL_MAX_RETRIES || '5', 10));
const RETRY_BASE_MS = Math.max(50, parseInt(process.env.BTL_RETRY_BASE_MS || '500', 10));
const DEEPSEEK_PRICING = {
  promptHit: 0.0028,
  promptMiss: 0.14,
  output: 0.28,
};
const OPENAI_PRICING = {
  prompt: 0.15,
  output: 0.60,
};
const DRY_RUN = process.argv.includes('--dry-run');
const VARIANTS = ['verbose', 'negative'];
const FORCE_OPENAI = process.env.BTL_FORCE_OPENAI === '1';
const DEEPSEEK_BUDGET_DOLLARS = Number.isFinite(Number.parseFloat(process.env.BTL_DEEPSEEK_BUDGET_DOLLARS || process.env.BTL_BUDGET_DOLLARS || '1.5'))
  ? Number.parseFloat(process.env.BTL_DEEPSEEK_BUDGET_DOLLARS || process.env.BTL_BUDGET_DOLLARS || '1.5')
  : 0;
const VARIANT_MAX_TOKENS = {
  minimal: Math.max(64, parseInt(process.env.BTL_MINIMAL_MAX_TOKENS || '768', 10)),
  verbose: Math.max(64, parseInt(process.env.BTL_VERBOSE_MAX_TOKENS || '1536', 10)),
  negative: Math.max(64, parseInt(process.env.BTL_NEGATIVE_MAX_TOKENS || '2048', 10)),
};
const VARIANT_TEMPERATURE = {
  minimal: 0.2,
  verbose: 0.7,
  negative: 0.8,
};

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function fillUtterance(tmpl, pool) {
  return tmpl.replace(/\{(\w+)\}/g, (_, k) => pool[k] ? pick(pool[k]) : `{${k}}`);
}

function hashCode(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function loadSpecs() {
  const files = ['coding.json', 'depth-1.json', 'depth-2.json', 'depth-3.json', 'depth-4.json'];
  const all = [];
  const perDepth = TASKS_PER_DEPTH;

  for (const f of files) {
    const data = JSON.parse(readFileSync(resolve(SPEC_DIR, f), 'utf-8'));
    const depth = data.depth;
    const items = data.tasks || data.tools || data.chains || data.bugs || data.projects || [];

    if (depth === 0 || depth === 3 || depth === 4) {
      for (const item of items) {
        all.push({
          depth,
          id: item.id || item.bug || `proj-${item.id}`,
          prompt: item.prompt || item.utterance || item.description || item.name,
          extra: item.buggy_code ? `Buggy code:\n${item.buggy_code.slice(0, 1000)}` : null,
        });
      }
    } else if (depth === 1) {
      const tools = data.tools || [];
      const needed = perDepth;
      for (let n = 0; n < needed; n++) {
        const tool = pick(tools);
        const utterance = pick(tool.utterances || []);
        const filled = fillUtterance(utterance, tool.pool || {});
        const detail = `Tool: ${tool.tool}\nParams: ${JSON.stringify(Object.keys(tool.params || {}))}\nInstruction: ${filled}`;
        all.push({ depth, id: `${tool.tool}-${n}`, prompt: filled, extra: detail });
      }
    } else if (depth === 2) {
      const chains = data.chains || [];
      const needed = perDepth;
      for (let n = 0; n < needed; n++) {
        const chain = pick(chains);
        const utterance = pick(chain.utterances || []);
        const filled = fillUtterance(utterance, chain.pool || {});
        const stepList = (chain.output || []).map(s => `${s.tool} -> ${JSON.stringify(s.params)}`).join('\n');
        const detail = `Chain: ${chain.pattern}\nSteps:\n${stepList}\nInstruction: ${filled}`;
        all.push({ depth, id: `chain-${chain.pattern?.replace(/[^a-z]/gi,'')}-${n}`, prompt: filled, extra: detail });
      }
    }
  }

  return all;
}

async function generateTraces(tasks, trainPath, evalPath) {
  let deepseekCost = 0;
  let openaiCost = 0;
  let nextIndex = 0;
  let incompleteResponses = 0;
  let failedCalls = 0;
  let primaryBudgetReached = false;

  async function worker(workerId) {
    while (true) {
      const i = nextIndex++;
      if (i >= tasks.length) return;

      const task = tasks[i];
      for (const variant of VARIANTS) {
        const templateId = `${task.id}-${variant}`;
        let provider = (primaryBudgetReached || FORCE_OPENAI || !DEEPSEEK_API_KEY) ? 'openai' : 'deepseek';
        let triedDeepSeek = false;
        let success = false;

        while (!success) {
          try {
            const userMsg = buildUserMessage(task, variant);
            const modelOverride = (variant === 'negative' && NEGATIVE_MODEL !== OPENAI_MODEL) ? NEGATIVE_MODEL : null;
            const response = await callProvider(provider, userMsg, variant, modelOverride);
            if (provider === 'deepseek') {
              deepseekCost += response.cost;
            } else {
              openaiCost += response.cost;
            }
            const row = parseVariantResponse(response.text, task, variant);
            if (row) {
              const split = hashCode(templateId) % 10 === 0 ? 'eval' : 'train';
              const outPath = split === 'train' ? trainPath : evalPath;
              appendFileSync(outPath, JSON.stringify(row) + '\n');
            } else {
              incompleteResponses += 1;
            }

            if (!primaryBudgetReached && deepseekCost >= DEEPSEEK_BUDGET_DOLLARS) {
              primaryBudgetReached = true;
              console.warn(`  DeepSeek budget cap reached: $${deepseekCost.toFixed(4)} >= $${DEEPSEEK_BUDGET_DOLLARS.toFixed(4)}; switching to GPT-4o mini`);
            }
            success = true;
          } catch (err) {
            failedCalls += 1;
            console.error(`  task ${i + 1} variant ${variant} via ${provider} failed: ${err.message}`);

            if (provider === 'deepseek' && OPENAI_API_KEY) {
              triedDeepSeek = true;
              provider = 'openai';
              console.warn(`  falling back to GPT-4o mini for task ${i + 1} variant ${variant}`);
              continue;
            }

            break;
          }
        }
      }

      if ((i + 1) % 50 === 0) {
        const totalCost = deepseekCost + openaiCost;
        console.log(`  ${i + 1}/${tasks.length} tasks done, cost: $${totalCost.toFixed(4)} (ds: $${deepseekCost.toFixed(4)}, oai: $${openaiCost.toFixed(4)}), incomplete: ${incompleteResponses}, failed: ${failedCalls}`);
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, tasks.length) }, (_, i) => worker(i)));

  return { totalCost: deepseekCost + openaiCost, deepseekCost, openaiCost };
}

function buildSystemPrompt(variant) {
  if (variant === 'minimal') {
    return 'You are a coding assistant that gives only the final answer — no intro, no commentary, no markdown code fences. If the task asks for code, output the raw code. If it asks for a command, output the command. Never describe what you would do. Never use markdown formatting. Just the answer.';
  }
  if (variant === 'verbose') {
    return 'You are a coding assistant that solves tasks step by step. First think through the approach in a sentence or two, then output the solution. No markdown code fences. No meta-commentary about what you would do — actually do it. If the task asks for code, write the code. If it asks for a command, output the command.';
  }
  return 'You are a coding assistant. Output a solution that LOOKS correct but has one subtle bug. No disclaimers, no labels, no markdown. Just the answer.\n\nExamples of good subtle bugs:\n- Uses == instead of === in JavaScript\n- Forgets to handle empty array / null input\n- Off-by-one in loop boundary\n- Uses parseInt() without radix\n- Catches exception but does nothing (empty catch)\n- Mutates input array instead of returning a copy\n- Uses setTimeout(fn, 1000) instead of setInterval for recurring task\n- Uses `return` in forEach callback instead of filtering\n\nExample task: "Write a function to return unique elements from an array"\nBuggy: function unique(arr) { return [...new Set(arr)]; } — flaw: works for primitives, fails for array of objects (Set uses reference equality).\n\nImportant: the bug must be subtle. Not a syntax error or missing import. Something a tired reviewer would approve.';
}

function buildUserMessage(task, variant) {
  let parts = [task.prompt];
  if (task.extra) parts.push(`\nContext:\n${task.extra}`);
  return parts.join('\n');
}

function getProviderConfig(provider) {
  if (provider === 'openai') {
    if (!OPENAI_API_KEY) {
      throw new Error('OpenAI API key missing. Set OPENAI_API_KEY or BTL_OPENAI_API_KEY.');
    }
    return {
      name: 'openai',
      apiKey: OPENAI_API_KEY,
      apiUrl: OPENAI_API_URL,
      model: OPENAI_MODEL,
    };
  }

  if (!DEEPSEEK_API_KEY) {
    throw new Error('DeepSeek API key missing. Set BTL_API_KEY.');
  }

  return {
    name: 'deepseek',
    apiKey: DEEPSEEK_API_KEY,
    apiUrl: DEEPSEEK_API_URL,
    model: DEEPSEEK_MODEL,
  };
}

function estimateCost(provider, data) {
  if (provider === 'openai') {
    const prompt = data.usage?.prompt_tokens || 0;
    const output = data.usage?.completion_tokens || 0;
    return (prompt * OPENAI_PRICING.prompt + output * OPENAI_PRICING.output) / 1_000_000;
  }

  let cost = data.usage?.total_cost || 0;
  if (cost === 0 && data.usage) {
    const promptHit = data.usage.prompt_cache_hit_tokens || 0;
    const promptMiss = data.usage.prompt_cache_miss_tokens ?? Math.max(0, (data.usage.prompt_tokens || 0) - promptHit);
    const output = data.usage.completion_tokens || 0;
    cost = (
      promptHit * DEEPSEEK_PRICING.promptHit +
      promptMiss * DEEPSEEK_PRICING.promptMiss +
      output * DEEPSEEK_PRICING.output
    ) / 1_000_000;
  }
  return cost;
}

async function callProvider(provider, userMsg, variant, modelOverride) {
  const cfg = getProviderConfig(provider);
  const body = {
    model: modelOverride || cfg.model,
    messages: [
      { role: 'system', content: buildSystemPrompt(variant) },
      { role: 'user', content: userMsg },
    ],
    temperature: VARIANT_TEMPERATURE[variant] ?? 0.7,
    max_tokens: VARIANT_MAX_TOKENS[variant] ?? MAX_TOKENS,
  };

  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${cfg.apiKey}`,
  };
  if (provider === 'deepseek' && KEY_IS_OR) headers['HTTP-Referer'] = 'https://github.com/anomalyco/opencode';
  let lastErr = null;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(cfg.apiUrl, {
        method: 'POST', headers, body: JSON.stringify(body),
      });

      if (!res.ok) {
        const errText = await res.text();
        const err = new Error(`API error ${res.status}: ${errText}`);
        err.status = res.status;
        throw err;
      }

      const data = await res.json();
      const text = data.choices?.[0]?.message?.content || '';
      const cost = estimateCost(provider, data);
      return { text, cost };
    } catch (err) {
      lastErr = err;
      const status = err?.status;
      const retryable = status === 429 || status === 500 || status === 503 || !status;
      if (!retryable || attempt === MAX_RETRIES) {
        throw err;
      }
      const delay = RETRY_BASE_MS * (2 ** attempt) + Math.floor(Math.random() * RETRY_BASE_MS);
      console.warn(`  retrying ${variant} after ${delay}ms (attempt ${attempt + 1}/${MAX_RETRIES})`);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastErr || new Error(`${provider} request failed`);
}

function parseVariantResponse(text, task, variant) {
  let content = text.trim();
  content = content.replace(/<\/?im_start>/g, '').replace(/<\/?im_end>/g, '').trim();
  content = content.replace(/^(Here's|Here is|The |I'll |I would |You can |For this |To )/i, '').trim();
  content = content.replace(/^the (shortest|correct|best|simplest) (answer|way|solution|approach) (is|would be):?\s*/i, '').trim();
  content = content.replace(/^"|"$/g, '').trim();

  if (!content || content.length < 5) return null;

  return {
    messages: [
      { role: 'system', content: 'You are a helpful coding assistant.' },
      { role: 'user', content: task.prompt },
      { role: 'assistant', content },
    ],
    provenance: {
      template_id: `${task.id}-${variant}`,
      source_depth: task.depth,
      source_family: task.id.split('-')[0] || task.id,
      variant,
      api_or_template: 'api',
      negative_type: variant === 'negative' ? 'counterfactual' : null,
    },
  };
}

async function main() {
  console.log('Loading and expanding specs...');
  const tasks = loadSpecs();
  console.log(`  ${tasks.length} total tasks (~${tasks.length * 3} rows)`);

  const counts = {};
  for (const t of tasks) counts[t.depth] = (counts[t.depth] || 0) + 1;
  for (const d of Object.keys(counts).sort()) console.log(`  depth ${d}: ${counts[d]} tasks`);

  if (DRY_RUN) {
    console.log('\nDRY RUN — showing first task:');
    console.log(JSON.stringify(tasks[0], null, 2));
    console.log(`\nDry-run OK. Set BTL_API_KEY and remove --dry-run to generate.`);
    return;
  }

  if (!DEEPSEEK_API_KEY && !FORCE_OPENAI) {
    console.error('Set BTL_API_KEY=<your DeepSeek/OpenRouter API key> or BTL_FORCE_OPENAI=1');
    process.exit(1);
  }

  mkdirSync(OUT_DIR, { recursive: true });
  const trainPath = resolve(OUT_DIR, 'train.jsonl');
  const evalPath = resolve(OUT_DIR, 'eval.jsonl');

  // Wipe existing files so we start fresh
  writeFileSync(trainPath, '');
  writeFileSync(evalPath, '');

  const providerLabel = FORCE_OPENAI ? `GPT-4o mini (DeepSeek disabled via BTL_FORCE_OPENAI)` : `${KEY_IS_OR ? 'OpenRouter DeepSeek proxy' : 'DeepSeek V4 Flash'} with GPT-4o mini fallback`;
  console.log(`\nGenerating real traces via ${providerLabel}...`);
  console.log(`  concurrency=${CONCURRENCY}, max_tokens=${MAX_TOKENS}${FORCE_OPENAI ? '' : `, deepseek_budget=$${DEEPSEEK_BUDGET_DOLLARS.toFixed(4)}`}${OPENAI_API_KEY ? `, model=${OPENAI_MODEL}` : ' (no OpenAI key set)'}`);
  console.log(`  saving incrementally to:\n    train: ${trainPath}\n    eval:  ${evalPath}`);
  const { totalCost, deepseekCost, openaiCost } = await generateTraces(tasks, trainPath, evalPath);

  // Count final rows
  const trainText = readFileSync(trainPath, 'utf-8').trim();
  const evalText = readFileSync(evalPath, 'utf-8').trim();
  const trainCount = trainText ? trainText.split('\n').length : 0;
  const evalCount = evalText ? evalText.split('\n').length : 0;
  const totalRows = trainCount + evalCount;

  console.log(`\nDone. Generated ${totalRows} rows at $${totalCost.toFixed(4)} total (deepseek $${deepseekCost.toFixed(4)}, openai $${openaiCost.toFixed(4)})`);
  console.log(`  train: ${trainPath} (${trainCount} rows)`);
  console.log(`  eval:  ${evalPath} (${evalCount} rows)`);
}

main().catch(err => { console.error(err); process.exit(1); });
