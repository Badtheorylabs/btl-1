import { readFileSync, appendFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const TRAIN_PATH = resolve(ROOT, 'real-traces', 'train.jsonl');
const EVAL_PATH = resolve(ROOT, 'real-traces', 'eval.jsonl');
const LOG_PATH = resolve(ROOT, 'real-traces', 'negatives.log');

const API_KEY = process.env.BTL_OPENAI_API_KEY || process.env.OPENAI_API_KEY || '';
const MODEL = process.env.BTL_NEGATIVE_MODEL || 'gpt-5.4-mini';
const API_URL = 'https://api.openai.com/v1/chat/completions';
const CONCURRENCY = Math.max(1, parseInt(process.env.BTL_CONCURRENCY || '32', 10));
const MAX_TOKENS = 1536;
const TEMPERATURE = 0.8;
const MAX_RETRIES = 3;
const RETRY_BASE_MS = 500;
const LIMIT = parseInt(process.env.BTL_NEGATIVE_LIMIT || '5000', 10);

if (!API_KEY) {
  console.error('Set BTL_OPENAI_API_KEY');
  process.exit(1);
}

const SYSTEM_PROMPT = `You are a coding assistant. Output a solution that LOOKS correct but has one subtle bug. No disclaimers, no labels, no markdown. Just the answer.

Examples of good subtle bugs:
- Uses == instead of === in JavaScript
- Forgets to handle empty array / null input
- Off-by-one in loop boundary
- Uses parseInt() without radix
- Catches exception but does nothing (empty catch)
- Mutates input array instead of returning a copy
- Uses setTimeout(fn, 1000) instead of setInterval for recurring task
- Uses return in forEach callback instead of filtering

Example task: "Write a function to return unique elements from an array"
Buggy: function unique(arr) { return [...new Set(arr)]; } — flaw: works for primitives, fails for array of objects (Set uses reference equality).

Important: the bug must be subtle. Not a syntax error or missing import. Something a tired reviewer would approve.`;

function loadRows(path) {
  const text = readFileSync(path, 'utf-8').trim();
  if (!text) return [];
  return text.split('\n').map(line => JSON.parse(line));
}

async function callGPT(prompt) {
  const body = {
    model: MODEL,
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: prompt },
    ],
    temperature: TEMPERATURE,
    max_completion_tokens: MAX_TOKENS,
  };

  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`,
  };

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(API_URL, {
        method: 'POST', headers, body: JSON.stringify(body),
      });
      if (!res.ok) {
        const errText = await res.text();
        const err = new Error(`API error ${res.status}: ${errText}`);
        err.status = res.status;
        throw err;
      }
      const data = await res.json();
      const content = (data.choices?.[0]?.message?.content || '').trim();
      const cost = (data.usage?.prompt_tokens || 0) * 0.75 / 1_000_000
        + (data.usage?.completion_tokens || 0) * 4.50 / 1_000_000;
      return { content, cost };
    } catch (err) {
      const retryable = err?.status === 429 || err?.status === 500 || err?.status === 503 || !err?.status;
      if (!retryable || attempt === MAX_RETRIES) throw err;
      await new Promise(r => setTimeout(r, RETRY_BASE_MS * (2 ** attempt)));
    }
  }
  throw new Error('Request failed');
}

function makeNegativeRow(original, instruction, content) {
  let cleaned = content
    .replace(/^<variant:[^>]+>\s*/i, '')
    .replace(/^(Here's|Here is|The |I'll |I would |You can |For this |To )/i, '')
    .trim();

  if (!cleaned || cleaned.length < 5) return null;

  return {
    messages: [
      { role: 'system', content: 'You are a helpful coding assistant.' },
      { role: 'user', content: instruction },
      { role: 'assistant', content: cleaned },
    ],
    provenance: {
      template_id: `${original.provenance.template_id}-negative`,
      source_depth: original.provenance.source_depth,
      source_family: original.provenance.source_family,
      variant: 'negative',
      api_or_template: MODEL,
      negative_type: 'counterfactual',
    },
  };
}

async function main() {
  console.log(`Loading verbose rows...`);
  const trainRows = loadRows(TRAIN_PATH);
  const evalRows = loadRows(EVAL_PATH);
  const allRows = [...trainRows, ...evalRows];
  const verboseRows = allRows.filter(r => r.provenance?.variant === 'verbose');
  console.log(`  ${verboseRows.length} verbose rows found (limit=${LIMIT})`);

  const toProcess = verboseRows.slice(0, LIMIT);
  let cost = 0;
  let done = 0;
  let failed = 0;
  let skipped = 0;
  let nextIndex = 0;

  const log = (msg) => {
    const text = `[${new Date().toISOString()}] ${msg}\n`;
    appendFileSync(LOG_PATH, text);
  };

  appendFileSync(LOG_PATH, `=== Starting negative generation with ${MODEL} ===\n`);
  appendFileSync(LOG_PATH, `Total to process: ${toProcess.length}\n`);

  async function worker() {
    while (true) {
      const i = nextIndex++;
      if (i >= toProcess.length) return;

      const row = toProcess[i];
      const instruction = row.messages?.[1]?.content;
      if (!instruction) { skipped++; continue; }

      try {
        const result = await callGPT(instruction);
        cost += result.cost;

        const negRow = makeNegativeRow(row, instruction, result.content);
        if (negRow) {
          const isEval = Math.abs(hash(negRow.provenance.template_id)) % 10 === 0;
          const outPath = isEval ? EVAL_PATH : TRAIN_PATH;
          appendFileSync(outPath, JSON.stringify(negRow) + '\n');
          done++;
        } else {
          skipped++;
        }
      } catch (err) {
        failed++;
        log(`Row ${i} failed: ${err.message}`);
      }

      if ((i + 1) % 100 === 0) {
        const msg = `${i + 1}/${toProcess.length} done, cost: $${cost.toFixed(4)}, success: ${done}, failed: ${failed}, skipped: ${skipped}`;
        console.log(`  ${msg}`);
        log(msg);
      }
    }
  }

  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));

  console.log(`\nDone. Generated ${done} negatives at $${cost.toFixed(4)} (${failed} failed, ${skipped} skipped)`);
  log(`Done. Generated ${done} negatives at $${cost.toFixed(4)} (${failed} failed, ${skipped} skipped)`);
}

function hash(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h) + str.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h);
}

main().catch(err => { console.error(err); process.exit(1); });
