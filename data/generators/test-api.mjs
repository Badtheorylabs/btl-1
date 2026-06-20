import { readFileSync } from 'fs';

const API_KEY = process.env.BTL_OPENAI_API_KEY || '';
const MODEL = 'gpt-5.4-mini';

async function main() {
  const line = readFileSync('C:/Users/pc/Downloads/btl/btl-1/data/real-traces/train.jsonl', 'utf-8').split('\n')[0];
  const row = JSON.parse(line);
  const instruction = row.messages[1].content.substring(0, 200);

  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${API_KEY}` },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: 'system', content: 'Return a short answer.' },
        { role: 'user', content: instruction },
      ],
      max_tokens: 100,
    }),
  });

  const data = await res.json();
  if (!res.ok) {
    console.error(`Status ${res.status}:`, JSON.stringify(data));
    return;
  }
  console.log('Success:', data.choices?.[0]?.message?.content?.substring(0, 200));
  console.log('Tokens:', data.usage?.total_tokens);
}

main().catch(e => console.error(e.message));
