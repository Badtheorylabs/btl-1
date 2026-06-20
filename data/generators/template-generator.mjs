import { readFileSync, mkdirSync, createWriteStream } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  readCodingSpec, readDepth1Spec, readDepth2Spec, readDepth3Spec, readDepth4Spec, readAPIData,
} from "./spec-readers.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA = resolve(__dirname, "..");
const RAW = resolve(DATA, "raw");
const FINAL = resolve(DATA, "final");
mkdirSync(RAW, { recursive: true });
mkdirSync(FINAL, { recursive: true });

const SYSTEM_PROMPT = `You are BTL-1, a local AI agent that executes tool chains. Given a user request, output a JSON array of tool calls. Each tool call has: tool name, params object, and optional depends_on array referencing earlier steps by id. Use $step_{id}.result to reference outputs.

Available tools:
- file_search: { filename, ext?, time?, folder? }
- read_file: { path }
- write_file: { path, content }
- email: { to, subject, body, attachments? }
- web_search: { query }
- browse: { url }
- clipboard: { action: "read" | "write", content? }
- shell_command: { command }
- notes: { action: "read" | "write" | "append", title, content? }
- reasoning: { question, context? }

Rules:
- Use as few steps as possible. Never skip required params.
- For multi-step chains, each step references prior outputs with $step_id.result.
- If no tools apply, output: { "tool": "reasoning", "params": { "question": "..." } }
- First, think step-by-step inside <reasoning> tags.
- Then output the JSON array. No extra text after the JSON.`;

/* ───── helpers ───── */
function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

/* ───── value pools ───── */
const V = {
  filename: ["budget", "project_plan", "invoice", "report", "notes", "schedule", "proposal", "meeting_notes", "q2_report", "q3_report", "annual_review", "contract", "resume", "timesheet", "task_list", "roadmap", "spec", "design_doc", "research_paper", "dataset", "backup", "config", "log", "archive", "presentation", "readme", "summary", "minutes", "draft", "outline", "checklist", "inventory", "ledger", "forecast", "budget_v2", "strategy", "blueprint", "whitepaper", "case_study", "onboarding", "handbook", "policies", "guidelines", "template_file", "sprint_plan", "release_notes", "incident_report", "postmortem", "runbook", "playbook", "architecture_doc", "diagram", "mockup", "wireframe", "storyboard", "timeline", "milestones", "risk_assessment", "audit_report", "compliance_doc", "receipt", "purchase_order", "work_order", "manifest", "catalog", "newsletter", "press_release", "product_roadmap", "competitive_analysis", "user_research_data", "interview_notes", "survey_results", "analytics_dashboard", "kpi_report", "okr_tracking", "performance_review", "feedback_summary", "goals", "objectives", "action_items", "decision_log", "meeting_agenda", "travel_plan", "expense_report", "timesheet_v2", "roi_analysis", "cost_breakdown", "vendor_list", "client_list", "partner_agreement", "license", "warranty_doc", "manual", "guide", "faq", "troubleshooting_guide"],
  ext: ["pdf", "xlsx", "docx", "txt", "json", "csv", "png", "jpg", "md", "yaml", "xml", "ppt", "pptx", "py", "js", "ts", "css", "html", "sql", "log", "cfg", "ini", "env", "zip", "tar", "svg", "webp"],
  time: ["last week", "yesterday", "this month", "March", "Q1", "2025", "2026", "last month", "today", "April", "Q2", "this quarter", "last quarter", "this year", "Q3", "Q4", "January", "February", "May", "June", "July", "August", "September", "October", "November", "December", "this morning", "last night", "Monday", "last Friday"],
  folder: ["Documents", "Downloads", "Desktop", "Projects", "Work", "Research", "Backup", "Archive", "Presentations", "Spreadsheets", "PDFs", "Images", "Scripts", "Data", "Reports", "Invoices", "Contracts", "Notes", "Templates", "Config", "Logs", "Resources"],
  contact: ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank", "Iris", "the team", "my manager", "the client", "HR", "IT support", "my colleague", "support", "billing", "engineering"],
  subject: ["the report", "meeting notes", "Q3 update", "project timeline", "budget review", "weekly summary", "action items", "follow up", "proposal", "urgent notice", "reminder", "status update", "monthly review", "feedback", "approval needed", "new draft", "revised version", "deadline reminder", "introduction", "partnership", "quote", "estimate", "invoice", "payment confirmation", "technical specs", "design review", "deployment notice", "maintenance window"],
  topic: ["AI funding news", "GPU benchmarks", "LLM reasoning", "deep learning trends", "causal inference", "Python libraries", "quantization techniques", "tech news", "open source models", "RLHF advances", "tool use agents", "fine-tuning guides", "Macbook specs", "cloud GPU pricing", "retrieval augmented generation", "vector databases", "few shot prompting", "model distillation", "synthetic data", "attention mechanisms", "transformer architectures", "reinforcement learning", "computer vision", "natural language processing", "time series forecasting", "recommender systems", "graph neural networks", "transfer learning", "multi modal models", "speech recognition", "image generation", "knowledge graphs", "semantic search", "embedding models", "model compression", "dataset curation", "active learning", "program synthesis", "prompt engineering", "agent frameworks"],
  time_range: ["this week", "this month", "this year", "2026", "past 3 months", "recent", "last 24 hours", "this quarter"],
  detail: ["latest news", "research papers", "tutorials", "benchmarks", "comparisons", "reviews", "case studies", "documentation", "best practices", "guides"],
  content_phrase: ["the meeting summary", "my shopping list", "the research findings", "todays notes", "the Q3 report", "the budget breakdown", "a draft email", "task list for tomorrow", "the weekly plan", "project timeline", "meeting agenda", "follow up items", "the sprint retrospective", "code review feedback", "deployment checklist", "incident response steps", "onboarding plan", "training schedule", "performance metrics", "the quarterly OKRs", "stakeholder update", "risk register", "change log", "release notes draft", "architecture decision record", "API documentation", "user feedback summary", "competitive analysis", "SWOT analysis", "cost benefit analysis"],
  note_title: ["todo", "shopping list", "ideas", "meeting notes", "reading list", "weekly goals", "projects", "groceries", "contacts", "bookmarks", "journal", "recipes", "workout plan", "coding snippets", "books to read", "movies to watch", "gift ideas", "home improvement", "budget tracker", "learning log", "startup ideas", "inspiring quotes", "daily standup", "sprint backlog", "feature requests", "bug tracker", "code review notes", "deployment log", "server inventory"],
  command: ["list files", "show disk space", "check memory", "monitor CPU", "list processes", "find large files", "count files", "sort by date", "show network stats", "check disk health", "list USB devices", "show system info", "list installed programs", "check battery status", "show wifi networks", "list environment variables", "check python version", "list npm packages", "show git status", "list docker containers", "check node version", "list background tasks", "show firewall status", "check for updates"],
  url_target: ["the first result", "the top link", "that article", "the Wikipedia page", "the GitHub repo", "the blog post", "the paper", "the documentation", "the official website", "the tutorial page", "the discussion thread", "the news article", "the research page", "the forum post", "the documentation site", "the API reference", "the source code", "the release notes", "the FAQ page", "the landing page", "the product page", "the about page", "the support page", "the getting started guide"],
  clip_content: ["the URL I just copied", "the text I selected", "the code snippet", "the email address", "the meeting link", "the error message", "the command I ran", "the commit hash", "the server address", "the file path I copied", "the download link", "the IP address", "the deployment URL", "the PR link", "the invite link", "the token", "the database URL"],
  reasoning_q: ["What percentage of employees who completed training achieved high performance?", "If the company forced everyone to train, what percentage would achieve high performance?", "Given that someone trained and performed poorly, what would their performance have been without training?", "What is the correlation between training and performance?", "Does wealth cause income directly or through education?", "If everyone got advanced education, would income increase?", "Is the relationship between education and income confounded by wealth?", "What percentage of people with high wealth have high income?", "What percentage of people would have high income if everyone had advanced education?", "What is the probability that a patient with a positive test actually has the disease?", "If we administer the drug to everyone, what is the expected recovery rate?", "What would have happened to the patient if they had received the treatment?", "Does the new policy cause higher sales?", "What is the causal effect of fertilizer on crop yield?", "If we raise the minimum wage, what happens to employment levels?", "Does the advertisement cause more signups or just correlate?", "What is the direct effect of exercise on heart health?", "If we shut down the server, how many users would be affected?", "What would the conversion rate be if we changed the button color?", "Does this marketing campaign actually drive sales?", "If we increase prices by 10%, what happens to revenue?", "What is the net effect of a new feature on user retention?", "Given the confounding factors, what is the true treatment effect?"],
  reasoning_ctx: ["Given the causal chain Training -> Skill -> Performance", "Given the fork Wealth -> Education and Wealth -> Income", "Given the collider Skill -> Hiring <- Network", "Given the chain Ice Cream -> Swimming -> Drowning", "Given the fork Weather -> Ice Cream and Weather -> Drowning", "Given the backdoor path Drug <- Doctor -> Recovery", "Given the front-door path Fertilizer -> Soil Quality -> Crop Yield", "Given the instrument Price -> Demand <- Quality", "Given a confounded A/B test where user segment affects both exposure and outcome", "Given a marketing attribution problem with multiple touchpoints", "Considering a clinical trial with noncompliance", "Given an observational study with selection bias"],
};

/* ───── sentence templates per tool ───── */
const SENTENCES = {
  file_search: [
    "Find the {filename} {ext} from {time}",
    "Search for {filename} {ext} in my {folder} folder",
    "Look up {ext} files named {filename} from {time}",
    "Can you find the {filename} {ext}?",
    "Where is the {filename} file from {time}?",
    "List all {ext} files in my {folder} folder",
    "Show me all files from {time} containing {filename}",
    "I need the {filename} {ext}, it should be from {time}",
    "Find me the most recent {ext} file in {folder}",
    "Look through my {folder} for files modified {time}",
    "Locate all {ext} documents in {folder} from {time}",
    "Search my {folder} for anything named {filename}",
    "Where did I save the {filename} file?",
    "Can you locate {filename} somewhere on my system?",
    "What {ext} files do I have from {time}?",
    "Find me the file named {filename}",
    "Pull up the {filename} document",
    "Show everything in {folder} that is a {ext} file",
  ],
  read_file: [
    "Read the {filename} {ext} file",
    "Show me the contents of {filename}.{ext}",
    "Open {filename}.{ext} and display it",
    "Display the {filename} file",
    "What does {filename} say?",
    "Show the {filename}.{ext} file contents",
    "Print the contents of {filename}.{ext}",
    "Can you open {filename} and read it to me?",
    "Show what is inside {filename}",
    "Pull up {filename}.{ext} and show me",
    "Let me see what is in the {filename} file",
    "Open the file called {filename}",
    "Read my {filename} file back to me",
    "What is written in {filename}.{ext}?",
    "Get me the content of {filename}.{ext}",
    "Output {filename}.{ext} to the console",
  ],
  write_file: [
    "Save this to a file called {filename}.{ext}: {content_phrase}",
    "Create {filename}.{ext} with the following: {content_phrase}",
    "Write {content_phrase} to {filename}.{ext}",
    "Create a new file named {filename} and put {content_phrase} in it",
    "Save the following to {filename}.{ext}: {content_phrase}",
    "Write these notes to {filename}",
    "Output the results to {filename}.{ext}",
    "Store {content_phrase} in a new file named {filename}.{ext}",
    "Save this text as {filename}.{ext}",
    "Write a file containing {content_phrase} and name it {filename}",
    "Generate {filename}.{ext} containing {content_phrase}",
    "Make a new file {filename} and add {content_phrase}",
    "Put {content_phrase} into {filename}.{ext} and save it",
    "Save {content_phrase} as {filename}.{ext}",
  ],
  email: [
    "Email {contact} about {subject}",
    "Send an email to {contact} with {subject}",
    "Mail {contact} about {subject}",
    "Draft an email to {contact} regarding {subject}",
    "Write an email to {contact} about {subject}",
    "Compose an email for {contact} about {subject}",
    "Message {contact} about {subject}",
    "Contact {contact} via email about {subject}",
    "Send an update to {contact} on {subject}",
    "Forward the info on {subject} to {contact}",
    "Send the {filename} {ext} to {contact}",
    "Email {contact} the {subject} document",
    "Send the {filename} to {contact} as an attachment",
    "Attach {filename}.{ext} and email it to {contact}",
  ],
  web_search: [
    "Search the web for {topic}",
    "Look up {topic} online",
    "Find {detail} about {topic}",
    "Google {topic}",
    "Search for {topic} from {time_range}",
    "What are the latest {detail} on {topic}?",
    "Find me {detail} about {topic}",
    "I need information on {topic}",
    "Check online for {topic} {detail}",
    "Look up information about {topic} on the internet",
    "Can you find {topic} online?",
    "Do a web search for {topic}",
    "Find the latest on {topic}",
    "Research {topic} on the web",
    "Pull up search results for {topic}",
    "What does the internet say about {topic}?",
    "Look into {topic} online",
  ],
  browse: [
    "Open {url_target} and summarize it",
    "Visit {url_target} and tell me what it says",
    "Open the link and extract the main content",
    "Go to {url_target} and read the page",
    "Fetch the content from {url_target}",
    "Open the page at that link",
    "Navigate to {url_target} and get the text",
    "Load {url_target} in the browser",
    "Access {url_target} and pull the content",
    "Retrieve the page at {url_target}",
    "Open up {url_target} and parse the content",
    "Load up the page called {url_target}",
  ],
  clipboard: [
    "Read what is on my clipboard",
    "Check my clipboard contents",
    "What did I copy?",
    "Copy this to my clipboard: {clip_content}",
    "Save {clip_content} to clipboard",
    "Put {clip_content} on my clipboard",
    "Read the clipboard and tell me what is there",
    "What is stored on my clipboard right now?",
    "Check what I have copied",
    "Get the text from my clipboard",
    "Copy {clip_content} to system clipboard",
    "Set the clipboard to {clip_content}",
    "Store this on my clipboard: {clip_content}",
    "Read my clipboard contents please",
  ],
  shell_command: [
    "Run a command to {command}",
    "Execute {command} in the terminal",
    "Run {command} in the shell",
    "Open terminal and run {command}",
    "Fire off a shell command: {command}",
    "Run the following command: {command}",
    "Execute a terminal command: {command}",
    "Run from the command line: {command}",
    "Launch a shell and execute {command}",
    "Use the terminal to {command}",
    "Kick off a shell process: {command}",
    "Execute {command} in command prompt",
    "Run a system command: {command}",
    "Run {command} on the system",
  ],
  notes: [
    "Add {content_phrase} to my {note_title} notes",
    "Write {content_phrase} in my {note_title} note",
    "Show me my {note_title} notes",
    "Read my {note_title} list",
    "Append {content_phrase} to my {note_title}",
    "Save this as a note: {content_phrase}. Title: {note_title}",
    "What is on my {note_title}?",
    "Remind me to {content_phrase}",
    "Check my {note_title} notes for me",
    "Update my {note_title} with {content_phrase}",
    "Create a note titled {note_title} with {content_phrase}",
    "Log {content_phrase} in my {note_title}",
    "Read from my {note_title} collection",
    "Open my {note_title} and show the contents",
    "Add an entry to {note_title}: {content_phrase}",
    "Display what I have saved under {note_title}",
    "Jot down {content_phrase} in {note_title}",
    "Record {content_phrase} in my notes under {note_title}",
    "Append to my {note_title} file: {content_phrase}",
    "Fetch my {note_title} list please",
  ],
  reasoning: [
    "{reasoning_q}",
    "Think about this: {reasoning_q}",
    "Reason through this: {reasoning_ctx}. {reasoning_q}",
    "I need to figure out: {reasoning_q}",
    "Can you reason about this: {reasoning_ctx}. {reasoning_q}",
    "Work through this problem: {reasoning_q}",
    "Analyze this situation: {reasoning_ctx}. {reasoning_q}",
    "Apply causal reasoning here: {reasoning_q}",
    "Given {reasoning_ctx}, answer: {reasoning_q}",
    "Use the causal structure to determine: {reasoning_q}",
    "What does causal inference say about {reasoning_q}",
    "Using the provided structure, solve: {reasoning_q}",
    "Think step by step about {reasoning_q}",
    "Reason causally about this: {reasoning_ctx}. {reasoning_q}",
  ],
};

/* ───── reasoning text generator ───── */
const TOOL_REASON_DESC = {
  file_search: (v) => `find the "${v.filename}" file`,
  read_file: (v) => `read "${v.filename}"`,
  write_file: (v) => `write to "${v.filename}.${v.ext}"`,
  email: (v) => `email ${v.to || v.contact}`,
  web_search: (v) => `search for ${v.topic || v.query}`,
  browse: (v) => `open ${v.url || v.url_target}`,
  clipboard: (v) => `${v.action || "read"} the clipboard`,
  shell_command: (v) => `run: ${v.command}`,
  notes: (v) => `${v.action || "read"} my "${v.title}" notes`,
  reasoning: (v) => `reason about: ${v.question}`,
};

const SYNTH_REASONING = {
  minimal: {
    single: [
      (s, v) => { const d = defaultDesc(s[0].tool, { ...v, ...s[0].params }); return `The user wants me to ${d}. I'll do this directly.`; },
      (s, v) => { const d = defaultDesc(s[0].tool, { ...v, ...s[0].params }); return `One step: ${d}.`; },
      (s, v) => { const d = defaultDesc(s[0].tool, { ...v, ...s[0].params }); return `Straightforward — ${d}.`; },
      (s, v) => { const d = defaultDesc(s[0].tool, { ...v, ...s[0].params }); return `${d}.`; },
      (s, v) => { const d = defaultDesc(s[0].tool, { ...v, ...s[0].params }); return `Request maps to ${d}. Executing.`; },
      (s, v) => { const d = defaultDesc(s[0].tool, { ...v, ...s[0].params }); return `Dry run: ${d}.`; },
    ],
    chain: [
      (s, v) => `Plan: ${s.map(st => defaultDesc(st.tool, { ...v, ...st.params })).join(" → ")}.`,
      (s, v) => `${s.length} steps: ${s.map((st, i) => `${i+1}. ${defaultDesc(st.tool, { ...v, ...st.params })}`).join(" ")}`,
      (s, v) => `Chain: ${s.map(st => st.tool).join(" → ")}.`,
      (s, v) => `Multi-step: ${s.map(st => defaultDesc(st.tool, { ...v, ...st.params })).join(", then ")}.`,
    ],
  },
  verbose: {
    single: [
      (s, v) => { const d = defaultDesc(s[0].tool, { ...v, ...s[0].params }); return `The user wants me to ${d}. I'll use the correct tool with appropriate parameters and execute.`; },
      (s, v) => { const d = defaultDesc(s[0].tool, { ...v, ...s[0].params }); return `Let me handle this request. The task is to ${d}. I'll set up the parameters carefully and call the tool.`; },
      (s, v) => { const d = defaultDesc(s[0].tool, { ...v, ...s[0].params }); return `Processing the user request. I need to ${d}. This is a single-step operation. I'll use the right parameters and return the result.`; },
      (s, v) => { const d = defaultDesc(s[0].tool, { ...v, ...s[0].params }); return `The user's intent is clear: ${d}. Let me execute this operation with proper parameter configuration.`; },
      (s, v) => { const d = defaultDesc(s[0].tool, { ...v, ...s[0].params }); return `I understand the request. The correct action is to ${d}. I'll proceed with the appropriate tool call now.`; },
    ],
    chain: [
      (s, v) => `I need to handle this multi-step request. Plan: ${s.map(st => defaultDesc(st.tool, { ...v, ...st.params })).join(", then ")}. Each step depends on the previous one.`,
      (s, v) => `Let me break this down. Step 1: ${defaultDesc(s[0].tool, { ...v, ...s[0].params })}. Step 2: ${defaultDesc(s[1]?.tool, { ...v, ...s[1]?.params }) || ""}. I'll chain these with proper dependency tracking.`,
      (s, v) => `The user wants a multi-step workflow. First, ${defaultDesc(s[0].tool, { ...v, ...s[0].params })}. Then, using that result, ${defaultDesc(s[1]?.tool, { ...v, ...s[1]?.params }) || ""}. Chaining dependencies.`,
    ],
  },
};

function defaultDesc(tool, v) {
  const fn = TOOL_REASON_DESC[tool];
  return fn ? fn(v) : `execute ${tool}`;
}

function generateSynthReasoning(steps, vars, isMultiStep, variant) {
  const bank = variant === "verbose" ? SYNTH_REASONING.verbose : SYNTH_REASONING.minimal;
  const pool = isMultiStep ? bank.chain : bank.single;
  const tpl = pick(pool);
  try { return tpl(steps, vars); } catch { return "Proceed."; }
}

/* ───── param transformers ───── */
const PARAM_TRANSFORMS = {
  file_search: (vars) => {
    const p = {};
    if (vars.filename) p.filename = vars.filename;
    if (vars.ext) p.ext = vars.ext;
    if (vars.time) p.time = vars.time;
    if (vars.folder) p.folder = vars.folder;
    return p;
  },
  read_file: (vars) => {
    if (vars.path) return { path: vars.path };
    const filename = vars.filename || pick(V.filename);
    const ext = vars.ext || "";
    return { path: ext ? `${filename}.${ext}` : filename };
  },
  write_file: (vars) => ({
    path: vars.filename && vars.ext ? `${vars.filename}.${vars.ext}` : (vars.path || "output.txt"),
    content: vars.content_phrase || pick(V.content_phrase),
  }),
  email: (vars) => {
    const p = {};
    p.to = vars.contact || pick(V.contact);
    p.subject = vars.subject || pick(V.subject);
    if (Math.random() < 0.3 && vars.filename && vars.ext) {
      p.attachments = [`${vars.filename}.${vars.ext}`];
    }
    return p;
  },
  web_search: (vars) => {
    let q = vars.topic || pick(V.topic);
    if (vars.detail) q = `${q} ${vars.detail}`;
    if (vars.time_range) q = `${q} ${vars.time_range}`;
    return { query: q };
  },
  browse: (vars) => ({ url: vars.url_target || pick(V.url_target) }),
  clipboard: (vars) => {
    if (Math.random() < 0.5) return { action: "read" };
    return { action: "write", content: vars.clip_content || pick(V.clip_content) };
  },
  shell_command: (vars) => ({ command: vars.command || pick(V.command) }),
  notes: (vars) => {
    const action = pick(["read", "write", "append"]);
    const p = { action, title: vars.note_title || pick(V.note_title) };
    if (action !== "read") p.content = vars.content_phrase || pick(V.content_phrase);
    return p;
  },
  reasoning: (vars) => {
    const p = { question: vars.reasoning_q || pick(V.reasoning_q) };
    if (vars.reasoning_ctx) p.context = vars.reasoning_ctx;
    return p;
  },
};

/* ───── phrasing variations ───── */
const PHRASINGS = [
  (s) => s,
  (s) => `Can you ${s[0].toLowerCase() + s.slice(1)}`,
  (s) => `Please ${s[0].toLowerCase() + s.slice(1)}`,
  (s) => `Hey BTL, ${s[0].toLowerCase() + s.slice(1)}`,
  (s) => `I need you to ${s[0].toLowerCase() + s.slice(1)}`,
  (s) => `Could you ${s[0].toLowerCase() + s.slice(1)}`,
  (s) => `${s}, please`,
  (s) => `Help me ${s[0].toLowerCase() + s.slice(1)}`,
  (s) => `${s} asap`,
  (s) => `Can you please ${s[0].toLowerCase() + s.slice(1)}?`,
  (s) => `I would like you to ${s[0].toLowerCase() + s.slice(1)}`,
  (s) => `${s} thanks`,
  (s) => `Would you ${s[0].toLowerCase() + s.slice(1)}?`,
  (s) => `Now ${s[0].toLowerCase() + s.slice(1)}`,
];

function applyPhrasing(s) {
  const fn = pick(PHRASINGS);
  let r = fn(s).replace(/\s+/g, " ").trim();
  return r.charAt(0).toUpperCase() + r.slice(1);
}

/* ───── synthetic triplet builder (minimal / verbose / negative) ───── */
function buildSynthTriplet(tool) {
  const tpls = SENTENCES[tool];
  const tpl = pick(tpls);
  const vars = {};
  let result = tpl;
  const matches = [...tpl.matchAll(/\{(\w+)\}/g)].map((m) => m[1]);
  for (const key of matches) {
    if (V[key]) {
      const val = pick(V[key]);
      vars[key] = val;
      result = result.replace(new RegExp(`\\{${key}\\}`, "g"), val);
    }
  }
  const input = applyPhrasing(result);
  const transform = PARAM_TRANSFORMS[tool];
  let params = {};
  if (transform) params = transform(vars);
  else for (const [k, v] of Object.entries(vars)) params[k] = v;

  const siblingTools = TOOL_REASON_DESC[tool] ? Object.keys(TOOL_REASON_DESC).filter(t => t !== tool && t !== "reasoning") : [tool];
  const wrongTool = pick(siblingTools);
  const wrongParams = PARAM_TRANSFORMS[wrongTool] ? PARAM_TRANSFORMS[wrongTool](vars) : { ...params };

  const variants = [];
  for (const variant of ["minimal", "verbose", "negative"]) {
    const steps = variant === "negative"
      ? [{ tool: wrongTool, params: wrongParams, id: "step_1" }]
      : [{ tool, params: { ...params }, id: "step_1" }];
    const reasoning = generateSynthReasoning(steps, vars, false, variant);
    const output = `<reasoning>${reasoning}</reasoning>\n${JSON.stringify(steps)}`;
    variants.push({ input, output, tool, template_id: `synth-${tool}-${variant}`, variant });
  }
  return variants;
}

/* ───── chain definitions ───── */
const CHAINS = [
  { sentence: "Find the {filename} {ext} and read it", steps: [
    { tool: "file_search", params: (v) => ({ filename: v.filename, ext: v.ext }) },
    { tool: "read_file", params: (v) => ({ path: `${v.filename}.${v.ext}` }), depends: true },
  ]},
  { sentence: "Search for {topic} and open the first link", steps: [
    { tool: "web_search", params: (v) => ({ query: v.topic }) },
    { tool: "browse", params: () => ({ url: "the top result" }), depends: true },
  ]},
  { sentence: "Find the {filename} {ext} and email it to {contact}", steps: [
    { tool: "file_search", params: (v) => ({ filename: v.filename, ext: v.ext }) },
    { tool: "email", params: (v) => ({ to: v.contact, subject: v.filename, attachments: [`${v.filename}.${v.ext}`] }), depends: true },
  ]},
  { sentence: "Search for {topic} and save the findings to my {note_title}", steps: [
    { tool: "web_search", params: (v) => ({ query: v.topic }) },
    { tool: "notes", params: (v) => ({ action: "write", title: v.note_title, content: "search results" }), depends: true },
  ]},
  { sentence: "Read my {note_title} and email them to {contact}", steps: [
    { tool: "notes", params: (v) => ({ action: "read", title: v.note_title }) },
    { tool: "email", params: (v) => ({ to: v.contact, subject: v.note_title }), depends: true },
  ]},
  { sentence: "Search for {topic}, open the top link, and save it to my {note_title}", steps: [
    { tool: "web_search", params: (v) => ({ query: v.topic }) },
    { tool: "browse", params: () => ({ url: "the top result" }), depends: true },
    { tool: "notes", params: (v) => ({ action: "write", title: v.note_title, content: "browsed content" }), depends: true },
  ]},
  { sentence: "Search for {topic}, open the article, and email the summary to {contact}", steps: [
    { tool: "web_search", params: (v) => ({ query: v.topic }) },
    { tool: "browse", params: () => ({ url: "the first result" }), depends: true },
    { tool: "email", params: (v) => ({ to: v.contact, subject: `${v.topic} summary` }), depends: true },
  ]},
  { sentence: "Read the clipboard and search for that term online", steps: [
    { tool: "clipboard", params: () => ({ action: "read" }) },
    { tool: "web_search", params: () => ({ query: "clipboard content" }), depends: true },
  ]},
  { sentence: "Find the {filename} {ext} and tell me what it says", steps: [
    { tool: "file_search", params: (v) => ({ filename: v.filename, ext: v.ext }) },
    { tool: "read_file", params: (v) => ({ path: `${v.filename}.${v.ext}` }), depends: true },
    { tool: "reasoning", params: (v) => ({ question: `Summarize the ${v.filename} document` }), depends: true },
  ]},
  { sentence: "Search for {topic} and copy the summary to clipboard", steps: [
    { tool: "web_search", params: (v) => ({ query: v.topic }) },
    { tool: "browse", params: () => ({ url: "the first result" }), depends: true },
    { tool: "clipboard", params: () => ({ action: "write", content: "browsed content" }), depends: true },
  ]},
  { sentence: "Read my {note_title} notes and find files related to them", steps: [
    { tool: "notes", params: (v) => ({ action: "read", title: v.note_title }) },
    { tool: "file_search", params: (v) => ({ filename: `related to ${v.note_title}` }), depends: true },
  ]},
  { sentence: "Count all {ext} files and run a command to list them sorted by size", steps: [
    { tool: "file_search", params: (v) => ({ ext: v.ext }) },
    { tool: "shell_command", params: () => ({ command: "list files sorted by size" }), depends: true },
  ]},
  { sentence: "Search for {topic} and save the results to a file named {filename}.{ext}", steps: [
    { tool: "web_search", params: (v) => ({ query: v.topic }) },
    { tool: "write_file", params: (v) => ({ path: `${v.filename}.${v.ext}`, content: "search results" }), depends: true },
  ]},
  { sentence: "Run a command to find {ext} files, then email the list to {contact}", steps: [
    { tool: "shell_command", params: (v) => ({ command: `find ${v.ext} files` }) },
    { tool: "email", params: (v) => ({ to: v.contact, subject: "file list" }), depends: true },
  ]},
  { sentence: "Copy the clipboard to a file called {filename}.{ext} then email it to {contact}", steps: [
    { tool: "clipboard", params: () => ({ action: "read" }) },
    { tool: "write_file", params: (v) => ({ path: `${v.filename}.${v.ext}`, content: "clipboard content" }), depends: true },
    { tool: "email", params: (v) => ({ to: v.contact, subject: "clipboard", attachments: [`${v.filename}.${v.ext}`] }), depends: true },
  ]},
  { sentence: "Answer this: {reasoning_q}. Then save the answer to my {note_title}", steps: [
    { tool: "reasoning", params: (v) => ({ question: v.reasoning_q }) },
    { tool: "notes", params: (v) => ({ action: "write", title: v.note_title, content: "reasoning answer" }), depends: true },
  ]},
  { sentence: "Read the {filename} {ext} and email its content to {contact}", steps: [
    { tool: "read_file", params: (v) => ({ path: `${v.filename}.${v.ext}` }) },
    { tool: "email", params: (v) => ({ to: v.contact, subject: v.filename }), depends: true },
  ]},
  { sentence: "Search for {topic}, summarize the top link, and copy to clipboard", steps: [
    { tool: "web_search", params: (v) => ({ query: v.topic }) },
    { tool: "browse", params: () => ({ url: "the first result" }), depends: true },
    { tool: "clipboard", params: () => ({ action: "write", content: "summary" }), depends: true },
  ]},
  { sentence: "Find the {filename} {ext} from {time} and tell me what it says", steps: [
    { tool: "file_search", params: (v) => ({ filename: v.filename, ext: v.ext, time: v.time }) },
    { tool: "read_file", params: (v) => ({ path: `${v.filename}.${v.ext}` }), depends: true },
    { tool: "reasoning", params: (v) => ({ question: `Summarize the ${v.filename} document` }), depends: true },
  ]},
  { sentence: "Search for {topic} and add the best result to my {note_title} and email it to {contact}", steps: [
    { tool: "web_search", params: (v) => ({ query: v.topic }) },
    { tool: "notes", params: (v) => ({ action: "write", title: v.note_title, content: "search results" }), depends: true },
    { tool: "email", params: (v) => ({ to: v.contact, subject: `Results: ${v.topic}` }), depends: true },
  ]},
];

/* ───── synthetic chain triplet ───── */
function makeChainTriplet(chainDef) {
  const vars = {};
  const sentence = chainDef.sentence;
  const varMatches = [...sentence.matchAll(/\{(\w+)\}/g)].map((m) => m[1]);
  for (const v of varMatches) { if (V[v]) vars[v] = pick(V[v]); }
  let input = sentence;
  for (const [k, v] of Object.entries(vars)) input = input.replace(new RegExp(`\\{${k}\\}`, "g"), v);
  input = input.replace(/\{(\w+)\}/g, (_, key) => V[key] ? pick(V[key]) : `{${key}}`);
  input = applyPhrasing(input);

  const correctSteps = [];
  let prevStepId = null;
  for (let i = 0; i < chainDef.steps.length; i++) {
    const stepDef = chainDef.steps[i];
    const stepId = `step_${i + 1}`;
    const params = JSON.parse(JSON.stringify(stepDef.params(vars)));
    const step = { tool: stepDef.tool, params, id: stepId };
    if (stepDef.depends && prevStepId) step.depends_on = [prevStepId];
    correctSteps.push(step);
    prevStepId = stepId;
  }

  // Build negative: swap the last tool with a sibling
  const wrongSteps = correctSteps.map(s => ({ ...s, params: { ...s.params } }));
  if (wrongSteps.length > 0) {
    const last = wrongSteps[wrongSteps.length - 1];
    const siblings = Object.keys(TOOL_REASON_DESC).filter(t => t !== last.tool && t !== "reasoning");
    if (siblings.length > 0) {
      last.tool = pick(siblings);
      if (PARAM_TRANSFORMS[last.tool]) {
        last.params = PARAM_TRANSFORMS[last.tool](vars);
      }
    }
  }

  const toolKey = correctSteps.map(s => s.tool).join("_");
  const variants = [];

  for (const variant of ["minimal", "verbose", "negative"]) {
    const steps = variant === "negative" ? wrongSteps : correctSteps;
    const reasoning = generateSynthReasoning(steps, vars, true, variant);
    const output = `<reasoning>${reasoning}</reasoning>\n${JSON.stringify(steps)}`;
    variants.push({ input, output, template_id: `synth-chain-${toolKey}-${variant}`, variant });
  }
  return variants;
}

/* ───── scrub final row ───── */
function scrubFinalRow(row) {
  const copy = JSON.parse(JSON.stringify(row));
  const assistant = copy.messages?.find((m) => m.role === "assistant");
  if (!assistant || typeof assistant.content !== "string") return null;

  // Depth 0: code output, not tool chain — pass through
  if (copy.provenance?.source_depth === 0) return copy;

  const reasoningMatch = assistant.content.match(/^<reasoning>([\s\S]*?)<\/reasoning>\s*/);
  const reasoningTag = reasoningMatch ? reasoningMatch[0] : "";
  const jsonStr = assistant.content.replace(/<reasoning>[\s\S]*?<\/reasoning>\s*/g, "").trim();
  let steps;
  try { steps = JSON.parse(jsonStr); } catch { return null; }
  if (!Array.isArray(steps)) return null;
  for (const step of steps) {
    if (!step || typeof step !== "object") return null;
    if (step.tool === "file_search" && step.params && typeof step.params === "object") {
      if (step.params.name && !step.params.filename) { step.params.filename = step.params.name; delete step.params.name; }
    }
    if (step.tool === "email" && step.params && typeof step.params === "object") {
      if (step.params.content && !step.params.body) { step.params.body = step.params.content; delete step.params.content; }
      if (!step.params.body) {
        if (Array.isArray(step.params.attachments) && step.params.attachments.length > 0) step.params.body = "Please see the attached file.";
        else if (Array.isArray(step.depends_on) && step.depends_on.length > 0) step.params.body = `$${step.depends_on[step.depends_on.length - 1]}.result`;
        else if (step.params.subject) step.params.body = `Follow up on ${step.params.subject}.`;
        else step.params.body = "Please review this.";
      }
    }
    if (step.tool === "write_file" && typeof step.params?.path === "string" && /\.(ppt|pptx)$/i.test(step.params.path)) return null;
  }
  assistant.content = reasoningTag + JSON.stringify(steps);
  return copy;
}

/* ───── MAIN ───── */
async function main() {
  const allRows = [];

  /* ── 1. Spec-driven rows ── */
  const startSpec = Date.now();
  const codingRows = readCodingSpec();
  const d1Rows = readDepth1Spec();
  const d2Rows = readDepth2Spec();
  const d3Rows = readDepth3Spec();
  const d4Rows = readDepth4Spec();
  const specTotal = codingRows.length + d1Rows.length + d2Rows.length + d3Rows.length + d4Rows.length;
  allRows.push(...codingRows, ...d1Rows, ...d2Rows, ...d3Rows, ...d4Rows);
  console.log(`Spec rows: ${specTotal} (coding:${codingRows.length} d1:${d1Rows.length} d2:${d2Rows.length} d3:${d3Rows.length} d4:${d4Rows.length}) [${Date.now()-startSpec}ms]`);

  /* ── 2. Synthetic rows (triplets: minimal/verbose/negative) ── */
  const startSynth = Date.now();
  const seen = new Set();
  const synthRows = [];
  const tools = Object.keys(SENTENCES);
  for (const tool of tools) {
    for (let i = 0; i < 15000; i++) {
      const triplet = buildSynthTriplet(tool);
      for (const { input, output, template_id, variant } of triplet) {
        if (input.length < 5) continue;
        const key = input + "|" + variant;
        if (seen.has(key)) continue;
        seen.add(key);
        synthRows.push({
          messages: [
            { role: "system", content: SYSTEM_PROMPT },
            { role: "user", content: input },
            { role: "assistant", content: output },
          ],
          provenance: { template_id, source_depth: 1, source_family: tool, variant, api_or_template: "synthetic", negative_type: variant === "negative" ? `wrong_tool:${tool}` : null },
        });
      }
    }
  }
  console.log(`Single-tool synthetic: ${synthRows.length} [${Date.now()-startSynth}ms]`);

  let chainCount = 0;
  const chainTripletLimit = 25000;
  for (let i = 0; i < 400000 && chainCount < chainTripletLimit; i++) {
    const chainDef = pick(CHAINS);
    const triplet = makeChainTriplet(chainDef);
    for (const { input, output, template_id, variant } of triplet) {
      if (input.length < 5) continue;
      const key = input + "|" + variant;
      if (seen.has(key)) continue;
      seen.add(key);
      synthRows.push({
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: input },
          { role: "assistant", content: output },
        ],
        provenance: { template_id, source_depth: 2, source_family: "chain", variant, api_or_template: "synthetic", negative_type: variant === "negative" ? "wrong-step" : null },
      });
    }
    chainCount++;
  }
  console.log(`Chain synthetic: ${chainCount} triplets [total synth: ${synthRows.length}] [${Date.now()-startSynth}ms]`);

  for (const sr of synthRows) allRows.push(sr);

  /* ── 3. API data ── */
  const apiDir = resolve(DATA, "teacher", "runs", "2026-06-10T04-26-26-868Z");
  const apiPath = resolve(apiDir, "completed.jsonl");
  let apiRows = [];
  try {
    apiRows = readAPIData(apiPath);
    console.log(`API rows: ${apiRows.length}`);
    for (const ar of apiRows) allRows.push(ar);
  } catch (e) {
    console.log(`API data not found at ${apiPath}: ${e.message}`);
  }

  console.log(`Total rows before dedup: ${allRows.length}`);

  /* ── 4. Dedup by (user_message + variant) ── */
  const dedupSeen = new Set();
  const deduped = allRows.filter(r => {
    const inp = r.messages?.[1]?.content || "";
    const variant = r.provenance?.variant || "";
    const key = inp + "|" + variant;
    if (dedupSeen.has(key)) return false;
    dedupSeen.add(key);
    return true;
  });
  console.log(`After dedup: ${deduped.length}`);

  /* ── 5. Template-grouped eval split ── */
  const groups = {};
  for (const row of deduped) {
    const tid = row.provenance?.template_id || "unknown";
    if (!groups[tid]) groups[tid] = [];
    groups[tid].push(row);
  }
  const groupKeys = Object.keys(groups);
  // Shuffle group keys
  for (let i = groupKeys.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [groupKeys[i], groupKeys[j]] = [groupKeys[j], groupKeys[i]];
  }
  // 90/10 split by template group
  const splitIdx = Math.floor(groupKeys.length * 0.9);
  const trainGroups = groupKeys.slice(0, splitIdx);
  const evalGroups = groupKeys.slice(splitIdx);
  const train = [];
  const eval_ = [];
  for (const k of trainGroups) train.push(...groups[k]);
  for (const k of evalGroups) eval_.push(...groups[k]);
  console.log(`Template-grouped split: train=${train.length} eval=${eval_.length} (${trainGroups.length} train groups, ${evalGroups.length} eval groups)`);

  /* ── 6. Scrub ── */
  const cleaned = train.map(scrubFinalRow).filter(Boolean);
  const cleanedEval = eval_.map(scrubFinalRow).filter(Boolean);
  console.log(`After scrub: train=${cleaned.length} eval=${cleanedEval.length}`);

  /* ── 7. Write ── */
  function writeJsonl(path, data) {
    return new Promise((resolve, reject) => {
      const stream = createWriteStream(path, "utf-8");
      for (const row of data) stream.write(JSON.stringify(row) + "\n");
      stream.end();
      stream.on("finish", resolve);
      stream.on("error", reject);
    });
  }

  await writeJsonl(resolve(RAW, "train.jsonl"), deduped);
  await writeJsonl(resolve(RAW, "eval.jsonl"), eval_);
  await writeJsonl(resolve(FINAL, "train.jsonl"), cleaned);
  await writeJsonl(resolve(FINAL, "eval.jsonl"), cleanedEval);

  /* ── 8. Summary ── */
  const depthDist = {};
  for (const r of cleaned) {
    const d = r.provenance?.source_depth ?? "?";
    depthDist[d] = (depthDist[d] || 0) + 1;
  }
  const variantDist = {};
  for (const r of cleaned) {
    const v = r.provenance?.variant ?? "?";
    variantDist[v] = (variantDist[v] || 0) + 1;
  }
  const sourceDist = {};
  for (const r of cleaned) {
    const s = r.provenance?.api_or_template ?? "?";
    sourceDist[s] = (sourceDist[s] || 0) + 1;
  }

  console.log(`\n=== Summary ===`);
  console.log(`Train: ${cleaned.length}  Eval: ${cleanedEval.length}`);
  console.log(`By depth:`, JSON.stringify(depthDist));
  console.log(`By variant:`, JSON.stringify(variantDist));
  console.log(`By source:`, JSON.stringify(sourceDist));
  console.log(`\nWritten:`);
  console.log(`  ${resolve(RAW, "train.jsonl")}`);
  console.log(`  ${resolve(RAW, "eval.jsonl")}`);
  console.log(`  ${resolve(FINAL, "train.jsonl")}`);
  console.log(`  ${resolve(FINAL, "eval.jsonl")}`);
}

main();
