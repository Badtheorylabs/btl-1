const RESERVED_WORDS = new Set([
  "and", "as", "assert", "async", "await", "break", "class", "const", "continue",
  "def", "del", "elif", "else", "except", "False", "finally", "for", "from", "global",
  "if", "import", "in", "is", "lambda", "let", "null", "None", "not", "or", "pass",
  "public", "private", "protected", "return", "static", "super", "switch", "this",
  "throw", "true", "True", "try", "type", "var", "void", "while", "with", "yield",
  "function", "console", "new", "export", "default", "extends", "package", "interface",
  "implements", "final", "abstract", "int", "long", "double", "float", "char", "bool",
  "boolean", "String", "Map", "List", "Set", "Array", "Object", "undefined",
  "abs", "all", "any", "append", "copy", "count", "csv", "defaultdict", "dict", "enumerate",
  "filter", "float", "format", "hash", "int", "isinstance", "iter", "json", "len", "list",
  "map", "max", "min", "open", "print", "range", "reversed", "set", "sorted", "str", "sum",
  "time", "zip", "Counter", "Flask", "jsonify", "request", "wraps", "sqlite3", "math",
  "random", "datetime", "pathlib", "os", "sys", "re", "typing",
]);

export function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function slugify(value) {
  return String(value)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .replace(/_{2,}/g, "_");
}

export function replaceWords(text, replacements) {
  let output = String(text);
  for (const [from, to] of Object.entries(replacements)) {
    if (from === to) continue;
    const pattern = new RegExp(`\\b${escapeRegExp(from)}\\b`, "g");
    output = output.replace(pattern, to);
  }
  return output;
}

export function extractIdentifiers(text) {
  const matches = String(text).match(/\b[A-Za-z_][A-Za-z0-9_]*\b/g) || [];
  return [...new Set(matches.filter((token) => !RESERVED_WORDS.has(token)))];
}

export function extractFunctionName(code) {
  const text = String(code);
  const patterns = [
    /\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/m,
    /\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/m,
    /\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/m,
    /\bfun\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/m,
    /\bsub\s+([A-Za-z_][A-Za-z0-9_]*)\b/m,
    /\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b/m,
    /\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^()]*\)\s*\{/m,
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match && match[1] && !RESERVED_WORDS.has(match[1])) {
      return match[1];
    }
  }

  return null;
}

export function extractParams(code) {
  const text = String(code);
  const match = text.match(/\(([^)]*)\)/m);
  if (!match) return [];

  return match[1]
    .split(",")
    .map((part) => part.trim())
    .map((part) => part.replace(/=.*$/, "").replace(/[:?].*$/, "").replace(/^\.\.\./, "").trim())
    .filter((part) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(part) && !RESERVED_WORDS.has(part));
}
