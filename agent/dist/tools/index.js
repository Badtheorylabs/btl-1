import { promises as fs } from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
function toText(value) {
    if (typeof value === "string")
        return value;
    if (value === null || value === undefined)
        return "";
    if (typeof value === "number" || typeof value === "boolean")
        return String(value);
    return JSON.stringify(value);
}
function isObject(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
function resolveParamValue(value, results) {
    if (typeof value !== "string")
        return value;
    const exact = value.match(/^\$step_(\d+)\.result(?:\.(.+))?$/);
    if (exact) {
        const step = results.get(`step_${exact[1]}`);
        const base = step?.result;
        if (!exact[2])
            return base;
        return exact[2].split(".").reduce((acc, key) => {
            if (acc && typeof acc === "object" && key in acc) {
                return acc[key];
            }
            return undefined;
        }, base);
    }
    return value.replace(/\$step_(\d+)\.result(?:\.[A-Za-z0-9_]+)*/g, (match) => {
        const ref = match.match(/^\$step_(\d+)\.result(?:\.(.+))?$/);
        if (!ref)
            return match;
        const step = results.get(`step_${ref[1]}`);
        if (!step)
            return match;
        const resolved = resolveParamValue(match, results);
        return toText(resolved);
    });
}
function resolveParams(params, results) {
    if (!params)
        return {};
    return Object.fromEntries(Object.entries(params).map(([key, value]) => [
        key,
        Array.isArray(value)
            ? value.map((item) => resolveParamValue(item, results))
            : isObject(value)
                ? resolveParams(value, results)
                : resolveParamValue(value, results),
    ]));
}
async function ensureDir(filePath) {
    await fs.mkdir(path.dirname(filePath), { recursive: true });
}
async function readNotes(filePath) {
    try {
        return JSON.parse(await fs.readFile(filePath, "utf8"));
    }
    catch {
        return {};
    }
}
async function writeNotes(filePath, notes) {
    await ensureDir(filePath);
    await fs.writeFile(filePath, `${JSON.stringify(notes, null, 2)}\n`, "utf8");
}
function normalizePath(cwd, projectRoot, input) {
    if (path.isAbsolute(input))
        return input;
    const local = path.resolve(cwd, input);
    if (path.parse(input).root)
        return input;
    if (input.startsWith(".") || input.includes(path.sep))
        return local;
    return path.resolve(projectRoot, input);
}
async function walkFiles(root) {
    const found = [];
    async function visit(dir) {
        try {
            const entries = await fs.readdir(dir, { withFileTypes: true });
            for (const entry of entries) {
                const full = path.join(dir, entry.name);
                if (entry.isDirectory()) {
                    await visit(full);
                }
                else {
                    found.push(full);
                }
            }
        }
        catch {
            return;
        }
    }
    await visit(root);
    return found;
}
function quarterMatches(date, query) {
    const month = date.getMonth();
    const year = date.getFullYear();
    const q = Math.floor(month / 3) + 1;
    const target = query.toLowerCase();
    if (target.includes("q1"))
        return q === 1;
    if (target.includes("q2"))
        return q === 2;
    if (target.includes("q3"))
        return q === 3;
    if (target.includes("q4"))
        return q === 4;
    if (/\b2025\b/.test(target))
        return year === 2025;
    if (/\b2026\b/.test(target))
        return year === 2026;
    return true;
}
function timeMatches(stat, query) {
    if (typeof query !== "string" || !query.trim())
        return true;
    const target = query.toLowerCase();
    const mtime = stat.mtime;
    const now = new Date();
    if (target.includes("today"))
        return mtime.toDateString() === now.toDateString();
    if (target.includes("yesterday")) {
        const yesterday = new Date(now);
        yesterday.setDate(yesterday.getDate() - 1);
        return mtime.toDateString() === yesterday.toDateString();
    }
    if (target.includes("this month")) {
        return mtime.getMonth() === now.getMonth() && mtime.getFullYear() === now.getFullYear();
    }
    if (target.includes("this year"))
        return mtime.getFullYear() === now.getFullYear();
    return quarterMatches(mtime, target);
}
const handlers = {
    async file_search(params, ctx) {
        const folder = toText(params.folder) || ctx.cwd;
        const filename = toText(params.filename || params.name).trim().toLowerCase();
        const ext = toText(params.ext).trim().replace(/^\./, "").toLowerCase();
        const root = normalizePath(ctx.cwd, ctx.projectRoot, folder || ".");
        const paths = await walkFiles(root);
        const candidates = [];
        for (const filePath of paths) {
            const stat = await fs.stat(filePath);
            if (!timeMatches(stat, params.time))
                continue;
            const base = path.parse(filePath).name.toLowerCase();
            const suffix = path.extname(filePath).replace(/^\./, "").toLowerCase();
            if (ext && suffix !== ext)
                continue;
            if (filename && !base.includes(filename) && !path.basename(filePath).toLowerCase().includes(filename))
                continue;
            let score = 0;
            if (filename && base === filename)
                score += 100;
            if (filename && base.includes(filename))
                score += 40;
            if (ext)
                score += 20;
            if (folder)
                score += 5;
            candidates.push({ path: filePath, score, mtime: stat.mtimeMs });
        }
        candidates.sort((a, b) => b.score - a.score || b.mtime - a.mtime);
        const top = candidates[0];
        return {
            path: top?.path ?? null,
            files: candidates.slice(0, 20).map((item) => item.path),
            count: candidates.length,
            folder: root,
            query: { filename: filename || undefined, ext: ext || undefined, time: toText(params.time) || undefined },
        };
    },
    async read_file(params, ctx) {
        const filePath = normalizePath(ctx.cwd, ctx.projectRoot, toText(params.path));
        const text = await fs.readFile(filePath, "utf8");
        return { path: filePath, text };
    },
    async write_file(params, ctx) {
        const filePath = normalizePath(ctx.cwd, ctx.projectRoot, toText(params.path));
        await ensureDir(filePath);
        const content = toText(params.content);
        const mode = toText(params.mode).toLowerCase();
        if (mode === "append") {
            await fs.appendFile(filePath, content, "utf8");
        }
        else {
            await fs.writeFile(filePath, content, "utf8");
        }
        return { path: filePath, bytes: Buffer.byteLength(content, "utf8") };
    },
    async notes(params, ctx) {
        const title = toText(params.title).trim() || "untitled";
        const action = toText(params.action).toLowerCase();
        const notes = await readNotes(ctx.notesFile);
        const current = notes[title] || [];
        if (action === "read") {
            return { title, content: current.join("\n"), entries: current.length };
        }
        const content = toText(params.content);
        const next = action === "append" ? [...current, content] : [content];
        notes[title] = next;
        await writeNotes(ctx.notesFile, notes);
        return { title, entries: next.length, content: next.join("\n") };
    },
    async clipboard(params) {
        const action = toText(params.action).toLowerCase();
        if (process.platform === "win32") {
            if (action === "read") {
                const out = await runShell("powershell.exe", ["-NoProfile", "-Command", "Get-Clipboard"]);
                return { action, content: out.stdout.trim() };
            }
            await runShell("powershell.exe", ["-NoProfile", "-Command", `Set-Clipboard -Value @'\n${toText(params.content)}\n'@`]);
            return { action, content: toText(params.content) };
        }
        throw new Error("clipboard tool is only wired for Windows in this scaffold.");
    },
    async shell_command(params, ctx) {
        const command = toText(params.command);
        const shell = process.platform === "win32" ? "powershell.exe" : "/bin/sh";
        const args = process.platform === "win32" ? ["-NoProfile", "-Command", command] : ["-lc", command];
        const out = await runShell(shell, args, ctx.cwd);
        return { command, stdout: out.stdout, stderr: out.stderr, code: out.code };
    },
    async web_search(params) {
        const query = encodeURIComponent(toText(params.query));
        const url = `https://duckduckgo.com/html/?q=${query}`;
        const response = await fetch(url, { headers: { "user-agent": "Mozilla/5.0" } });
        const html = await response.text();
        const matches = [...html.matchAll(/<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g)];
        const results = matches.slice(0, 5).map((match) => ({
            title: decodeHtml(match[2]),
            url: normalizeDuckDuckGoUrl(match[1]),
        }));
        return { query: toText(params.query), results };
    },
    async browse(params) {
        const url = toText(params.url);
        const response = await fetch(url, { headers: { "user-agent": "Mozilla/5.0" } });
        const html = await response.text();
        const text = html
            .replace(/<script[\s\S]*?<\/script>/gi, " ")
            .replace(/<style[\s\S]*?<\/style>/gi, " ")
            .replace(/<[^>]+>/g, " ")
            .replace(/\s+/g, " ")
            .trim();
        return { url, text: text.slice(0, 8000) };
    },
    async email(params, ctx) {
        const to = toText(params.to);
        const subject = toText(params.subject) || "(no subject)";
        const body = toText(params.body);
        const attachments = Array.isArray(params.attachments) ? params.attachments.map((item) => toText(item)) : [];
        const smtpHost = process.env.BTL_SMTP_HOST?.trim();
        if (smtpHost) {
            const nodemailer = await import("nodemailer");
            const transport = nodemailer.createTransport({
                host: smtpHost,
                port: Number(process.env.BTL_SMTP_PORT || 587),
                secure: process.env.BTL_SMTP_SECURE === "true",
                auth: process.env.BTL_SMTP_USER
                    ? { user: process.env.BTL_SMTP_USER, pass: process.env.BTL_SMTP_PASS || "" }
                    : undefined,
            });
            await transport.sendMail({
                from: process.env.BTL_SMTP_FROM || process.env.BTL_SMTP_USER || "btl@localhost",
                to,
                subject,
                text: body,
                attachments: await Promise.all(attachments.map(async (filePath) => ({ filename: path.basename(filePath), path: normalizePath(ctx.cwd, ctx.projectRoot, filePath) }))),
            });
            return { transport: "smtp", to, subject, body, attachments };
        }
        const outbox = path.join(ctx.outboxDir, `${Date.now()}-${slugify(subject)}.json`);
        await ensureDir(outbox);
        await fs.writeFile(outbox, JSON.stringify({ to, subject, body, attachments }, null, 2), "utf8");
        return { transport: "outbox", to, subject, body, attachments, path: outbox };
    },
    async reasoning(params, ctx) {
        const question = toText(params.question);
        const context = toText(params.context);
        if (ctx.model) {
            const prompt = [
                "Answer the question clearly and briefly.",
                context ? `Context: ${context}` : null,
                `Question: ${question}`,
            ].filter(Boolean).join("\n");
            const answer = await ctx.model.complete(prompt);
            return { question, context: context || undefined, answer };
        }
        return {
            question,
            context: context || undefined,
            answer: context ? `${question} :: ${context}` : question,
        };
    },
};
async function runShell(cmd, args, cwd = process.cwd()) {
    return new Promise((resolve, reject) => {
        const child = spawn(cmd, args, { cwd, windowsHide: true });
        let stdout = "";
        let stderr = "";
        child.stdout.on("data", (chunk) => (stdout += String(chunk)));
        child.stderr.on("data", (chunk) => (stderr += String(chunk)));
        child.on("error", reject);
        child.on("close", (code) => resolve({ stdout, stderr, code: code ?? 0 }));
    });
}
function slugify(text) {
    return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64) || "email";
}
function decodeHtml(text) {
    return text
        .replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/&quot;/g, "\"")
        .replace(/&#39;/g, "'");
}
function normalizeDuckDuckGoUrl(raw) {
    try {
        const parsed = new URL(raw, "https://duckduckgo.com");
        const target = parsed.searchParams.get("uddg");
        if (target)
            return decodeURIComponent(target);
        return parsed.href;
    }
    catch {
        return raw;
    }
}
export function createToolHandlers() {
    return handlers;
}
export function resolveToolParams(step, results) {
    return resolveParams(step.params, results);
}
