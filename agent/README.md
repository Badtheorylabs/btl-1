# BTL-1 Agent

Node.js CLI scaffold for the local BTL-1 tool agent.

## Dev

```bash
cd btl-1/agent
npm install
npm run dev -- --prompt "Search for vector databases and email the summary to Hank"
```

## Notes

- `BTL_MODEL_PATH` should point at a local GGUF file.
- `BTL_MODEL_BIN` defaults to `llama-cli`.
- If SMTP env vars are missing, `email` writes to `.btl-outbox/` instead of sending.
