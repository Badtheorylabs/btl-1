import json, os, random, math, sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_PATH = os.environ.get("BTL_TRAIN_PATH", str(PROJECT_ROOT / "data" / "real-traces" / "train.jsonl"))
EVAL_PATH = os.environ.get("BTL_EVAL_PATH", str(PROJECT_ROOT / "data" / "real-traces" / "eval.jsonl"))
BASE_MODEL = os.environ.get("BTL_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
OUTPUT_DIR = Path(os.environ.get("BTL_OUTPUT_DIR", PROJECT_ROOT / "artifacts"))
DEVICE = os.environ.get("BTL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

LR = float(os.environ.get("BTL_LR", "2e-4"))
WD = float(os.environ.get("BTL_WD", "0.0"))
WARMUP = float(os.environ.get("BTL_WARMUP", "0.05"))
MAX_GRAD = float(os.environ.get("BTL_MAX_GRAD", "1.0"))
MAX_LEN = int(os.environ.get("BTL_MAX_LEN", "1024"))
BATCH = int(os.environ.get("BTL_BATCH", "8"))
GRAD_ACCUM = int(os.environ.get("BTL_GRAD_ACCUM", "8"))
EPOCHS = int(os.environ.get("BTL_EPOCHS", "1"))
PREF_BETA = float(os.environ.get("BTL_PREF_BETA", "0.1"))
CONTRASTIVE_MARGIN = float(os.environ.get("BTL_CONTRASTIVE_MARGIN", "0.5"))
SFT_LIMIT = int(os.environ.get("BTL_SFT_LIMIT", "20000"))
DPO_LIMIT = int(os.environ.get("BTL_DPO_LIMIT", "7500"))
CONTRASTIVE_LIMIT = int(os.environ.get("BTL_CONTRASTIVE_LIMIT", "1500"))
CONTRASTIVE_SOURCE_GROUPS = int(os.environ.get("BTL_CONTRASTIVE_SOURCE_GROUPS", "800"))
DPO_SOURCE_GROUPS = int(os.environ.get("BTL_DPO_SOURCE_GROUPS", "4000"))
SEED = int(os.environ.get("BTL_SEED", "42"))

LORA_R = int(os.environ.get("BTL_LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("BTL_LORA_ALPHA", "128"))
LORA_DROPOUT = float(os.environ.get("BTL_LORA_DROPOUT", "0.05"))
LORA_TARGET = os.environ.get("BTL_LORA_TARGET", "q_proj,v_proj").split(",")
FLASH_ATTN = int(os.environ.get("BTL_FLASH_ATTN", "1"))

DRY_RUN = int(os.environ.get("BTL_DRY_RUN", "0"))

random.seed(SEED)
torch.manual_seed(SEED)

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def group_by_task(rows):
    groups = defaultdict(list)
    for r in rows:
        prov = r.get("provenance", {})
        uid = r.get("messages", [{}])[1] if len(r.get("messages", [])) > 1 else {}
        key = (uid.get("content", ""), prov.get("source_depth", -1))
        groups[key].append(r)
    return groups

@dataclass
class SFTBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor

@dataclass
class DPOBatch:
    chosen_ids: torch.Tensor
    chosen_mask: torch.Tensor
    rejected_ids: torch.Tensor
    rejected_mask: torch.Tensor

@dataclass
class ContrastiveBatch:
    anchor_ids: torch.Tensor
    anchor_mask: torch.Tensor
    positive_ids: torch.Tensor
    positive_mask: torch.Tensor
    negative_ids: torch.Tensor
    negative_mask: torch.Tensor

def tokenize_messages(messages, tokenizer):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return enc["input_ids"][0], enc["attention_mask"][0]

class PackedSFTDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        eos = tokenizer.eos_token_id
        flat = []
        for r in rows:
            prov = r.get("provenance", {})
            if prov.get("variant") not in ("minimal", "verbose"):
                continue
            msgs = r.get("messages", [])
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            ids = tokenizer.encode(text, truncation=True, max_length=MAX_LEN)
            flat.extend(ids)
            flat.append(eos)

        self.packed = []
        self.pad_counts = []
        for i in range(0, len(flat), MAX_LEN):
            chunk = flat[i:i + MAX_LEN]
            if len(chunk) < MAX_LEN // 2:
                continue
            npad = 0
            if len(chunk) < MAX_LEN:
                npad = MAX_LEN - len(chunk)
                chunk = chunk + [tokenizer.pad_token_id] * npad
            self.packed.append(chunk)
            self.pad_counts.append(npad)

    def __len__(self):
        return len(self.packed)

    def __getitem__(self, i):
        ids = torch.tensor(self.packed[i], dtype=torch.long)
        labels = ids.clone()
        if self.pad_counts[i] > 0:
            labels[-self.pad_counts[i]:] = -100
        mask = torch.ones(MAX_LEN, dtype=torch.long)
        if self.pad_counts[i] > 0:
            mask[-self.pad_counts[i]:] = 0
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

def collate_sft(batch, pad_id):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    return SFTBatch(input_ids, labels, attention_mask)

class DPODataset(Dataset):
    def __init__(self, groups, tokenizer):
        self.pairs = []
        for tasks in groups.values():
            by_var = {}
            for r in tasks:
                v = r.get("provenance", {}).get("variant")
                if v in ("minimal", "verbose"):
                    by_var[v] = r.get("messages", [])
            if "minimal" in by_var and "verbose" in by_var:
                c_ids, c_mask = tokenize_messages(by_var["minimal"], tokenizer)
                r_ids, r_mask = tokenize_messages(by_var["verbose"], tokenizer)
                self.pairs.append((c_ids, c_mask, r_ids, r_mask))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        c_ids, c_mask, r_ids, r_mask = self.pairs[i]
        return {"chosen_ids": c_ids, "chosen_mask": c_mask, "rejected_ids": r_ids, "rejected_mask": r_mask}

def collate_dpo(batch, pad_id):
    max_c = max(b["chosen_ids"].shape[0] for b in batch)
    max_r = max(b["rejected_ids"].shape[0] for b in batch)
    max_c = min(max_c, MAX_LEN)
    max_r = min(max_r, MAX_LEN)
    B = len(batch)
    c_ids = torch.full((B, max_c), pad_id, dtype=torch.long)
    c_mask = torch.zeros((B, max_c), dtype=torch.long)
    r_ids = torch.full((B, max_r), pad_id, dtype=torch.long)
    r_mask = torch.zeros((B, max_r), dtype=torch.long)
    for i, b in enumerate(batch):
        l = min(b["chosen_ids"].shape[0], max_c)
        c_ids[i, :l] = b["chosen_ids"][:l]
        c_mask[i, :l] = b["chosen_mask"][:l]
        l = min(b["rejected_ids"].shape[0], max_r)
        r_ids[i, :l] = b["rejected_ids"][:l]
        r_mask[i, :l] = b["rejected_mask"][:l]
    return DPOBatch(c_ids, c_mask, r_ids, r_mask)

class ContrastiveDataset(Dataset):
    def __init__(self, groups, tokenizer):
        self.triples = []
        for tasks in groups.values():
            by_var = {}
            for r in tasks:
                v = r.get("provenance", {}).get("variant")
                by_var[v] = r.get("messages", [])
            if all(v in by_var for v in ("minimal", "verbose", "negative")):
                a_ids, a_mask = tokenize_messages(by_var["minimal"], tokenizer)
                p_ids, p_mask = tokenize_messages(by_var["verbose"], tokenizer)
                n_ids, n_mask = tokenize_messages(by_var["negative"], tokenizer)
                self.triples.append((a_ids, a_mask, p_ids, p_mask, n_ids, n_mask))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, i):
        a_ids, a_mask, p_ids, p_mask, n_ids, n_mask = self.triples[i]
        return {"anc_ids": a_ids, "anc_mask": a_mask, "pos_ids": p_ids, "pos_mask": p_mask, "neg_ids": n_ids, "neg_mask": n_mask}

def collate_contrastive(batch, pad_id):
    max_a = max(b["anc_ids"].shape[0] for b in batch)
    max_p = max(b["pos_ids"].shape[0] for b in batch)
    max_n = max(b["neg_ids"].shape[0] for b in batch)
    max_a = min(max_a, MAX_LEN)
    max_p = min(max_p, MAX_LEN)
    max_n = min(max_n, MAX_LEN)
    B = len(batch)
    a_ids = torch.full((B, max_a), pad_id, dtype=torch.long)
    a_mask = torch.zeros((B, max_a), dtype=torch.long)
    p_ids = torch.full((B, max_p), pad_id, dtype=torch.long)
    p_mask = torch.zeros((B, max_p), dtype=torch.long)
    n_ids = torch.full((B, max_n), pad_id, dtype=torch.long)
    n_mask = torch.zeros((B, max_n), dtype=torch.long)
    for i, b in enumerate(batch):
        for tag, src_ids, src_mask in [("anc", b["anc_ids"], b["anc_mask"]), ("pos", b["pos_ids"], b["pos_mask"]), ("neg", b["neg_ids"], b["neg_mask"])]:
            l = min(src_ids.shape[0], {"anc": max_a, "pos": max_p, "neg": max_n}[tag])
            dst_ids = {"anc": a_ids, "pos": p_ids, "neg": n_ids}[tag]
            dst_mask = {"anc": a_mask, "pos": p_mask, "neg": n_mask}[tag]
            dst_ids[i, :l] = src_ids[:l]
            dst_mask[i, :l] = src_mask[:l]
    return ContrastiveBatch(a_ids, a_mask, p_ids, p_mask, n_ids, n_mask)

def get_hidden(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    last_hidden = out.hidden_states[-1]
    mask_expanded = attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
    return pooled

def contrastive_loss(anc, pos, neg, margin=0.5):
    d_pos = 1 - F.cosine_similarity(anc, pos)
    d_neg = 1 - F.cosine_similarity(anc, neg)
    loss = F.relu(d_pos - d_neg + margin).mean()
    return loss, d_pos.mean().item(), d_neg.mean().item()

def compute_log_probs(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    loss_mask = attention_mask[:, 1:]
    return (token_log_probs * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)

def dpo_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, beta=0.1):
    log_ratio = (policy_chosen_logp - ref_chosen_logp) - (policy_rejected_logp - ref_rejected_logp)
    loss = -F.log_sigmoid(beta * log_ratio).mean()
    acc = (log_ratio > 0).float().mean().item()
    return loss, acc

def eval_crr(model, eval_rows, tokenizer, depth_map=None):
    if depth_map is None:
        depth_map = {0: 0.3, 1: 0.5, 2: 0.5, 3: 0.7, 4: 0.8}
    model.eval()
    depth_scores = defaultdict(list)
    with torch.no_grad():
        for r in eval_rows[:200]:
            depth = r.get("provenance", {}).get("source_depth", -1)
            msgs = r.get("messages", [])
            prompt_msgs = msgs[:2]
            full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            gen_text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            target = msgs[2]["content"] if len(msgs) > 2 else ""
            score = 1.0 if gen_text.strip() == target.strip() else 0.0
            depth_scores[depth].append(score)
    model.train()
    in_dist = sum(depth_scores.get(d, [0]) for d in (1, 2))
    total_id = len(depth_scores.get(1, [])) + len(depth_scores.get(2, []))
    in_dist_avg = sum(in_dist) / max(total_id, 1)
    transfer = sum(depth_scores.get(d, [0]) for d in (3, 4))
    total_tr = len(depth_scores.get(3, [])) + len(depth_scores.get(4, []))
    transfer_avg = sum(transfer) / max(total_tr, 1)
    crr = transfer_avg / max(in_dist_avg, 1e-8)
    return crr, {"in_dist": in_dist_avg, "transfer": transfer_avg, "by_depth": {d: sum(v)/len(v) for d, v in depth_scores.items()}}

def setup_model():
    model, _ = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if FLASH_ATTN else None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    return model

def main():
    print(f"Loading data from {TRAIN_PATH}")
    train_rows = load_jsonl(TRAIN_PATH)
    eval_rows = load_jsonl(EVAL_PATH)
    print(f"  train: {len(train_rows)}  eval: {len(eval_rows)}")

    groups = group_by_task(train_rows)
    print(f"  task groups: {len(groups)}")

    print(f"Loading tokenizer + model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use most data for SFT; sample groups for DPO/contrastive.
    sft_dataset = PackedSFTDataset(train_rows[:SFT_LIMIT], tokenizer)
    group_list = list(groups.values())
    rng = random.Random(SEED)
    rng.shuffle(group_list)
    dpo_dataset = DPODataset({i: g for i, g in enumerate(group_list[:DPO_SOURCE_GROUPS])}, tokenizer)
    contra_dataset = ContrastiveDataset({i: g for i, g in enumerate(group_list[:CONTRASTIVE_SOURCE_GROUPS])}, tokenizer)
    if len(dpo_dataset) > DPO_LIMIT: dpo_dataset.pairs = dpo_dataset.pairs[:DPO_LIMIT]
    if len(contra_dataset) > CONTRASTIVE_LIMIT: contra_dataset.triples = contra_dataset.triples[:CONTRASTIVE_LIMIT]
    print(f"  SFT: {len(sft_dataset)}  DPO: {len(dpo_dataset)}  Contrastive: {len(contra_dataset)}")

    pad_id = tokenizer.pad_token_id
    sft_loader = DataLoader(sft_dataset, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate_sft(b, pad_id))
    dpo_loader = DataLoader(dpo_dataset, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate_dpo(b, pad_id)) if len(dpo_dataset) > 0 else None
    contra_loader = DataLoader(contra_dataset, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate_contrastive(b, pad_id)) if len(contra_dataset) > 0 else None

    model = setup_model()
    ref_model = setup_model()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sft_steps = len(sft_dataset) * EPOCHS // (BATCH * GRAD_ACCUM)
    dpo_steps = len(dpo_dataset) * EPOCHS // (BATCH * GRAD_ACCUM)
    contra_steps = len(contra_dataset) * EPOCHS // (BATCH * GRAD_ACCUM)
    total_steps = sft_steps + dpo_steps + contra_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * WARMUP), total_steps)

    # Precompute reference log probs for DPO
    ref_logp_cache = []
    if len(dpo_dataset) > 0:
        print("Precomputing reference log probs for DPO...")
        for i in range(len(dpo_dataset)):
            c_ids, c_mask, r_ids, r_mask = dpo_dataset.pairs[i]
            if len(ref_logp_cache) < 32 or i % 100 == 0:
                pass  # progress indicator
            with torch.no_grad():
                c_ids_b = c_ids.unsqueeze(0).to(ref_model.device)
                c_mask_b = c_mask.unsqueeze(0).to(ref_model.device)
                r_ids_b = r_ids.unsqueeze(0).to(ref_model.device)
                r_mask_b = r_mask.unsqueeze(0).to(ref_model.device)
                rc = compute_log_probs(ref_model, c_ids_b, c_mask_b)
                rr = compute_log_probs(ref_model, r_ids_b, r_mask_b)
            ref_logp_cache.append((rc.item(), rr.item()))
        print(f"  cached {len(ref_logp_cache)} DPO reference scores")
    else:
        print("  skipping DPO precomputation (no pairs)")

    if DRY_RUN:
        print("DRY RUN — evaluating before training, then exiting")
        crr, details = eval_crr(model, eval_rows, tokenizer)
        print(f"  CRR before: {crr:.4f}  details: {details}")
        sys.exit(0)

    step = 0
    global_step = 0
    for epoch in range(EPOCHS):
        # Stage 1: SFT
        print(f"\nEpoch {epoch+1}/{EPOCHS} — Stage 1: SFT")
        model.train()
        for batch in sft_loader:
            batch.input_ids = batch.input_ids.to(model.device)
            batch.labels = batch.labels.to(model.device)
            batch.attention_mask = batch.attention_mask.to(model.device)
            out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
            loss = out.loss
            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            step += 1
            if step % 100 == 0:
                print(f"  SFT step {step}: loss={loss.item():.4f}")

        # Stage 2: DPO (using cached ref logprobs)
        if dpo_loader is None:
            print(f"\nEpoch {epoch+1}/{EPOCHS} — Stage 2: DPO (skipped, no pairs)")
        else:
            print(f"\nEpoch {epoch+1}/{EPOCHS} — Stage 2: DPO")
            model.train()
            dpo_step = 0
            for batch_idx, batch in enumerate(dpo_loader):
                ref_c, ref_r = ref_logp_cache[batch_idx * BATCH:(batch_idx + 1) * BATCH] if batch_idx < len(ref_logp_cache) else (0.0, 0.0)
                # Handle last batch being smaller
                if batch_idx >= len(ref_logp_cache) // BATCH:
                    break

                batch.chosen_ids = batch.chosen_ids.to(model.device)
                batch.chosen_mask = batch.chosen_mask.to(model.device)
                batch.rejected_ids = batch.rejected_ids.to(model.device)
                batch.rejected_mask = batch.rejected_mask.to(model.device)

                pi_chosen = compute_log_probs(model, batch.chosen_ids, batch.chosen_mask)
                pi_rejected = compute_log_probs(model, batch.rejected_ids, batch.rejected_mask)

                ref_c_t = torch.tensor([rc for rc, _ in ref_logp_cache[batch_idx * BATCH:(batch_idx + 1) * BATCH]], device=model.device)
                ref_r_t = torch.tensor([rr for _, rr in ref_logp_cache[batch_idx * BATCH:(batch_idx + 1) * BATCH]], device=model.device)

                loss, acc = dpo_loss(pi_chosen, pi_rejected, ref_c_t, ref_r_t, beta=PREF_BETA)
                loss.backward()

                if (dpo_step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                dpo_step += 1
                if dpo_step % 50 == 0:
                    print(f"  DPO step {dpo_step}: loss={loss.item():.4f} acc={acc:.3f}")

        # Stage 3: Contrastive
        if contra_loader is None:
            print(f"\nEpoch {epoch+1}/{EPOCHS} — Stage 3: Contrastive (skipped, no triples)")
        else:
            print(f"\nEpoch {epoch+1}/{EPOCHS} — Stage 3: Contrastive")
            model.train()
            contra_step = 0
            for batch in contra_loader:
                batch.anchor_ids = batch.anchor_ids.to(model.device)
                batch.anchor_mask = batch.anchor_mask.to(model.device)
                batch.positive_ids = batch.positive_ids.to(model.device)
                batch.positive_mask = batch.positive_mask.to(model.device)
                batch.negative_ids = batch.negative_ids.to(model.device)
                batch.negative_mask = batch.negative_mask.to(model.device)

                anc = get_hidden(model, batch.anchor_ids, batch.anchor_mask)
                pos = get_hidden(model, batch.positive_ids, batch.positive_mask)
                neg = get_hidden(model, batch.negative_ids, batch.negative_mask)

                loss, dp, dn = contrastive_loss(anc, pos, neg, margin=CONTRASTIVE_MARGIN)
                loss.backward()

                if (contra_step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                contra_step += 1
                if contra_step % 25 == 0:
                    print(f"  Contrastive step {contra_step}: loss={loss.item():.4f} d_pos={dp:.4f} d_neg={dn:.4f}")

        # Eval after each epoch
        crr, details = eval_crr(model, eval_rows, tokenizer)
        print(f"\n  Epoch {epoch+1} CRR: {crr:.4f}  details: {details}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR / "compression-adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "compression-adapter"))
    print(f"\nSaved to {OUTPUT_DIR / 'compression-adapter'}")

    # Final eval
    crr, details = eval_crr(model, eval_rows, tokenizer)
    print(f"\nFinal CRR: {crr:.4f}  details: {json.dumps(details)}")

if __name__ == "__main__":
    main()
