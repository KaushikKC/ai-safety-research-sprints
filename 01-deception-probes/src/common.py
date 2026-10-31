"""Shared helpers: model loading, chat formatting, batched generation, activation extraction."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TAG = MODEL.split("/")[-1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ACTS = os.path.join(ROOT, "activations", TAG)
RESULTS = os.path.join(ROOT, "results", TAG)
for d in (DATA, ACTS, RESULTS):
    os.makedirs(d, exist_ok=True)


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device()).eval()
    return tok, model


def prompt_text(tok, messages):
    """Chat-templated prompt ending at the start of the assistant turn."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(tok, model, prompts, max_new_tokens=48, bs=16):
    """Greedy batched generation. prompts: list of chat-templated strings."""
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  generated {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
    return [o.strip() for o in outs]


@torch.no_grad()
def extract(tok, model, prompts, responses, bs=8):
    """Residual stream at every layer (incl. embeddings), pooled over response tokens only.

    Returns dict with 'mean' and 'last', each float16 tensor [N, n_layers+1, d_model].
    Prompt tokens are never pooled, so the probe cannot read the instruction directly.
    """
    means, lasts = [], []
    for i in range(0, len(prompts), bs):
        disk_guard()
        P, R = prompts[i:i + bs], responses[i:i + bs]
        p_lens = [len(tok(p, add_special_tokens=False).input_ids) for p in P]
        enc = tok([p + r for p, r in zip(P, R)], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        T = enc.input_ids.shape[1]
        # response mask (left padding): positions >= n_pad + prompt_len
        n_pad = (enc.attention_mask == 0).sum(1)
        start = torch.clamp(n_pad + torch.tensor(p_lens, device=model.device), max=T - 1)  # empty response -> last token
        mask = (torch.arange(T, device=model.device)[None] >= start[:, None]).float()[..., None]  # [B, T, 1]
        hidden = model(**enc, output_hidden_states=True).hidden_states  # tuple of L+1 x [B, T, d]
        # pool one layer at a time so we never materialise a [B, L+1, T, d] tensor
        means.append(torch.stack([(h.float() * mask).sum(1) / mask.sum(1) for h in hidden], 1).half().cpu())
        lasts.append(torch.stack([h[:, -1] for h in hidden], 1).half().cpu())
        del hidden, enc, mask
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"  extracted {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
    return {"mean": torch.cat(means), "last": torch.cat(lasts)}


def disk_guard(min_gb=2.0):
    import shutil
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < min_gb:
        raise SystemExit(f"ABORT: only {free:.1f} GB free disk (swap may be growing). Free space and rerun.")
