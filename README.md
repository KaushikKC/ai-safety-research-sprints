# AI Safety Research Sprints

A series of short, hands-on research sprints in AI safety. Each sprint has pre-registered predictions, reproducible code, and a write-up that reports where the method fails.

| # | Sprint | Status | Headline |
|---|---|---|---|
| 01 | [Linear probes for deception](01-deception-probes/) | ✅ Done | On Qwen2.5-1.5B-Instruct, the probe detects the *instruction to deceive*, not the lie (AUC 0.997 on truthful answers under a deceive prompt; 0% recall at 1% FPR). |
| 02 | Emergent misalignment (LoRA model organisms) | ⏳ Next | |

Hardware: Apple M3 MacBook (16 GB) for small models. AWS GPU instances for 7B+ models (see each sprint's `CLOUD.md`).
