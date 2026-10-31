# Sprint 1: Linear probes for deception

Per-layer logistic-regression probes on **Qwen2.5-1.5B-Instruct** residual-stream activations, trained on instructed honest/dishonest pairs and tested out of distribution.

**Result:** the probe reaches AUC 1.00 in distribution, but it detects the *instruction to deceive*, not the lie. It scores truthful answers given under a deceive prompt as high as real lies (AUC 0.997). It reaches only 0.58–0.67 on uninstructed role-play deception, and catches 0% of deceptive examples at a 1% false-positive threshold.

- 📄 **Write-up:** [`writeup.md`](writeup.md)
- 🔮 **Pre-registered predictions:** [`predictions.md`](predictions.md)
- 🧪 **Pilot (model won't lie on command):** [`notes/pilot_compliance.md`](notes/pilot_compliance.md)
- 📈 **Figures and metrics:** [`results/Qwen2.5-1.5B-Instruct/`](results/Qwen2.5-1.5B-Instruct/)
- 🗺️ **Plan:** [`PLAN.md`](PLAN.md) · ☁️ **Larger models on AWS:** [`CLOUD.md`](CLOUD.md)

## Reproduce
```bash
pip install torch transformers scikit-learn pandas matplotlib datasets pyyaml
./run_all.sh                                   # ~1 h on an M3 MacBook (16 GB)
MODEL=Qwen/Qwen2.5-7B-Instruct ./run_all.sh    # on a 24 GB GPU
```
`src/build_and_extract.py` builds the datasets and saves pooled activations. `src/probes.py` trains the probes and writes the plots and `results.json`.

Datasets: Apollo Research `deception-detection` (instructed facts, roleplaying), Marks & Tegmark `geometry-of-truth` (cities), and Alpaca.
