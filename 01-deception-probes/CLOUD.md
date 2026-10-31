# Running the big model on AWS

The pipeline is model-agnostic: `MODEL=<hf id> ./run_all.sh`. It uses CUDA automatically. On the 7B model, everything except probe training needs the GPU. Probe training is CPU-only and can run back on the laptop.

## 0. Do this first: GPU quota
New AWS accounts often have a **GPU quota of 0**. Go to Service Quotas → EC2 → "Running On-Demand G and VT instances" and request **8 vCPUs**. If you plan to use spot, also request "All G and VT Spot Instance Requests". Approval can take hours to a couple of days, so request it now.

## 1. Instance
| Instance | GPU | ~On-demand $/hr (us-east-1) | Fits |
|---|---|---|---|
| **g6.xlarge** (recommended) | L4 24 GB | ~$0.80 | Qwen2.5-7B, Llama-3.1-8B in bf16 |
| g5.xlarge | A10G 24 GB | ~$1.00 | same |
| g4dn.xlarge | T4 16 GB | ~$0.53 | **No**. 7B bf16 is 15 GB, too tight. |

- **AMI:** "Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)". It comes with torch and CUDA preinstalled.
- **Disk:** 100 GB gp3.
- **Spot:** costs ~60% less. The run is short, so an interruption only means rerunning.

Expected runtime for 7B is about 1–2 hours, so roughly **$2–4 total**. **Stop the instance when it finishes.**

## 2. Run
```bash
# from the laptop
rsync -av --exclude activations --exclude results --exclude logs \
  ~/Documents/Ai-Safety/ResearchGym/01-deception-probes/ ubuntu@<ip>:~/ResearchGym/

# on the instance
source /opt/pytorch/bin/activate        # path depends on AMI version; see the AMI's MOTD
pip install transformers scikit-learn pandas matplotlib datasets pyyaml
cd ResearchGym
MODEL=Qwen/Qwen2.5-7B-Instruct nohup ./run_all.sh &   # survives SSH disconnect
tail -f logs/run_*.log

# back on the laptop
rsync -av ubuntu@<ip>:~/ResearchGym/{results,activations,data,logs} ~/Documents/Ai-Safety/ResearchGym/01-deception-probes/
```
The 7B activations come to about 1–2 GB, so check the laptop's free disk before copying them back. Alternatively, copy only `results/` and `data/`.

## 3. Llama-3.1-8B
The Llama models are gated. Accept the license on its HF page, then run `huggingface-cli login` on the instance before `MODEL=meta-llama/Llama-3.1-8B-Instruct ./run_all.sh`.
