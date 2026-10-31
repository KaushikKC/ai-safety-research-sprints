"""Train per-layer probes, evaluate in-distribution and OOD, write plots + results.json."""
import json, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from common import ACTS, RESULTS, DATA, TAG

C = 0.1
rng = np.random.default_rng(0)


def load(name, pool="mean"):
    d = torch.load(os.path.join(ACTS, f"{name}.pt"))
    rows = [json.loads(l) for l in open(os.path.join(DATA, TAG, f"{name}.jsonl"))]
    return d[pool].float().numpy(), d["labels"].numpy(), d["groups"].numpy(), rows


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def dom_fit(X, y):
    """Difference-of-means (mass-mean) probe from Marks & Tegmark."""
    mu = X.mean(0)
    return mu, X[y == 1].mean(0) - X[y == 0].mean(0)


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def cv_by_layer(X, y, g, shuffle=False):
    """Grouped 5-fold CV (honest+deceptive versions of one question stay in the same fold)."""
    m = y >= 0
    X, y, g = X[m], y[m], g[m]
    if shuffle:
        y = rng.permutation(y)
    out = {"lr": [], "dom": []}
    for L in range(X.shape[1]):
        a_lr, a_dom = [], []
        for tr, te in GroupKFold(5).split(X[:, L], y, g):
            a_lr.append(auc(y[te], lr().fit(X[tr, L], y[tr]).decision_function(X[te, L])))
            mu, d = dom_fit(X[tr, L], y[tr])
            a_dom.append(auc(y[te], (X[te, L] - mu) @ d))
        out["lr"].append((np.mean(a_lr), np.std(a_lr)))
        out["dom"].append((np.mean(a_dom), np.std(a_dom)))
    return out


def fit_all_layers(X, y):
    m = y >= 0
    return [lr().fit(X[m, L], y[m]) for L in range(X.shape[1])]


def scores(probes, X):
    return np.stack([p.decision_function(X[:, L]) for L, p in enumerate(probes)], 1)  # [N, L]


def main():
    res = {"model": TAG, "C": C}
    A, yA, gA, _ = load("setA_instructed")
    B, yB, gB, rowsB = load("setB_cities")
    n_layers = A.shape[1]
    layers = np.arange(n_layers)

    # ---------- 1. In-distribution AUC by layer
    print("CV setA", flush=True); cvA = cv_by_layer(A, yA, gA)
    print("CV setB", flush=True); cvB = cv_by_layer(B, yB, gB)
    print("CV shuffled", flush=True); cvA_shuf = cv_by_layer(A, yA, gA, shuffle=True)
    A_last, *_ = load("setA_instructed", "last")
    print("CV setA last-token", flush=True); cvA_last = cv_by_layer(A_last, yA, gA)
    res["cv"] = {"setA": cvA, "setB": cvB, "setA_shuffled": cvA_shuf, "setA_lasttoken": cvA_last}

    # bag-of-words baseline on Set B response text (can the lie be read off the tokens alone?)
    m = yB >= 0
    texts = np.array([r["response"] for r in rowsB])[m]
    bow = []
    for tr, te in GroupKFold(5).split(texts, yB[m], gB[m]):
        v = CountVectorizer(min_df=2).fit(texts[tr])
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(v.transform(texts[tr]), yB[m][tr])
        bow.append(auc(yB[m][te], clf.decision_function(v.transform(texts[te]))))
    res["setB_bow_auc"] = float(np.mean(bow))

    best_A = int(np.argmax([a for a, _ in cvA["lr"]]))
    res["best_layer_setA"] = best_A

    # ---------- 2. Train on Set A, evaluate OOD at every layer
    probesA = fit_all_layers(A, yA)
    alp, *_ = load("alpaca")
    s_alp = scores(probesA, alp)
    thr = np.quantile(s_alp, 0.99, axis=0)  # 1% FPR on ordinary chat traffic, per layer

    ood = {}
    kinds = np.array([r["kind"] for r in rowsB])
    sB = scores(probesA, B)
    hon = kinds == "honest_true"
    for name, pos in [("setB_lie_vs_honest", kinds == "deceive_false"),
                      ("setB_compliedhonestly_vs_honest", kinds == "deceive_true"),
                      ("setB_honestmistake_vs_honest", kinds == "honest_false")]:
        sel = hon | pos
        if pos.sum() == 0:
            continue
        ood[name] = {"auc": [auc(pos[sel].astype(int), sB[sel, L]) for L in layers],
                     "recall_at_1pct_fpr": [float((sB[pos, L] > thr[L]).mean()) for L in layers],
                     "n_pos": int(pos.sum())}
    for name in ["roleplay", "got_cities"]:
        X, y, *_ = load(name)
        s = scores(probesA, X)
        ood[name] = {"auc": [auc(y, s[:, L]) for L in layers],
                     "recall_at_1pct_fpr": [float((s[y == 1, L] > thr[L]).mean()) for L in layers],
                     "n_pos": int((y == 1).sum())}
    res["ood_trainA"] = ood
    # 2x2 mean probe score per Set B cell at the best Set A layer (prompt x answer truth)
    res["setB_2x2_mean_score"] = {k: float(sB[kinds == k, best_A].mean()) for k in sorted(set(kinds))}
    res["setB_2x2_flag_rate_1pct"] = {k: float((sB[kinds == k, best_A] > thr[best_A]).mean()) for k in sorted(set(kinds))}
    res["alpaca_threshold_best_layer"] = float(thr[best_A])

    # reverse direction: train on Set B, test on roleplay
    probesB = fit_all_layers(B, yB)
    Xr, yr, *_ = load("roleplay")
    res["ood_trainB_roleplay_auc"] = [auc(yr, s) for s in scores(probesB, Xr).T]

    # on-policy roleplay: dump probe scores at best layer for manual reading
    Xo, _, _, rows_o = load("roleplay_onpolicy")
    so = scores(probesA, Xo)[:, best_A]
    order = np.argsort(-so)
    with open(os.path.join(RESULTS, "roleplay_onpolicy_scored.jsonl"), "w") as f:
        for i in order:
            f.write(json.dumps({"score": float(so[i]), "flagged_1pct": bool(so[i] > thr[best_A]),
                                **rows_o[i]}) + "\n")
    res["roleplay_onpolicy_flag_rate_1pct"] = float((so > thr[best_A]).mean())

    json.dump(res, open(os.path.join(RESULTS, "results.json"), "w"), indent=1)

    # ---------- 3. Plots
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, lab, c, ls in [("setA", "Set A instructed (LR)", "C0", "-"), ("setB", "Set B on-policy cities (LR)", "C1", "-"),
                            ("setA_lasttoken", "Set A, last token only", "C0", ":"),
                            ("setA_shuffled", "Set A, shuffled labels", "gray", "--")]:
        mu = np.array([a for a, _ in res["cv"][key]["lr"]]); sd = np.array([s for _, s in res["cv"][key]["lr"]])
        ax.plot(layers, mu, ls, color=c, label=lab); ax.fill_between(layers, mu - sd, mu + sd, color=c, alpha=.15)
    ax.plot(layers, [a for a, _ in cvA["dom"]], "-.", color="C2", label="Set A diff-of-means")
    ax.axhline(res["setB_bow_auc"], color="C1", lw=.8, ls="--", label=f"Set B bag-of-words ({res['setB_bow_auc']:.2f})")
    ax.set(xlabel="layer (0 = embeddings)", ylabel="held-out AUC (grouped 5-fold)", ylim=(0.3, 1.02),
           title=f"{TAG}: in-distribution probe AUC by layer")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig1_auc_by_layer.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, [a for a, _ in cvA["lr"]], color="k", label="Set A in-dist (CV)")
    for i, (k, v) in enumerate(ood.items()):
        ax.plot(layers, v["auc"], color=f"C{i+1}", label=f"{k} (n+={v['n_pos']})")
    ax.plot(layers, res["ood_trainB_roleplay_auc"], ":", color="C9", label="train Set B → roleplay")
    ax.axhline(.5, color="gray", lw=.6)
    ax.set(xlabel="layer", ylabel="AUC", ylim=(0, 1.02), title=f"{TAG}: probe trained on Set A, evaluated OOD")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig2_ood_by_layer.png"), dpi=150)

    # ---------- 4. Console summary
    print(f"\nbest Set A layer: {best_A}  AUC={cvA['lr'][best_A][0]:.3f}")
    print(f"Set A layer0 AUC={cvA['lr'][0][0]:.3f}  Set B layer0 AUC={cvB['lr'][0][0]:.3f}  Set B BoW AUC={res['setB_bow_auc']:.3f}")
    print(f"Set B best layer {int(np.argmax([a for a,_ in cvB['lr']]))} AUC={max(a for a,_ in cvB['lr']):.3f}")
    for k, v in ood.items():
        print(f"OOD {k:35s} AUC@L{best_A}={v['auc'][best_A]:.3f}  max={np.nanmax(v['auc']):.3f}@L{int(np.nanargmax(v['auc']))}"
              f"  recall@1%FPR={v['recall_at_1pct_fpr'][best_A]:.2f}")
    print("Set B 2x2 mean score @best layer:", {k: round(v, 2) for k, v in res["setB_2x2_mean_score"].items()})
    print("Set B 2x2 flag rate @1%FPR:", {k: round(v, 2) for k, v in res["setB_2x2_flag_rate_1pct"].items()})
    print(f"train B -> roleplay max AUC={np.nanmax(res['ood_trainB_roleplay_auc']):.3f}")
    print(f"on-policy roleplay flagged @1%FPR: {res['roleplay_onpolicy_flag_rate_1pct']:.2f}")


if __name__ == "__main__":
    main()
