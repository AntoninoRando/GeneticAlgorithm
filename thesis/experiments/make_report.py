"""Turn results.json into LaTeX tables and PDF figures for the thesis."""
import json, os, sys, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "thesis/experiments/results"
R = json.load(open(os.path.join(d, "results.json")))
runs = R["runs"]
ms = lambda xs: (st.fmean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)
fmt = lambda xs, p=3: "%.*f $\\pm$ %.*f" % ((p,) + ms(xs)[:1] + (p,) + ms(xs)[1:])
sel = lambda tag, v=None: [r for r in runs if r["tag"] == tag and (v is None or r["value"] == v)]
col = lambda rs, k, j: [r[k][j] for r in rs]
out = []

# --- main comparison -------------------------------------------------------
main = sel("main")
out.append("\\begin{tabular}{lccc}\\toprule\nMethod & Completion ratio $C$ & Mean response $R$ (min) & Energy $\\mathcal{E}$\\\\\\midrule")
for n, m in R["baselines"].items():
    out.append(f"{n} & {m[0]:.3f} & {m[1]:.2f} & {m[2]:.1f}\\\\")
rnd = [r["random"] for r in main]
out.append("Random trees (initial population) & %s & %s & %s\\\\" % tuple(fmt([x[j] for x in rnd], p) for j, p in ((0, 3), (1, 2), (2, 1))))
out.append("\\midrule\nEvolved GP (%d seeds) & %s & %s & %s\\\\\\bottomrule\n\\end{tabular}" %
           ((len(main),) + tuple(fmt(col(main, "test", j), p) for j, p in ((0, 3), (1, 2), (2, 1)))))
open(os.path.join(d, "tab_main.tex"), "w").write("\n".join(out))

# --- sweeps ----------------------------------------------------------------
LAB = dict(depth="Max.\\ depth", pop="Population", gens="Generations", hoist="Hoist rate",
           switch="World switch period (gen.)", weights="Weights $(w_C,w_R,w_E)$")
for sw in ["depth", "pop", "gens", "hoist", "switch", "weights"]:
    rows = ["\\begin{tabular}{lcccc}\\toprule\n%s & $C$ & $R$ (min) & $\\mathcal{E}$ & Tree size\\\\\\midrule" % LAB[sw]]
    vals = []
    for r in sel("sweep:" + sw):
        if r["value"] not in vals: vals.append(r["value"])
    for v in vals:
        rs = sel("sweep:" + sw, v)
        lab = ("$\\infty$" if v >= 10**9 else str(v)) if sw == "switch" else (str(tuple(v)) if sw == "weights" else str(v))
        rows.append("%s & %s & %s & %s & %.1f\\\\" % (lab, fmt(col(rs, "test", 0)), fmt(col(rs, "test", 1), 2),
                                                    fmt(col(rs, "test", 2), 1), st.fmean(r["size"] for r in rs)))
    rows.append("\\bottomrule\n\\end{tabular}")
    open(os.path.join(d, f"tab_{sw}.tex"), "w").write("\n".join(rows))

# --- convergence figure -----------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for j, (name, lab) in enumerate([("C", "Completion ratio"), ("R", "Mean response (min)"), ("E", "Energy")]):
    gens = [c[0] for c in main[0]["curve"]]
    mat = [[c[j + 1] for c in r["curve"]] for r in main]
    mu = [st.fmean(col_) for col_ in zip(*mat)]; sd = [st.stdev(col_) for col_ in zip(*mat)]
    ax[j].plot(gens, mu, label="GP best-of-generation"); ax[j].fill_between(gens, [a - b for a, b in zip(mu, sd)], [a + b for a, b in zip(mu, sd)], alpha=.2)
    for n, m in R["baselines"].items(): ax[j].axhline(m[j], ls="--", lw=.8, c="gray")
    ax[j].set_xlabel("Generation"); ax[j].set_title(lab, fontsize=9)
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(d, "convergence.pdf"))

# --- evolved trees ----------------------------------------------------------
best = sorted(main, key=lambda r: -r["test"][0])[:3]
open(os.path.join(d, "trees.tex"), "w").write(
    "\\begin{enumerate}\n" + "\n".join("\\item $C=%.3f$, size %d:\\\\\n{\\small\\texttt{%s}}" % (r["test"][0], r["size"],
        r["tree"].replace("_", "\\_").replace("&", "\\&")) for r in best) + "\n\\end{enumerate}")
print("mean time per default300 run: %.1fs" % st.fmean(r["seconds"] for r in sel("default300")))
