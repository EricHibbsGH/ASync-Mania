#!/usr/bin/env python3
"""
make_figures.py — Reproducible, publication-grade figures for the TLT review paper.

Every figure is generated from INLINE data arrays (auditable; no hand-drawn numbers).
Provenance per figure:
  [Paper Fig/Tbl]  = exact values transcribed from the TLT paper (arXiv:2511.16665v2).
  [Dashboard]      = analytical, assumption-based (Eric Hibbs Model Math). NOT measured.
  approx/SCHEMATIC = values read from a figure (OCR) or illustrative; not exact paper data.

Outputs: figures/*.pdf and figures/*.svg (vector). Style: matplotlib, Okabe-Ito
colorblind-safe palette, no chartjunk. Run:  python make_figures.py
Tested with matplotlib 3.8.3, numpy 1.26.4.
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito colorblind-safe palette
C = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
         blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#999999", black="#222222")

plt.rcParams.update({
    "figure.dpi": 120, "savefig.bbox": "tight", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "font.family": "DejaVu Sans",
})

def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf|svg")

# ---------------------------------------------------------------- Fig 11
def fig_endtoend():
    """[Paper Fig 11] End-to-end speedup vs VeRL(=1.0). Open-R1 bars approx (figure-read);
    some A100 Open-R1 cells illegible (n/a)."""
    models = ["Qwen-7B", "DeepSeek-7B", "Qwen-32B", "Llama-70B", "Geomean"]
    H = {"Open-R1": [0.25,0.22,0.07,0.18,0.16], "VeRL":[1,1,1,1,1],
         "TLT-Base":[1.42,1.38,1.54,1.31,1.41], "TLT":[1.76,1.71,2.12,1.86,1.86]}
    A = {"Open-R1": [0.24,0.21,np.nan,np.nan,0.17], "VeRL":[1,1,1,1,1],
         "TLT-Base":[1.35,1.23,1.49,1.31,1.38], "TLT":[1.79,1.66,2.07,1.73,1.70]}
    order = ["Open-R1","VeRL","TLT-Base","TLT"]
    col = {"Open-R1":C["grey"],"VeRL":C["sky"],"TLT-Base":C["orange"],"TLT":C["green"]}
    fig, axes = plt.subplots(1,2, figsize=(11,4.2), sharey=True)
    x = np.arange(len(models)); w = 0.2
    for ax,(title,D) in zip(axes, [("H100",H),("A100",A)]):
        for i,sys in enumerate(order):
            vals = np.array(D[sys], dtype=float)
            ax.bar(x+(i-1.5)*w, np.nan_to_num(vals), w, label=sys, color=col[sys],
                   hatch=("//" if sys=="Open-R1" else None), edgecolor="white", linewidth=0.5)
            for xi,v in zip(x+(i-1.5)*w, vals):
                if sys=="TLT" and not math.isnan(v):
                    ax.text(xi, v+0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=col["TLT"])
        ax.axhline(1.0, color=C["black"], lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{title}", fontsize=11); ax.set_ylim(0,2.4)
    axes[0].set_ylabel("Norm. training throughput (VeRL = 1.0)")
    axes[0].legend(ncol=4, loc="upper left", fontsize=8.5, bbox_to_anchor=(0,1.16))
    fig.suptitle("End-to-end RL training speedup  [Paper Fig 11]  (Open-R1 bars approx; A100 n/a = illegible)",
                 fontsize=10, y=1.02)
    save(fig, "fig11_endtoend_speedup")

# ---------------------------------------------------------------- Tbl 4
def fig_batch_erosion():
    """[Paper Tbl 4] SD speedup vs Tokens_to_Verify across batch sizes (Qwen-32B, TP4, depth10, topK8)."""
    verify = [16,32,48,64]
    BS = {1:[3.22,3.46,3.56,3.62], 2:[3.08,3.28,3.39,3.38], 4:[3.01,3.09,3.13,2.98],
          8:[2.73,2.63,2.51,2.27], 16:[2.67,2.52,2.24,1.91], 32:[2.48,2.23,1.90,1.70]}
    cmap = [C["blue"],C["sky"],C["green"],C["yellow"],C["orange"],C["verm"]]
    fig, ax = plt.subplots(figsize=(6.4,4.4))
    for (bs,vals),c in zip(BS.items(), cmap):
        ax.plot(verify, vals, "-o", color=c, label=f"BS={bs}", lw=1.8, ms=5)
    ax.axhline(1.0, color=C["black"], ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Tokens_to_Verify"); ax.set_ylabel("SD speedup ×")
    ax.set_xticks(verify); ax.set_ylim(1.5,3.8)
    ax.set_title("Batch-size erosion of SD speedup  [Paper Tbl 4]")
    ax.legend(ncol=2, fontsize=9)
    ax.annotate("3.62× (BS=1)", (64,3.62), textcoords="offset points", xytext=(-6,6), fontsize=8, color=C["blue"])
    ax.annotate("1.70× (BS=32)", (64,1.70), textcoords="offset points", xytext=(-10,-12), fontsize=8, color=C["verm"])
    save(fig, "tbl4_batch_erosion")

# ---------------------------------------------------------------- Tbl 2
def fig_gpu_diversity():
    """[Paper Tbl 2] Rollout SD speedup across GPUs (Qwen2.5-7B, BS=1, TP=1)."""
    gpus = ["B200","H100","A100","RTX5090","RTX4090","RTX3090"]
    sp = [2.33,2.61,2.79,2.91,2.87,3.22]
    cols = [C["blue"]]*3 + [C["orange"]]*3
    fig, ax = plt.subplots(figsize=(6.6,3.8))
    b = ax.bar(gpus, sp, color=cols, edgecolor="white")
    for r,v in zip(b,sp): ax.text(r.get_x()+r.get_width()/2, v+0.03, f"{v:.2f}×", ha="center", fontsize=8)
    ax.set_ylabel("Rollout speedup ×  (w/ SD vs w/o)"); ax.set_ylim(0,3.6)
    ax.set_title("GPU diversity: more I/O-bound GPUs gain more  [Paper Tbl 2]")
    ax.text(0.01,0.97,"data-center", transform=ax.transAxes, color=C["blue"], fontsize=8, va="top")
    ax.text(0.55,0.97,"consumer", transform=ax.transAxes, color=C["orange"], fontsize=8, va="top")
    save(fig, "tbl2_gpu_diversity")

# ---------------------------------------------------------------- Tbl 3
def fig_cluster_scaling():
    """[Paper Tbl 3] End-to-end speedup vs cluster size; 32B OOM at 1-2 nodes."""
    nodes = [1,2,4,8]
    q7 = [1.21,1.45,1.62,1.76]
    q32 = [np.nan,np.nan,1.83,2.12]
    fig, ax = plt.subplots(figsize=(6.0,4.0))
    ax.plot(nodes, q7, "-o", color=C["green"], lw=2, ms=6, label="Qwen2.5-7B")
    ax.plot(nodes, q32, "-s", color=C["verm"], lw=2, ms=6, label="Qwen2.5-32B")
    for n in (1,2):
        ax.text(n, 1.05, "OOM", color=C["verm"], ha="center", fontsize=8, rotation=90, va="bottom")
    for n,v in zip(nodes,q7): ax.text(n, v+0.03, f"{v:.2f}", ha="center", fontsize=7.5, color=C["green"])
    for n,v in zip(nodes,q32):
        if not math.isnan(v): ax.text(n, v+0.03, f"{v:.2f}", ha="center", fontsize=7.5, color=C["verm"])
    ax.set_xticks(nodes); ax.set_xlabel("Nodes (8 GPUs each)"); ax.set_ylabel("End-to-end speedup ×")
    ax.set_ylim(1.0,2.3); ax.set_title("Scalability: gains grow with cluster & model size  [Paper Tbl 3]")
    ax.legend(); save(fig, "tbl3_cluster_scaling")

# ---------------------------------------------------------------- Fig 17
def fig_spot_ladders():
    """[Paper Fig 17] Selective async checkpointing + sequence packing."""
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,3.8))
    names = ["Vanilla\nCkpt","Async\nCkpt","Selective\nAsync Ckpt"]; ms=[893,280,97]
    b1=ax1.bar(names, ms, color=[C["grey"],C["sky"],C["green"]], edgecolor="white")
    for r,v in zip(b1,ms): ax1.text(r.get_x()+r.get_width()/2, v+12, f"{v} ms", ha="center", fontsize=9)
    ax1.set_ylabel("Checkpoint latency (ms)"); ax1.set_ylim(0,1000)
    ax1.set_title("Selective async checkpointing")
    ax1.annotate("3.2×", (1,280), xytext=(0.45,560), textcoords="data", fontsize=9, color=C["black"])
    ax1.annotate("9.2×", (2,97), xytext=(1.45,560), textcoords="data", fontsize=9, color=C["black"])
    nm2=["Vanilla\nBatching","Sequence\nPacking"]; sps=[13.3,29.6]
    b2=ax2.bar(nm2, sps, color=[C["grey"],C["orange"]], edgecolor="white")
    for r,v in zip(b2,sps): ax2.text(r.get_x()+r.get_width()/2, v+0.5, f"{v}", ha="center", fontsize=9)
    ax2.set_ylabel("Training throughput (samples/s)"); ax2.set_ylim(0,34)
    ax2.set_title("Sequence packing (2.2×)")
    fig.suptitle("Spot-Trainer optimizations  [Paper Fig 17]", y=1.03, fontsize=10)
    save(fig, "fig17_spot_trainer")

# ---------------------------------------------------------------- Tbl 6
def fig_adaptive_drafter():
    """[Paper Tbl 6] Accept length: Target-Base -> Target-R (RL co-training)."""
    groups=["RL Training","Downstream"]; base=[4.59,3.76]; tr=[6.53,5.15]
    x=np.arange(len(groups)); w=0.34
    fig,ax=plt.subplots(figsize=(5.6,4.0))
    ax.bar(x-w/2, base, w, label="Target-Base", color=C["grey"], edgecolor="white")
    ax.bar(x+w/2, tr, w, label="Target-R (adaptive)", color=C["green"], edgecolor="white")
    for xi,v in zip(x-w/2,base): ax.text(xi,v+0.05,f"{v}",ha="center",fontsize=8.5)
    for xi,v in zip(x+w/2,tr): ax.text(xi,v+0.05,f"{v}",ha="center",fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("Avg. accept length"); ax.set_ylim(0,7.2)
    ax.set_title("Adaptive drafter accept length  [Paper Tbl 6]  (+42% RL)")
    ax.legend(); save(fig, "tbl6_adaptive_drafter")

# ---------------------------------------------------------------- Fig 13 heatmaps
def fig_sd_heatmaps():
    """[Paper Fig 13] interior cells read from figure (approx); extrema 9.78 / 3.66x verified & boxed."""
    depth=[4,6,8,10,12,16]; verify=[16,32,48,64,96,128]
    AL=np.array([[4.63,4.74,4.78,4.81,4.81,4.82],[5.71,6.08,6.16,6.31,6.31,6.36],
                 [6.34,7.01,7.24,7.39,7.49,7.51],[6.80,7.63,7.97,8.21,8.31,8.39],
                 [7.07,8.05,8.49,8.76,8.86,9.06],[7.25,8.44,8.97,9.23,9.51,9.78]])
    SP=np.array([[2.71,2.67,2.63,2.60,2.51,2.40],[3.09,3.16,3.15,3.17,3.06,2.94],
                 [3.19,3.40,3.43,3.47,3.39,3.29],[3.22,3.46,3.56,3.62,3.56,3.46],
                 [3.12,3.47,3.58,3.66,3.59,3.54],[2.94,3.26,3.41,3.48,3.49,3.45]])
    fig,axes=plt.subplots(1,2, figsize=(11,4.4))
    for ax,(M,ttl,box,fmt) in zip(axes,[(AL,"Avg. accept length",(5,5),"%.2f"),
                                        (SP,"Speedup x",(4,3),"%.2f")]):
        im=ax.imshow(M, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(len(verify))); ax.set_xticklabels(verify)
        ax.set_yticks(range(len(depth))); ax.set_yticklabels(depth)
        ax.set_xlabel("Tokens_to_Verify"); ax.set_ylabel("Draft_Depth")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j,i,fmt%M[i,j],ha="center",va="center",fontsize=6.5,
                        color="white" if M[i,j]<M.max()*0.75 else "black")
        bi,bj=box; ax.add_patch(Rectangle((bj-0.5,bi-0.5),1,1,fill=False,edgecolor=C["verm"],lw=2.2))
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("SD hyperparameter grid  [Paper Fig 13]  - interior cells figure-read (approx); "
                 "boxed extrema 9.78 / 3.66x verified", y=1.04, fontsize=9)
    save(fig, "fig13_sd_grid_heatmaps")

# ---------------------------------------------------------------- Fig 16 (schematic)
def fig_accept_decay():
    """[Paper Fig 16 / Dashboard] SCHEMATIC: only endpoints anchored (~32% adaptive vs ~6% vanilla @ idx 8)."""
    idx=np.arange(1,9)
    adaptive=np.array([99,92,82,72,63,52,42,32])   # SCHEMATIC trend; endpoint 32% (dashboard)
    vanilla =np.array([98,80,58,40,27,17,10,6])     # SCHEMATIC trend; endpoint 6% (dashboard)
    fig,ax=plt.subplots(figsize=(6.2,4.0))
    ax.plot(idx,adaptive,"-o",color=C["green"],lw=2,label="Adaptive drafter")
    ax.plot(idx,vanilla,"-s",color=C["verm"],lw=2,label="Vanilla drafter")
    ax.scatter([8,8],[32,6],s=70,facecolors="none",edgecolors=C["black"],zorder=5)
    ax.annotate("~32% (anchor)",(8,32),xytext=(-70,8),textcoords="offset points",fontsize=8,color=C["green"])
    ax.annotate("~6% (anchor)",(8,6),xytext=(-66,8),textcoords="offset points",fontsize=8,color=C["verm"])
    ax.set_xlabel("Draft token index"); ax.set_ylabel("Accept rate (%)"); ax.set_ylim(0,100)
    ax.set_title("Accept-rate decay  [Paper Fig 16 - SCHEMATIC; endpoints from Dashboard]")
    ax.legend(); save(fig, "fig16_accept_decay_SCHEMATIC")

# ---------------------------------------------------------------- Dashboard analytical surfaces
def fig_markov_surface():
    """[Dashboard Model Math 2 - assumption-based, NOT measured] speedup(alpha, gamma), c=8."""
    a=np.linspace(0.5,0.98,120); g=np.arange(1,25); c=8.0
    A,G=np.meshgrid(a,g)
    En=(1-A**(G+1))/(1-A); SP=En/(1+G/c)
    fig,ax=plt.subplots(figsize=(6.6,4.2))
    cs=ax.contourf(A,G,SP,levels=14,cmap="magma")
    ax.set_xlabel(r"per-token accept rate $\alpha$"); ax.set_ylabel(r"draft depth $\gamma$")
    ax.set_title("Markov SD speedup model  [Dashboard 2 - assumption-based; uniform alpha]")
    fig.colorbar(cs,ax=ax,label="modeled speedup x")
    save(fig, "dashboard_markov_surface")

def fig_threshold_curve():
    """[Dashboard Model Math 3 - assumption-based] aggregate benefit k*b(k)=b0*k*exp(-k/tau), peak at k=tau."""
    tau=16; k=np.linspace(0,80,400); b0=1.0
    g=b0*k*np.exp(-k/tau)
    fig,ax=plt.subplots(figsize=(6.2,4.0))
    ax.plot(k,g,color=C["blue"],lw=2)
    ax.axvline(tau,color=C["verm"],ls="--",lw=1.2); ax.text(tau+1,g.max()*0.9,r"$k=\tau=16$",color=C["verm"],fontsize=9)
    ax.axhline(0.25,color=C["black"],ls=":",lw=1); ax.text(60,0.27,r"$O_{SD}$ (illustrative)",fontsize=8)
    ax.set_xlabel("active requests k"); ax.set_ylabel(r"aggregate SD benefit $k\,b(k)$")
    ax.set_title("Optimal-stopping threshold model  [Dashboard 3 - assumption-based, tau=16]")
    save(fig, "dashboard_threshold_curve")

# ---------------------------------------------------------------- Long-tail schematic
def fig_longtail_schematic():
    """[Paper Fig 1a/Fig 2 - SCHEMATIC, not to scale] illustrative; raw per-request data not published."""
    rng=np.random.default_rng(7)
    x=np.linspace(0,30000,500)
    pdf=np.exp(-((np.log(x+1)-7.4)**2)/(2*0.62**2))/(x+1)  # log-normal-ish (illustrative)
    pdf/=pdf.max()
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.5,3.8))
    ax1.fill_between(x,pdf,color=C["sky"],alpha=0.6); ax1.axvline(20480,color=C["verm"],ls="--")
    ax1.text(20600,0.6,"max length\n(20,480)",color=C["verm"],fontsize=8)
    ax1.set_xlabel("Response length (tokens)"); ax1.set_ylabel("density (a.u.)")
    ax1.set_title("Long-tail length distribution")
    steps=np.arange(1,120)
    mx=3000+1500*np.abs(rng.standard_normal(len(steps)))
    p50=np.full_like(steps,1200,dtype=float); p75=np.full_like(steps,2200,dtype=float)
    ax2.fill_between(steps,p75,mx.clip(max=20480),color=C["orange"],alpha=0.35,label="p75-max (wasted)")
    ax2.plot(steps,p50,color=C["blue"],label="median"); ax2.plot(steps,p75,color=C["green"],label="p75")
    ax2.set_xlabel("RL step"); ax2.set_ylabel("Response length"); ax2.set_title("Under-utilized zone")
    ax2.legend(fontsize=8)
    fig.suptitle("SCHEMATIC - not to scale; illustrates Paper Fig 1a/Fig 2 (raw data not published as numbers)",
                 y=1.04, fontsize=9, color=C["verm"])
    save(fig, "fig1_longtail_SCHEMATIC")

if __name__ == "__main__":
    print("Generating figures ->", OUT)
    for f in (fig_endtoend, fig_batch_erosion, fig_gpu_diversity, fig_cluster_scaling,
              fig_spot_ladders, fig_adaptive_drafter, fig_sd_heatmaps, fig_accept_decay,
              fig_markov_surface, fig_threshold_curve, fig_longtail_schematic):
        f()
    print("Done.")
