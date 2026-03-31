"""
figures/system.py
-----------------
Sistem figürleri: quarter-car modeli, quantum devre, fuzzy MF.

Çalıştırma:
    python -m figures.system
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from figures._base import set_style, save

OUT = "figures/system"


def quarter_car():
    set_style()
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.set_xlim(0, 10); ax.set_ylim(-0.8, 12.5); ax.axis("off")

    def spring(x, y1, y2, n=5, col="k"):
        pts = n * 4 + 2
        t   = np.linspace(0, 1, pts)
        amp = 0.25
        zz  = amp * np.sin(np.linspace(0, n * 2 * np.pi, pts))
        ax.plot(x + zz, y1 + t * (y2 - y1), color=col, lw=1.2, zorder=2)

    def damper(x, y1, y2):
        mid = (y1 + y2) / 2
        ax.plot([x, x], [y1, mid - 0.3], "k-", lw=1.2)
        ax.plot([x, x], [mid + 0.3, y2], "k-", lw=1.2)
        ax.add_patch(plt.Rectangle((x-.3, mid-.3), .6, .6,
                     fc="white", ec="k", lw=1.2, zorder=3))

    # Road
    ax.add_patch(plt.Rectangle((1, 0), 8, .5, fc="#AAAAAA", ec="k", lw=1))
    ax.text(5, -.3, r"$z_r(t)$ — Road profile", ha="center", fontsize=9)
    # Tire spring
    spring(4, .5, 2.2, n=4, col="#555")
    ax.text(4.6, 1.4, r"$k_t$", fontsize=10)
    # Unsprung mass
    ax.add_patch(plt.Rectangle((2.5, 2.2), 3, .8,
                 fc="#AACCEE", ec="k", lw=1.5, zorder=3))
    ax.text(4, 2.6, r"$m_u$ (unsprung)", ha="center", va="center",
            fontsize=9, fontweight="bold")
    # Suspension
    spring(3.0, 3.0, 5.8, n=4, col="#555")
    ax.text(2.3, 4.4, r"$k_s$", fontsize=10)
    damper(4.8, 3.0, 5.8)
    ax.text(5.2, 4.4, r"$c_s$", fontsize=10)
    ax.annotate("", xy=(4.0, 5.6), xytext=(4.0, 3.2),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.text(3.3, 4.7, r"$f_a$", fontsize=10, color="blue", fontweight="bold")
    # Sprung mass
    ax.add_patch(plt.Rectangle((2.5, 5.8), 3, 1.0,
                 fc="#FFDDAA", ec="k", lw=1.5, zorder=3))
    ax.text(4, 6.3, r"$m_s$ (sprung)", ha="center", va="center",
            fontsize=9, fontweight="bold")
    # State labels
    for y, lbl in [(2.6, r"$z_u$"), (6.3, r"$z_s$")]:
        ax.annotate("", xy=(6.8, y), xytext=(7.8, y),
                    arrowprops=dict(arrowstyle="<->", color="#333"))
        ax.text(8.0, y, lbl, fontsize=11, va="center")
    # Controller
    ax.add_patch(mpatches.FancyBboxPatch(
        (.8, 9.2), 8.4, 1.5, boxstyle="round,pad=.15",
        fc="#DDEEFF", ec="#2255BB", lw=1.8, zorder=4))
    ax.text(5, 9.95,
            "SAC / TD3 Controller\n+ QFM Encoder  +  Fuzzy Reward",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color="#1144AA", zorder=5)
    ax.annotate("", xy=(4, 9.2), xytext=(4, 6.8),
                arrowprops=dict(arrowstyle="->", color="#3366CC", lw=1.5,
                                connectionstyle="arc3,rad=.3"))
    ax.text(5.4, 7.9, r"$[z_s,z_u,\dot z_s,\dot z_u,z_r]$",
            fontsize=8, color="#3366CC")
    ax.annotate("", xy=(3.5, 5.8), xytext=(3.5, 9.2),
                arrowprops=dict(arrowstyle="<-", color="blue", lw=1.5,
                                connectionstyle="arc3,rad=-.3"))
    ax.text(1.2, 7.8, r"$f_a$", fontsize=10, color="blue", fontweight="bold")
    ax.set_title("Quarter-Car Active Suspension — Control Architecture",
                 fontsize=11, pad=8)
    save(f"{OUT}/fig1_quarter_car.png", fig)


def quantum_circuit():
    set_style()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(-.8, 6.0); ax.axis("off")

    q_labels = [r"$q_0: z_s$", r"$q_1: z_u$",
                r"$q_2: \dot{z}_s$", r"$q_3: \dot{z}_u$",
                r"$q_4: z_r$"]
    colors   = ["#4878CF","#6ACC65","#D65F5F","#B47CC7","#77BEDB"]

    def gate(x, y, txt, fc="#DDEEFF", ec="#3366CC", w=1.2, h=.45):
        ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h,
                     fc=fc, ec=ec, lw=1.0, zorder=4))
        ax.text(x, y, txt, ha="center", va="center", fontsize=7.5, zorder=5)

    def cnot(x, yc, yt):
        ax.plot(x, yc, "ko", ms=7, zorder=5)
        ax.plot([x, x], [yc, yt], "k-", lw=1.2, zorder=3)
        ax.add_patch(plt.Circle((x, yt), .18, fc="white", ec="k", lw=1.2, zorder=4))
        ax.plot([x-.18, x+.18], [yt, yt], "k-", lw=1.0, zorder=5)
        ax.plot([x, x], [yt-.18, yt+.18], "k-", lw=1.0, zorder=5)

    for i, (lbl, col) in enumerate(zip(q_labels, colors)):
        y = 4 - i
        ax.plot([.9, 11.0], [y, y], "-", color="#DDDDDD", lw=1.5, zorder=1)
        ax.text(.85, y, lbl, ha="right", va="center", fontsize=9, color=col)

    hdrs = [(2.0,"Layer 1\nRY Encoding"),(4.3,"Layer 2\nPhysics CNOT"),
            (7.0,"Layer 3\nVariational"),(9.8,"Layer 4\nMeasure")]
    for xh, lbl in hdrs:
        ax.text(xh, 5.5, lbl, ha="center", fontsize=8.5, fontweight="bold",
                color="#333", bbox=dict(fc="#F5F5F5", ec="#CCCCCC", pad=2, lw=.5))

    for i in range(5):
        gate(2.0, 4-i, r"$RY(\pi x_"+str(i)+r")$", fc="#DDEEFF", ec="#3366CC")

    for ctrl, tgt, xc in [(0,1,3.6),(2,3,3.9),(1,4,4.4),(0,2,4.7)]:
        cnot(xc, 4-ctrl, 4-tgt)

    for i in range(5):
        gate(6.4, 4-i, r"$RZ(\theta_"+str(i)+r")$", fc="#FFEEDD", ec="#CC6633")
        gate(7.7, 4-i, r"$RX(\phi_"+str(i)+r")$",  fc="#FFEEDD", ec="#CC6633")

    for i in range(5):
        gate(9.8, 4-i, r"$\langle Z\rangle$", fc="#EEFFEE", ec="#33AA33", w=.8)

    ax.text(10.7, 4,   r"$\langle Z_0\rangle$",   va="center", fontsize=8)
    ax.text(10.7, 2.5, r"$\vdots$",               va="center", fontsize=9)
    ax.text(10.7, 1.5, r"$\langle Z_0Z_1\rangle$",va="center", fontsize=8)
    ax.text(10.7, 1.0, r"$\langle Z_2Z_3\rangle$",va="center", fontsize=8)
    ax.text(10.7, 0.5, r"$\langle Z_1Z_4\rangle$",va="center", fontsize=8)

    # QFM note
    ax.text(5.5, -.5,
            "QFM mode: variational parameters are fixed (frozen) — "
            "no gradient, ~100× faster than VQC.",
            ha="center", fontsize=8.5, color="#555", style="italic")

    ax.set_title(r"Physics-Guided Quantum Feature Map ($n=5$ qubits)",
                 fontsize=11)
    save(f"{OUT}/fig2_quantum_circuit.png", fig)


def fuzzy_membership():
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    def trimf(x, a, b, c):
        return np.maximum(0., np.minimum((x-a)/max(b-a,1e-9),
                                         (c-x)/max(c-b,1e-9)))

    p95, mx = 0.9116, 4.27
    x = np.linspace(0, 5.5, 600)

    ax = axes[0]
    ax.plot(x, trimf(x,0,0,p95),               "#4878CF", lw=2, label="LOW")
    ax.plot(x, trimf(x,0,p95*.5,p95),          "#6ACC65", lw=2, label="MED")
    ax.plot(x, trimf(x,p95*.5,p95,max(mx*1.5,p95*4.0)),
            "#D65F5F", lw=2, label="HIGH")
    ax.axvline(p95, ls="--", color="gray", lw=1, label=fr"$p_{{95}}$={p95:.2f}")
    ax.axvline(mx,  ls=":",  color="gray", lw=1, label=fr"max={mx:.2f}")
    ax.set_xlabel(r"$|a_b|$ (m/s²)")
    ax.set_ylabel("Membership degree")
    ax.set_title(r"Input: Body acceleration $|a_b|$")
    ax.legend(fontsize=8); ax.set_xlim(0,5.5); ax.set_ylim(-.05,1.15)

    x2 = np.linspace(.4, 2.0, 400)
    ax2 = axes[1]
    ax2.plot(x2, trimf(x2,.5,.8,1.0),   "#4878CF", lw=2, label="LOW (0.8)")
    ax2.plot(x2, trimf(x2,.9,1.15,1.35),"#6ACC65", lw=2, label="MED (1.15)")
    ax2.plot(x2, trimf(x2,1.1,1.5,1.8), "#D65F5F", lw=2, label="HIGH (1.5)")
    ax2.axvline(1.0, ls="--", color="gray", lw=1,
                label=r"Baseline $\alpha=1$")
    ax2.set_xlabel(r"Fuzzy coefficient $\alpha$")
    ax2.set_ylabel("Membership degree")
    ax2.set_title(r"Output: Reward coefficient $\alpha$ (bounded $[0.8,1.5]$)")
    ax2.legend(fontsize=8); ax2.set_xlim(.4,2.0); ax2.set_ylim(-.05,1.15)

    plt.suptitle("Fuzzy Membership Functions (Mamdani Inference)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    save(f"{OUT}/fig3_fuzzy_membership.png", fig)


def main():
    print("=== System Figures ===")
    quarter_car()
    quantum_circuit()
    fuzzy_membership()
    print("Done →", OUT)


if __name__ == "__main__":
    main()
