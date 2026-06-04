"""
Heterogeneous Sensor Placement for 2D Barrier Coverage
=======================================================
Extension of Kim et al. (2025) — now with two sensor types and a budget limit.

Two methods compared:
  1. Budget-Aware Greedy  → picks the best sensor type + location step by step
  2. Uniform Placement    → spreads cheap sensors evenly (baseline)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import multivariate_normal

np.random.seed(42)

# ──────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────
AREA_KM      = 20.0   # total area: -10 to +10 km in both axes
GRID_STEP    = 0.5    # spacing between candidate sensor positions (km)
TOTAL_BUDGET = 10.0   # total money/resource units available

# Two sensor types: long-range expensive (A) vs short-range cheap (B)
SENSORS = {
    'A': {'label': 'Type A – Long Range',  'rho': 0.95, 'sigma': 2.5, 'cost': 3.0, 'color': 'red'},
    'B': {'label': 'Type B – Short Range', 'rho': 0.80, 'sigma': 1.0, 'cost': 1.0, 'color': 'blue'},
}

# Representation space axes
ALPHA = np.linspace(0, np.pi, 73)    # heading angle: 0° to 180°
P     = np.linspace(-10, 10, 41)     # perpendicular distance: -10 to +10 km


# ──────────────────────────────────────────────────────────────
# STEP 1 — BUILD TARGET INTENSITY MAP
# Simulates where ships/targets travel (two busy corridors)
# ──────────────────────────────────────────────────────────────
def build_intensity():
    AA, PP = np.meshgrid(ALPHA, P, indexing='ij')

    # Corridor 1: ships heading ~80°, near p = +2 km
    lane1 = multivariate_normal(mean=[np.deg2rad(80),  2.0],
                                cov=[[np.deg2rad(15)**2, 0], [0, 1.5**2]])
    # Corridor 2: ships heading ~100°, near p = -3 km
    lane2 = multivariate_normal(mean=[np.deg2rad(100), -3.0],
                                cov=[[np.deg2rad(10)**2, 0], [0, 1.0**2]])

    pts = np.stack([AA.ravel(), PP.ravel()], axis=1)
    raw = 8.0 * lane1.pdf(pts) + 5.0 * lane2.pdf(pts)
    intensity = raw.reshape(AA.shape)

    # Scale so total ~ 15 targets per time period
    da, dp = ALPHA[1] - ALPHA[0], P[1] - P[0]
    intensity = intensity / intensity.sum() * 15.0 / da / dp
    return intensity


# ──────────────────────────────────────────────────────────────
# STEP 2 — HOW WELL DOES ONE SENSOR DETECT A TRAJECTORY?
# Returns detection probability for every (alpha, p) point
# ──────────────────────────────────────────────────────────────
def detection_prob(alpha_grid, p_grid, sx, sy, rho, sigma):
    # Closest approach distance from sensor (sx, sy) to line (alpha, p)
    tan_a = np.tan(alpha_grid - np.pi / 2 + 1e-9)
    p_closest = (sy + sx / (tan_a + 1e-9)) / np.sqrt(1 + 1 / (tan_a**2 + 1e-9))
    dist_sq = (p_grid - p_closest) ** 2
    return rho * np.exp(-dist_sq / sigma)


# ──────────────────────────────────────────────────────────────
# STEP 3 — VOID PROBABILITY
# Probability that a random target escapes all sensors undetected
# Lower void probability = better sensor network
# ──────────────────────────────────────────────────────────────
def void_probability(intensity, placed_sensors):
    AA, PP = np.meshgrid(ALPHA, P, indexing='ij')
    da, dp = ALPHA[1] - ALPHA[0], P[1] - P[0]

    # Start with "no sensor detects anything" = 1 everywhere
    prob_miss = np.ones_like(intensity)

    for (sx, sy, stype) in placed_sensors:
        s = SENSORS[stype]
        gamma = detection_prob(AA, PP, sx, sy, s['rho'], s['sigma'])
        prob_miss *= (1.0 - gamma)  # multiply miss probabilities

    integral = np.sum(intensity * prob_miss) * da * dp
    return np.exp(-integral)


# ──────────────────────────────────────────────────────────────
# METHOD 1 — BUDGET-AWARE GREEDY
# Each iteration: try every (location × sensor type) combo,
# pick whichever improves detection the most per cost unit
# ──────────────────────────────────────────────────────────────
def greedy_placement(intensity, budget):
    candidates_x = np.arange(-10, 10.5, GRID_STEP)
    candidates_y = np.arange(-10, 10.5, GRID_STEP)

    placed  = []
    history = []   # void probability after each sensor added
    remaining = budget
    min_cost = min(s['cost'] for s in SENSORS.values())

    while remaining >= min_cost:
        best_vp    = -1
        best_pick  = None
        current_vp = void_probability(intensity, placed)

        for stype, s in SENSORS.items():
            if s['cost'] > remaining:
                continue
            for x in candidates_x:
                for y in candidates_y:
                    vp = void_probability(intensity, placed + [(x, y, stype)])
                    if vp > best_vp:
                        best_vp   = vp
                        best_pick = (x, y, stype)

        if best_pick is None:
            break

        placed.append(best_pick)
        remaining -= SENSORS[best_pick[2]]['cost']
        history.append(void_probability(intensity, placed))
        x, y, t = best_pick
        print(f"  Greedy → placed Type {t} at ({x:.1f}, {y:.1f}) | "
              f"Budget left: {remaining:.1f} | VP: {history[-1]:.4f}")

    return placed, history


# ──────────────────────────────────────────────────────────────
# METHOD 2 — UNIFORM PLACEMENT (baseline)
# Just scatter cheap Type-B sensors on a regular grid
# ──────────────────────────────────────────────────────────────
def uniform_placement(intensity, budget):
    n = int(budget // SENSORS['B']['cost'])   # how many we can afford
    side = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(-8, 8, side)
    ys = np.linspace(-8, 8, side)

    placed  = []
    history = []
    for x in xs:
        for y in ys:
            if len(placed) >= n:
                break
            placed.append((x, y, 'B'))
            history.append(void_probability(intensity, placed))
        if len(placed) >= n:
            break

    print(f"  Uniform → placed {len(placed)} Type-B sensors | "
          f"Final VP: {history[-1]:.4f}")
    return placed, history


# ──────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────
def plot_intensity(intensity, ax):
    im = ax.pcolormesh(np.rad2deg(ALPHA), P, intensity.T,
                       cmap='hot_r', shading='auto')
    ax.set_xlabel('Heading α (degrees)', fontsize=11)
    ax.set_ylabel('Distance p (km)', fontsize=11)
    ax.set_title('Target Intensity λ(α, p)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Intensity (targets/km²)')


def plot_sensor_map(greedy, uniform, ax):
    ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f4f8')
    ax.grid(True, alpha=0.3)
    ax.add_patch(mpatches.FancyBboxPatch((-10, -10), 20, 20,
                 boxstyle="round,pad=0.1", lw=2,
                 edgecolor='black', facecolor='none'))

    # Uniform sensors (background)
    for (x, y, _) in uniform:
        ax.plot(x, y, '^', color='cornflowerblue', markersize=5, alpha=0.6)

    # Greedy sensors with detection circles
    for i, (x, y, t) in enumerate(greedy):
        color = SENSORS[t]['color']
        ax.plot(x, y, 'o', color=color, markersize=9,
                markeredgecolor='black', markeredgewidth=0.8, zorder=5)
        ax.add_patch(plt.Circle((x, y), SENSORS[t]['sigma'],
                                color=color, alpha=0.15))
        ax.annotate(str(i + 1), (x, y), fontsize=7,
                    ha='center', va='center', color='white', fontweight='bold')

    ax.legend(handles=[
        mpatches.Patch(color='red',           label='Greedy: Type A (σ=2.5, cost=3)'),
        mpatches.Patch(color='blue',          label='Greedy: Type B (σ=1.0, cost=1)'),
        mpatches.Patch(color='cornflowerblue',label='Uniform: Type B'),
    ], loc='upper right', fontsize=8)
    ax.set_xlabel('Easting (km)', fontsize=11)
    ax.set_ylabel('Northing (km)', fontsize=11)
    ax.set_title('Sensor Positions in Physical Space', fontsize=12)


def plot_void_comparison(hist_greedy, hist_uniform, ax):
    ax.plot(range(1, len(hist_greedy)  + 1), hist_greedy,
            'o-', color='darkred',  lw=2, ms=6, label='Budget-Aware Greedy')
    ax.plot(range(1, len(hist_uniform) + 1), hist_uniform,
            's--', color='steelblue', lw=2, ms=5, label='Uniform Placement')
    ax.set_xlabel('Sensors Placed', fontsize=11)
    ax.set_ylabel('Void Probability ν', fontsize=11)
    ax.set_title('Void Probability vs. Number of Sensors', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4); ax.set_ylim(0, 1)


def plot_residual(intensity, placed, ax, title):
    """Show what intensity is left AFTER sensors filter it out."""
    AA, PP = np.meshgrid(ALPHA, P, indexing='ij')
    prob_miss = np.ones_like(intensity)
    for (sx, sy, t) in placed:
        s = SENSORS[t]
        prob_miss *= (1.0 - detection_prob(AA, PP, sx, sy, s['rho'], s['sigma']))
    residual = intensity * prob_miss
    im = ax.pcolormesh(np.rad2deg(ALPHA), P, residual.T,
                       cmap='hot_r', shading='auto',
                       vmin=0, vmax=intensity.max())
    ax.set_xlabel('Heading α (degrees)', fontsize=11)
    ax.set_ylabel('Distance p (km)', fontsize=11)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, label='Residual Intensity')


# ──────────────────────────────────────────────────────────────
# MAIN — Run everything and save figures
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Heterogeneous Sensor Placement Simulation")
    print("=" * 55)

    intensity = build_intensity()
    da, dp = ALPHA[1] - ALPHA[0], P[1] - P[0]
    print(f"\nTotal simulated targets/period: "
          f"{np.sum(intensity) * da * dp:.2f}")

    # Run both methods
    print("\n[Method 1] Budget-Aware Greedy:")
    greedy,  hist_g = greedy_placement(intensity, TOTAL_BUDGET)

    print("\n[Method 2] Uniform Placement:")
    uniform, hist_u = uniform_placement(intensity, TOTAL_BUDGET)

    vp_g = hist_g[-1] if hist_g else 0
    vp_u = hist_u[-1] if hist_u else 0
    print("\n" + "=" * 55)
    print(f"Greedy  final void probability : {vp_g:.4f}")
    print(f"Uniform final void probability : {vp_u:.4f}")
    print(f"Greedy improvement             : {(vp_g - vp_u) / vp_u * 100:.1f}%")
    print("=" * 55)

    # ── Figure 1: Main results ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        'Heterogeneous Sensor Placement for Barrier Coverage\n'
        f'Budget={TOTAL_BUDGET} | Type A: cost=3, σ=2.5 km | Type B: cost=1, σ=1.0 km',
        fontsize=13, fontweight='bold')

    plot_intensity(intensity,              axes[0, 0])
    plot_sensor_map(greedy, uniform,       axes[0, 1])
    plot_void_comparison(hist_g, hist_u,   axes[1, 0])
    plot_residual(intensity, greedy,       axes[1, 1],
                  f'Residual Intensity — Greedy (VP={vp_g:.3f})')

    plt.tight_layout()
    plt.savefig('/home/claude/figure1_main_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nFigure 1 saved.")

    # ── Figure 2: Before vs after thinning ─────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle('Target Intensity Before and After Sensor Coverage', fontsize=13, fontweight='bold')
    plot_residual(intensity, [],      axes2[0], 'Original (No Sensors)')
    plot_residual(intensity, greedy,  axes2[1], f'After Greedy  (VP={vp_g:.3f})')
    plot_residual(intensity, uniform, axes2[2], f'After Uniform (VP={vp_u:.3f})')
    plt.tight_layout()
    plt.savefig('/home/claude/figure2_thinning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved.")

    # ── Figure 3: Budget sensitivity ───────────────────────
    print("\nRunning budget sensitivity analysis...")
    budgets = [4, 6, 8, 10, 12]
    vp_g_list, vp_u_list = [], []

    for b in budgets:
        print(f"  Budget = {b}")
        _, hg = greedy_placement(intensity, b)
        _, hu = uniform_placement(intensity, b)
        vp_g_list.append(hg[-1] if hg else 0)
        vp_u_list.append(hu[-1] if hu else 0)

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.plot(budgets, vp_g_list, 'o-',  color='darkred',   lw=2.5, ms=8, label='Greedy')
    ax3.plot(budgets, vp_u_list, 's--', color='steelblue', lw=2.5, ms=8, label='Uniform')
    ax3.fill_between(budgets, vp_u_list, vp_g_list,
                     alpha=0.15, color='green', label='Greedy advantage')
    ax3.set_xlabel('Total Budget (units)', fontsize=12)
    ax3.set_ylabel('Void Probability ν', fontsize=12)
    ax3.set_title('Void Probability vs. Budget', fontsize=13)
    ax3.legend(fontsize=10); ax3.grid(True, alpha=0.4); ax3.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('/home/claude/figure3_budget_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved.")


if __name__ == "__main__":
    main()
