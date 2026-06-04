import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)
AREA      = 20          # 20 x 20 grid
BUDGET    = 10
N_TARGETS = 100

# Sensor types
SENSOR_A = {'range': 4, 'cost': 3, 'color': 'red',   'label': 'Type A (range=4, cost=3)'}
SENSOR_B = {'range': 2, 'cost': 1, 'color': 'blue',  'label': 'Type B (range=2, cost=1)'}

# ── Generate targets: two shipping lanes ───────────────────────
# Lane 1: diagonal from bottom-left to center  (denser)
lane1 = np.column_stack([
    np.random.uniform(2, 10, 60),
    np.random.uniform(2, 10, 60) + np.random.normal(0, 1, 60)
])
# Lane 2: diagonal from center to top-right  (sparser)
lane2 = np.column_stack([
    np.random.uniform(10, 18, 40),
    np.random.uniform(10, 18, 40) + np.random.normal(0, 1, 40)
])
targets = np.vstack([lane1, lane2])
targets = np.clip(targets, 0, AREA)   # stay inside area

# ── Detection function ─────────────────────────────────────────
def detect(sensor_pos, sensor_range, all_targets):
    """A target is detected if it falls within the sensor's range."""
    distances = np.linalg.norm(all_targets - sensor_pos, axis=1)
    return distances <= sensor_range

def detection_rate(sensors, all_targets):
    """Fraction of targets detected by at least one sensor."""
    detected = np.zeros(len(all_targets), dtype=bool)
    for pos, stype in sensors:
        detected |= detect(pos, stype['range'], all_targets)
    return detected.sum() / len(all_targets)

# ── Method 1: Uniform Placement ────────────────────────────────
def uniform_placement(budget):
    """
    Place Type-B sensors (cheapest) at fixed corner + center positions.
    Simple baseline — no awareness of where targets actually are.
    """
    fixed_spots = [
        np.array([ 5,  5]),
        np.array([ 5, 15]),
        np.array([15,  5]),
        np.array([15, 15]),
        np.array([10, 10]),
        np.array([ 3, 10]),
        np.array([17, 10]),
        np.array([10,  3]),
        np.array([10, 17]),
        np.array([ 3,  3]),
    ]
    sensors = []
    remaining = budget
    for spot in fixed_spots:
        if remaining < SENSOR_B['cost']:
            break
        sensors.append((spot, SENSOR_B))
        remaining -= SENSOR_B['cost']
    return sensors

# ── Method 2: Density-Aware Greedy ────────────────────────────
def greedy_placement(budget, all_targets):
    """
    At each step: try every candidate position and sensor type.
    Place whichever detects the most NEW targets per cost unit.
    Focus naturally falls on dense target areas.
    """
    # Candidate positions: 5x5 grid over the area
    grid = np.linspace(2, 18, 8)
    candidates = [(np.array([x, y]), stype)
                  for x in grid for y in grid
                  for stype in [SENSOR_A, SENSOR_B]]

    sensors   = []
    remaining = budget
    detected_so_far = np.zeros(len(all_targets), dtype=bool)

    while remaining > 0:
        best_gain     = -1
        best_choice   = None

        for pos, stype in candidates:
            if stype['cost'] > remaining:
                continue
            new_detections = detect(pos, stype['range'], all_targets)
            # Only count targets not already detected
            gain = (new_detections & ~detected_so_far).sum() / stype['cost']
            if gain > best_gain:
                best_gain   = gain
                best_choice = (pos, stype)

        if best_choice is None or best_gain == 0:
            break

        pos, stype = best_choice
        sensors.append((pos, stype))
        detected_so_far |= detect(pos, stype['range'], all_targets)
        remaining -= stype['cost']
        print(f"  Placed {stype['label']} at ({pos[0]:.0f},{pos[1]:.0f}) | "
              f"Budget left: {remaining:.0f} | "
              f"Detected: {detected_so_far.sum()}/{len(all_targets)}")

    return sensors

# ── Run both methods ───────────────────────────────────────────
print("=" * 50)
print("  Toy Sensor Placement Simulation")
print("=" * 50)

print("\n[Method 1] Uniform Placement:")
uniform = uniform_placement(BUDGET)
rate_u  = detection_rate(uniform, targets)
print(f"  Sensors placed: {len(uniform)} | Detection rate: {rate_u*100:.1f}%")

print("\n[Method 2] Density-Aware Greedy:")
greedy = greedy_placement(BUDGET, targets)
rate_g = detection_rate(greedy, targets)
print(f"  Sensors placed: {len(greedy)} | Detection rate: {rate_g*100:.1f}%")

print(f"\nImprovement: +{(rate_g - rate_u)*100:.1f} percentage points")

# ── Plot ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Heterogeneous Sensor Placement — Toy Simulation\n'
             f'Budget = {BUDGET} | Type A: range=4, cost=3 | Type B: range=2, cost=1',
             fontsize=13, fontweight='bold')

def draw_map(ax, sensors, title, rate):
    ax.set_xlim(0, AREA); ax.set_ylim(0, AREA)
    ax.set_facecolor('#f5f7fa')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Mark which targets are detected
    detected = np.zeros(len(targets), dtype=bool)
    for pos, stype in sensors:
        detected |= detect(pos, stype['range'], targets)

    ax.scatter(targets[~detected, 0], targets[~detected, 1],
               c='gray', s=15, alpha=0.5, label='Missed', zorder=2)
    ax.scatter(targets[detected,  0], targets[detected,  1],
               c='limegreen', s=15, alpha=0.8, label='Detected', zorder=2)

    # Draw sensors + coverage circles
    for pos, stype in sensors:
        circle = plt.Circle(pos, stype['range'],
                            color=stype['color'], alpha=0.15)
        ax.add_patch(circle)
        ax.plot(*pos, 'o', color=stype['color'], markersize=9,
                markeredgecolor='black', markeredgewidth=0.8, zorder=5)

    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{title}\nDetection Rate: {rate*100:.1f}%', fontsize=11)

# Panel 1: target distribution
axes[0].set_facecolor('#f5f7fa')
axes[0].scatter(lane1[:, 0], lane1[:, 1], c='steelblue', s=15,
                alpha=0.6, label='Lane 1 (dense)')
axes[0].scatter(lane2[:, 0], lane2[:, 1], c='orange',    s=15,
                alpha=0.6, label='Lane 2 (sparse)')
axes[0].set_xlim(0, AREA); axes[0].set_ylim(0, AREA)
axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=9)
axes[0].set_title('Target Distribution\n(Two Shipping Lanes)', fontsize=11)
axes[0].set_xlabel('X (km)'); axes[0].set_ylabel('Y (km)')

draw_map(axes[1], uniform, 'Uniform Placement (Baseline)', rate_u)
draw_map(axes[2], greedy,  'Density-Aware Greedy (Proposed)', rate_g)

# Legend for sensor types
patch_A = mpatches.Patch(color='red',  label='Type A (range=4, cost=3)')
patch_B = mpatches.Patch(color='blue', label='Type B (range=2, cost=1)')
fig.legend(handles=[patch_A, patch_B], loc='lower center',
           ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig('toy_simulation.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("\nFigure saved: toy_simulation.png")