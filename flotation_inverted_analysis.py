#!/usr/bin/env python3
"""
Detailed analysis of the inverted (180°) recovery problem.
Generates an annotated GZ diagram showing WHY recovery is chaotic.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc
import matplotlib.patheffects as pe

OUT = "/Users/josemarianolopezurdiales/Documents/CAD"

# Load GZ data by re-running the hydrostatics core for cases 1 and 2
# (We'll use the saved data pattern from the hydrostatics script)
# For speed, re-compute just the two key cases inline.

import sys
sys.path.insert(0, "/Users/josemarianolopezurdiales/Documents/CAD")

print("Re-computing GZ curves for cases 1 & 2...")
# Import from hydrostatics
from flotation_hydrostatics import (
    build_mass_items, cg_and_inertia, make_body_points,
    find_equilibrium, rot_x, HEEL_ANGLES, RHO_WATER, Z_CENTER
)

# Case 1: no airbag, symmetric
items1 = build_mass_items(symmetric_crew=True, design_mass=False)
M1, cg1, I1 = cg_and_inertia(items1)
pts1, wts1 = make_body_points(include_airbag=False)

gz1 = []
for deg in HEEL_ANGLES:
    R = rot_x(np.radians(deg))
    h, V, cob = find_equilibrium(pts1, wts1, cg1, R, M1)
    gz1.append(cob[1])
gz1 = np.array(gz1)

# Case 1 but with FIXED mass asymmetry (symmetric subsystems)
# Move thermal and safety to distributed (theta=0, r=0)
items_sym = build_mass_items(symmetric_crew=True, design_mass=False)
# Patch: set thermal and safety to axis (r=0)
for it in items_sym:
    if it.name in ["Thermal", "Safety"]:
        it.x = 0.0
        it.y = 0.0
M_s, cg_s, I_s = cg_and_inertia(items_sym)

gz_sym = []
for deg in HEEL_ANGLES:
    R = rot_x(np.radians(deg))
    h, V, cob = find_equilibrium(pts1, wts1, cg_s, R, M_s)
    gz_sym.append(cob[1])
gz_sym = np.array(gz_sym)

print(f"  Case 1 (actual):    GZ(0)={gz1[0]:.4f}  GZ(90)={gz1[45]:.4f}  GZ(180)={gz1[90]:.4f}")
print(f"  Case symmetric:     GZ(0)={gz_sym[0]:.4f}  GZ(90)={gz_sym[45]:.4f}  GZ(180)={gz_sym[90]:.4f}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Annotated GZ comparison
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 1, figsize=(14, 11), gridspec_kw={'height_ratios': [3, 2]})

# ── Top panel: GZ curves with annotations ──
ax = axes[0]
ax.set_facecolor('#0f1117')
fig.patch.set_facecolor('#0f1117')

# Shade regions
angles = HEEL_ANGLES
ax.fill_between(angles, 0, gz1*100, where=gz1>0,
                alpha=0.15, color='#2ecc71', label='_')
ax.fill_between(angles, 0, gz1*100, where=gz1<0,
                alpha=0.15, color='#e74c3c', label='_')

# GZ curves
ax.plot(angles, gz1*100, color='#ff7f0e', lw=2.5,
        label='Actual mass distribution (245 kg at θ=288°)')
ax.plot(angles, gz_sym*100, color='#4f8cff', lw=2, ls='--',
        label='Corrected: thermal+safety moved to axis')

ax.axhline(0, color='#555', lw=0.8)

# Mark key points
# 180° starting point
ax.plot(180, gz1[90]*100, 'o', color='#e74c3c', ms=12, zorder=10)
ax.annotate('START\n180° inverted\nGZ ≈ 0 (unstable)',
            xy=(180, gz1[90]*100), xytext=(195, -45),
            fontsize=9, color='#e74c3c', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1d27', edgecolor='#e74c3c'))

# 0° target
ax.plot(0, gz1[0]*100, '*', color='#2ecc71', ms=15, zorder=10)
ax.annotate('TARGET\n0° upright\nGZ = +27 cm',
            xy=(0, gz1[0]*100), xytext=(25, 45),
            fontsize=9, color='#2ecc71', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1d27', edgecolor='#2ecc71'))

# Mark DANGER ZONES (negative GZ regions in 0-180° path)
neg_mask = gz1[:91] < 0  # 0-180°
neg_angles = angles[:91][neg_mask]
if len(neg_angles) > 0:
    for i, a in enumerate(neg_angles):
        if i == 0 or neg_angles[i] - neg_angles[i-1] > 4:
            start = a
        if i == len(neg_angles)-1 or neg_angles[i+1] - neg_angles[i] > 4:
            end = a
            mid = (start + end) / 2
            ax.axvspan(start-1, end+1, alpha=0.08, color='red', zorder=0)
            if end - start > 10:
                ax.text(mid, -55, 'WRONG-WAY\nZONE', ha='center', fontsize=8,
                       color='#e74c3c', fontweight='bold', alpha=0.7)

# Arrow showing recovery path
path_y = -62
ax.annotate('', xy=(5, path_y), xytext=(175, path_y),
            arrowprops=dict(arrowstyle='->', color='#f1c40f', lw=2.5))
ax.text(90, path_y-5, 'RECOVERY PATH (must traverse entire GZ curve)',
        ha='center', fontsize=9, color='#f1c40f', fontweight='bold')

ax.set_xlim(0, 360)
ax.set_ylabel('GZ (righting arm)  [cm]', color='#e8eaf0', fontsize=11)
ax.set_title('Why Inverted Recovery is Chaotic: The GZ Curve Problem',
            color='#e8eaf0', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=9, facecolor='#1a1d27',
         edgecolor='#2d3244', labelcolor='#e8eaf0')
ax.grid(True, alpha=0.15, color='#2d3244')
ax.tick_params(colors='#9aa0b0')
ax.set_xlabel('Heel angle [deg]', color='#e8eaf0')

# Add physics annotations
ax.text(0.02, 0.95,
    'Green zone: GZ > 0 → buoyancy pushes TOWARD upright\n'
    'Red zone: GZ < 0 → buoyancy pushes AWAY from upright',
    transform=ax.transAxes, fontsize=8.5, color='#9aa0b0',
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='#1a1d27', edgecolor='#2d3244'))

# ── Bottom panel: Physics explanation timeline ──
ax2 = axes[1]
ax2.set_facecolor('#0f1117')
ax2.set_xlim(0, 25)
ax2.set_ylim(-0.5, 4.5)
ax2.set_axis_off()

phases = [
    (0, 5, '#e74c3c', 'Phase 1: Slow drift',
     'GZ ≈ 0 near 180°\nTorque = M·g·GZ ≈ 140 N·m\nα = T/I = 0.19 rad/s²\n→ Takes ~4s to reach 10°/s'),
    (5, 10, '#f39c12', 'Phase 2: Acceleration',
     'GZ grows to ±20 cm\nTorque reaches ~2700 N·m\nCapsule accelerates\nthrough 150°→90°'),
    (10, 17, '#e74c3c', 'Phase 3: Wrong-way zones',
     'GZ flips sign!\nBuoyancy pushes BACKWARD\nMomentum carries through\nbut speed is unpredictable'),
    (17, 22, '#f1c40f', 'Phase 4: Overshoot',
     'Arrives near 0° with\nexcess angular velocity\nOvershoots to negative angles\nOscillates chaotically'),
    (22, 25, '#2ecc71', 'Phase 5: Eventual settling',
     'Drag dissipates energy\nCapsule oscillates ±30°\naround upright\nSettles in 30-60+ seconds'),
]

for x0, x1, col, title, desc in phases:
    ax2.add_patch(plt.Rectangle((x0, 0.5), x1-x0, 3.5,
                                facecolor=col, alpha=0.12, edgecolor=col, lw=1.5))
    ax2.text((x0+x1)/2, 4.2, title, ha='center', fontsize=9.5,
            color=col, fontweight='bold')
    ax2.text((x0+x1)/2, 2.2, desc, ha='center', fontsize=8,
            color='#c0c4d0', family='monospace', linespacing=1.4)

# Timeline arrow
ax2.annotate('', xy=(24.8, 0.1), xytext=(0.2, 0.1),
            arrowprops=dict(arrowstyle='->', color='#9aa0b0', lw=1.5))
ax2.text(12.5, -0.3, 'TIME (seconds)', ha='center', fontsize=9, color='#9aa0b0')

ax2.set_title('Recovery Timeline from 180° (estimated from Chrono simulation)',
             color='#9aa0b0', fontsize=10, pad=10)

plt.tight_layout()
plt.savefig(f"{OUT}/flotation_inverted_explained.png", dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
print(f"Saved flotation_inverted_explained.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: The root cause — mass asymmetry
# ═══════════════════════════════════════════════════════════════

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
fig2.patch.set_facecolor('#0f1117')

for ax, gz, title, col in [
    (ax3, gz1, 'CURRENT: 245 kg at θ=288°\n(thermal + safety on back wall)', '#ff7f0e'),
    (ax4, gz_sym, 'CORRECTED: mass redistributed\n(thermal + safety moved to axis)', '#4f8cff'),
]:
    ax.set_facecolor('#0f1117')
    mask90 = angles <= 180

    ax.fill_between(angles[mask90], 0, gz[mask90]*100,
                    where=gz[mask90]>0, alpha=0.2, color='#2ecc71')
    ax.fill_between(angles[mask90], 0, gz[mask90]*100,
                    where=gz[mask90]<0, alpha=0.2, color='#e74c3c')
    ax.plot(angles[mask90], gz[mask90]*100, color=col, lw=2.5)
    ax.axhline(0, color='#555', lw=0.8)

    # Count zero crossings in 0-180
    sign_changes = np.where(np.diff(np.sign(gz[mask90])))[0]
    n_cross = len(sign_changes)

    # Fraction of path with GZ > 0
    frac_pos = np.sum(gz[mask90] > 0) / np.sum(mask90) * 100

    ax.set_title(title, color='#e8eaf0', fontsize=11, fontweight='bold')
    ax.set_xlabel('Heel angle [deg]', color='#9aa0b0')
    ax.set_ylabel('GZ [cm]', color='#9aa0b0')
    ax.grid(True, alpha=0.15, color='#2d3244')
    ax.tick_params(colors='#9aa0b0')
    ax.set_xlim(0, 180)

    # Stats box
    verdict = "CLEAN RECOVERY" if frac_pos > 90 else "CHAOTIC RECOVERY"
    vcol = '#2ecc71' if frac_pos > 90 else '#e74c3c'
    ax.text(0.97, 0.95,
        f'Zero crossings: {n_cross}\n'
        f'GZ > 0: {frac_pos:.0f}% of path\n'
        f'GZ(0°) = {gz[0]*100:+.1f} cm\n'
        f'GZ(180°) = {gz[90]*100:+.1f} cm\n'
        f'→ {verdict}',
        transform=ax.transAxes, fontsize=9, color=vcol,
        verticalalignment='top', horizontalalignment='right',
        fontweight='bold', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#1a1d27', edgecolor=vcol))

fig2.suptitle('ROOT CAUSE: Mass asymmetry creates wrong-way GZ zones in the recovery path',
             color='#f1c40f', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/flotation_rootcause.png", dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
print(f"Saved flotation_rootcause.png")

print("\nDone.")
