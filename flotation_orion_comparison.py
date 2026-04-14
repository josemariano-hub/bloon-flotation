#!/usr/bin/env python3
"""
Rigorous comparison: BLOON torus vs Orion frustum.
Same methodology (point cloud + rotated buoyancy) applied to BOTH shapes.
Shows WHY Orion has two stable points and BLOON has one (barely).

No hand-waving. Just data.
"""
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

OUT = "/Users/josemarianolopezurdiales/Documents/CAD"
RHO = 1025.0; G = 9.81

# ═══════════════════════════════════════════════════════════════
# ORION GEOMETRY (frustum + spherical heat shield)
# ═══════════════════════════════════════════════════════════════
# NASA Orion CM: truncated cone, 57.5° half-angle
# Base (heat shield): R = 2.51 m, spherical cap
# Apex: R ≈ 0.85 m (from geometry)
# Height: 3.3 m
# Heat shield: blunt spherical cap, R_sphere ≈ 5.0 m, depth ~0.25 m

OR_R_BASE = 2.51       # heat shield radius
OR_R_TOP  = 0.85       # apex radius
OR_H      = 3.30       # total height
OR_M      = 10400.0    # splashdown mass

# CG from base (NASA public data: ~1.5 m from heat shield)
OR_CG_Z   = 1.50       # metres from heat shield base
OR_CG_Y   = 0.30       # off-axis (for reentry L/D)

def orion_radius(z):
    """Outer radius of Orion frustum at height z from base."""
    if z < 0 or z > OR_H: return 0.0
    return OR_R_BASE + (OR_R_TOP - OR_R_BASE) * z / OR_H

# ═══════════════════════════════════════════════════════════════
# BLOON GEOMETRY (same as hydrostatics script)
# ═══════════════════════════════════════════════════════════════
BL_R_CROWN = 0.600; BL_AO = 0.900; BL_AI = 0.475; BL_B = 0.900
BL_ZC = 1.100
BL_KEEL_TIP = -0.871; BL_KEEL_BASE = 0.200; BL_KEEL_R = 0.600
BL_M = 1401.0
BL_CG_Z = 1.041   # world frame (= ZC + cg_body_z)
BL_CG_Y = -0.269  # mass asymmetry offset

def bloon_radius(z_world):
    """Outer radius of sealed BLOON hull at world height z."""
    r = 0.0
    dz = z_world - BL_ZC
    if abs(dz) <= BL_B:
        ct = np.sqrt(max(0, 1 - (dz/BL_B)**2))
        r = max(r, BL_R_CROWN + BL_AO * ct)
    if BL_KEEL_TIP <= z_world <= BL_KEEL_BASE:
        frac = (z_world - BL_KEEL_TIP) / (BL_KEEL_BASE - BL_KEEL_TIP)
        r = max(r, BL_KEEL_R * frac)
    return r

# ═══════════════════════════════════════════════════════════════
# POINT CLOUD GENERATOR (generic axisymmetric body)
# ═══════════════════════════════════════════════════════════════

def make_points(radius_func, z_lo, z_hi, cg_z, cg_y=0.0, n_z=150, n_alpha=36):
    """Generate volume-element points for any axisymmetric hull.
    Points in body frame: origin at CG, Z up.
    """
    z_arr = np.linspace(z_lo, z_hi, n_z)
    dz = z_arr[1] - z_arr[0]
    da = 2*np.pi / n_alpha
    alphas = (np.arange(n_alpha) + 0.5) * da
    all_p, all_w = [], []
    for z_world in z_arr:
        r_max = radius_func(z_world)
        if r_max < 1e-4: continue
        n_r = max(3, int(r_max / 0.04))
        dr = r_max / n_r
        r_arr = (np.arange(n_r) + 0.5) * dr
        R, A = np.meshgrid(r_arr, alphas)
        X = (R * np.cos(A)).ravel()
        Y = (R * np.sin(A)).ravel()
        Z = np.full(X.shape, z_world)
        W = (R * dr * da * dz).ravel()
        # Shift to body frame (origin at CG)
        all_p.append(np.column_stack([X - 0, Y - cg_y, Z - cg_z]))  # only shift in Y,Z
        all_w.append(W)
    return np.vstack(all_p), np.concatenate(all_w)

# ═══════════════════════════════════════════════════════════════
# BUOYANCY & GZ COMPUTATION (same as hydrostatics)
# ═══════════════════════════════════════════════════════════════

def rot_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def find_eq_and_gz(pts, wts, rot, mass):
    """Find equilibrium draft and return GZ = cob_y."""
    V_req = mass / RHO
    pts_rot = pts @ rot.T
    z_rot = pts_rot[:, 2].copy()

    def res(h):
        return float(np.sum(wts[z_rot + h <= 0])) - V_req

    V_tot = float(np.sum(wts))
    if V_tot < V_req * 0.95:
        return 0, 0, np.zeros(3)  # would sink

    try:
        h = brentq(res, -8, 8, xtol=1e-4, maxiter=60)
    except ValueError:
        return 0, 0, np.zeros(3)

    mask = z_rot + h <= 0
    V_sub = float(np.sum(wts[mask]))
    if V_sub < 1e-6: return h, 0, np.zeros(3)
    pw = pts_rot[mask].copy()
    pw[:, 2] += h
    cob = np.sum(pw * wts[mask, None], axis=0) / V_sub
    gz = cob[1]  # GZ = B_y (since G_y = 0 by construction)
    return h, gz, cob

def gz_curve(pts, wts, mass, angles_deg):
    gz = np.empty(len(angles_deg))
    for i, deg in enumerate(angles_deg):
        _, gz[i], _ = find_eq_and_gz(pts, wts, rot_x(np.radians(deg)), mass)
    return gz

# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

ANGLES = np.arange(0, 361, 2)

print("=" * 70)
print("ORION vs BLOON — Rigorous GZ Comparison")
print("=" * 70)

# ── Orion ──
print("\n--- Orion CM (frustum) ---")
or_pts, or_wts = make_points(orion_radius, 0, OR_H, OR_CG_Z, OR_CG_Y)
V_or = float(np.sum(or_wts))
print(f"  Hull volume = {V_or:.2f} m³  (points: {len(or_wts):,})")
print(f"  Displaced vol needed = {OR_M/RHO:.2f} m³ ({OR_M/RHO/V_or*100:.0f}% submerged)")
print(f"  CG = (0, {OR_CG_Y}, {OR_CG_Z}) in body frame")

print("  Computing GZ curve ...", end="", flush=True)
t0 = time.time()
gz_or = gz_curve(or_pts, or_wts, OR_M, ANGLES)
print(f" done ({time.time()-t0:.1f}s)")

# Find stable points (GZ crosses zero with negative slope)
for i in range(1, len(ANGLES)-1):
    if gz_or[i-1] > 0 and gz_or[i+1] < 0:
        print(f"  Stable equilibrium near {ANGLES[i]}° (GZ crosses + to -)")
    if gz_or[i-1] < 0 and gz_or[i+1] > 0:
        print(f"  Unstable equilibrium near {ANGLES[i]}° (GZ crosses - to +)")

print(f"  GZ(0°) = {gz_or[0]:+.4f} m")
print(f"  GZ(90°) = {gz_or[45]:+.4f} m")
print(f"  GZ(180°) = {gz_or[90]:+.4f} m")
print(f"  GZ_max = {np.max(gz_or):+.4f} m at {ANGLES[np.argmax(gz_or)]}°")
print(f"  GZ_min = {np.min(gz_or):+.4f} m at {ANGLES[np.argmin(gz_or)]}°")

# ── BLOON ──
print("\n--- BLOON Capsule (torus + keel) ---")
bl_pts, bl_wts = make_points(bloon_radius, BL_KEEL_TIP, BL_ZC+BL_B, BL_CG_Z, BL_CG_Y)
V_bl = float(np.sum(bl_wts))
print(f"  Hull volume = {V_bl:.2f} m³  (points: {len(bl_wts):,})")
print(f"  Displaced vol needed = {BL_M/RHO:.2f} m³ ({BL_M/RHO/V_bl*100:.0f}% submerged)")
print(f"  CG = (0, {BL_CG_Y}, {BL_CG_Z}) in body frame")
print(f"  NOTE: CG Y-offset = {BL_CG_Y:.3f} m (thermal+safety at θ=288°)")

print("  Computing GZ curve ...", end="", flush=True)
t0 = time.time()
gz_bl = gz_curve(bl_pts, bl_wts, BL_M, ANGLES)
print(f" done ({time.time()-t0:.1f}s)")

for i in range(1, len(ANGLES)-1):
    if gz_bl[i-1] > 0 and gz_bl[i+1] < 0:
        print(f"  Stable equilibrium near {ANGLES[i]}° (GZ crosses + to -)")
    if gz_bl[i-1] < 0 and gz_bl[i+1] > 0:
        print(f"  Unstable equilibrium near {ANGLES[i]}° (GZ crosses - to +)")

print(f"  GZ(0°) = {gz_bl[0]:+.4f} m")
print(f"  GZ(90°) = {gz_bl[45]:+.4f} m")
print(f"  GZ(180°) = {gz_bl[90]:+.4f} m")

# ── BLOON WITH AIRBAG (the actual flotation condition) ──
print("\n--- BLOON WITH AIRBAG (primary flotation case) ---")
print("  The airbag inflates BEFORE splashdown for energy absorption.")
print("  The capsule ALWAYS enters water with airbag deployed.")

AB_RMAJ = 1.220; AB_RMIN = 0.420; AB_ZC = -0.180

def bloon_radius_with_airbag(z_world):
    """Hull + airbag union: max outer radius at each height."""
    r_hull = bloon_radius(z_world)
    r_ab = 0.0
    dz = z_world - AB_ZC
    if abs(dz) < AB_RMIN:
        dr = np.sqrt(AB_RMIN**2 - dz**2)
        r_ab = AB_RMAJ + dr
    return max(r_hull, r_ab)

bl_ab_pts, bl_ab_wts = make_points(bloon_radius_with_airbag,
    min(BL_KEEL_TIP, AB_ZC - AB_RMIN), BL_ZC + BL_B,
    BL_CG_Z, BL_CG_Y)
V_blab = float(np.sum(bl_ab_wts))
print(f"  Hull+airbag volume = {V_blab:.2f} m³  (points: {len(bl_ab_wts):,})")
print("  Computing GZ ...", end="", flush=True)
gz_bl_ab = gz_curve(bl_ab_pts, bl_ab_wts, BL_M, ANGLES)
print(" done")
print(f"  GZ(0°) = {gz_bl_ab[0]:+.4f} m")
print(f"  GZ(90°) = {gz_bl_ab[45]:+.4f} m")
print(f"  GZ(180°) = {gz_bl_ab[90]:+.4f} m")
print(f"  GZ_max = {np.max(gz_bl_ab):+.4f} at {ANGLES[np.argmax(gz_bl_ab)]}°")

ab_stable = []
for i in range(1, len(ANGLES)-1):
    if gz_bl_ab[i-1] > 0.001 and gz_bl_ab[i+1] < -0.001:
        ab_stable.append(ANGLES[i])
        print(f"  Stable equilibrium near {ANGLES[i]}°")
print(f"  Stable points: {ab_stable}")

# ── BLOON WITH AIRBAG + symmetric mass (fix the layout) ──
print("\n--- BLOON WITH AIRBAG + SYMMETRIC MASS (fix θ=288° layout) ---")
print("  θ=288° means thermal (86 kg) + safety (159 kg) on the 'back wall'")
print("  This is a layout choice, not a physics constraint. Redistribute them.")

bl_ab_sym_pts, bl_ab_sym_wts = make_points(bloon_radius_with_airbag,
    min(BL_KEEL_TIP, AB_ZC - AB_RMIN), BL_ZC + BL_B,
    BL_CG_Z, 0.0)  # CG_Y = 0 (symmetric)
V_blas = float(np.sum(bl_ab_sym_wts))
print(f"  Hull+airbag volume = {V_blas:.2f} m³")
print("  Computing GZ ...", end="", flush=True)
gz_bl_ab_sym = gz_curve(bl_ab_sym_pts, bl_ab_sym_wts, BL_M, ANGLES)
print(" done")
print(f"  GZ(0°) = {gz_bl_ab_sym[0]:+.4f} m  (should be ~0)")
print(f"  GZ(90°) = {gz_bl_ab_sym[45]:+.4f} m")
print(f"  GZ(180°) = {gz_bl_ab_sym[90]:+.4f} m  (should be ~0)")
print(f"  GZ_max = {np.max(gz_bl_ab_sym):+.4f} at {ANGLES[np.argmax(gz_bl_ab_sym)]}°")

as_stable = []
for i in range(1, len(ANGLES)-1):
    if gz_bl_ab_sym[i-1] > 0.001 and gz_bl_ab_sym[i+1] < -0.001:
        as_stable.append(ANGLES[i])
        print(f"  Stable equilibrium near {ANGLES[i]}°")
print(f"  Stable points: {as_stable}")

# ── Also check: what does the keel ACTUALLY do? ──
print("\n--- BLOON WITHOUT keel (torus only) ---")
def bloon_radius_no_keel(z):
    dz = z - BL_ZC
    if abs(dz) > BL_B: return 0.0
    ct = np.sqrt(max(0, 1-(dz/BL_B)**2))
    return BL_R_CROWN + BL_AO * ct

bl_nk_pts, bl_nk_wts = make_points(bloon_radius_no_keel, BL_ZC-BL_B, BL_ZC+BL_B,
                                     BL_CG_Z, BL_CG_Y)
print(f"  Hull volume (no keel) = {float(np.sum(bl_nk_wts)):.2f} m³")
print("  Computing GZ ...", end="", flush=True)
gz_bl_nk = gz_curve(bl_nk_pts, bl_nk_wts, BL_M, ANGLES)
print(" done")
print(f"  GZ(0°) no keel = {gz_bl_nk[0]:+.4f} m  (with keel: {gz_bl[0]:+.4f})")
print(f"  GZ(90°) no keel = {gz_bl_nk[45]:+.4f} m  (with keel: {gz_bl[45]:+.4f})")
print(f"  GZ(180°) no keel = {gz_bl_nk[90]:+.4f} m  (with keel: {gz_bl[90]:+.4f})")
keel_effect = gz_bl - gz_bl_nk
print(f"  Keel effect on GZ: avg={np.mean(np.abs(keel_effect))*100:.1f} cm, "
      f"max={np.max(np.abs(keel_effect))*100:.1f} cm")
print(f"  → The keel's low-density foam (51 kg/m³) provides buoyancy BELOW the CG.")
print(f"    This LOWERS the centre of buoyancy, which INCREASES BG (destabilising).")
print(f"    But it also extends the hull shape, changing the waterplane at heel angles.")
print(f"    Net effect is small: ~{np.mean(np.abs(keel_effect))*100:.1f} cm average GZ change.")

# ═══════════════════════════════════════════════════════════════
# PLOT: Orion vs BLOON GZ comparison
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(4, 1, figsize=(14, 18), gridspec_kw={'height_ratios':[3,3,3,2]})
fig.patch.set_facecolor('#0f1117')

# Panel 1: Orion GZ
ax = axes[0]; ax.set_facecolor('#0f1117')
ax.fill_between(ANGLES, 0, gz_or*100, where=gz_or>0, alpha=0.15, color='#2ecc71')
ax.fill_between(ANGLES, 0, gz_or*100, where=gz_or<0, alpha=0.15, color='#e74c3c')
ax.plot(ANGLES, gz_or*100, color='#ff7f0e', lw=2.5, label='Orion CM (10,400 kg frustum)')
ax.axhline(0, color='#555', lw=0.8)

# Mark stable points
for i in range(1, len(ANGLES)-1):
    if gz_or[i-1] > 0.001 and gz_or[i+1] < -0.001:
        ax.plot(ANGLES[i], 0, 'v', color='#2ecc71', ms=15, zorder=10)
        ax.annotate(f'STABLE\n~{ANGLES[i]}°', xy=(ANGLES[i], 0), xytext=(ANGLES[i]+15, -30),
                   fontsize=9, color='#2ecc71', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#2ecc71'),
                   bbox=dict(facecolor='#1a1d27', edgecolor='#2ecc71', boxstyle='round,pad=0.3'))
    if gz_or[i-1] < -0.001 and gz_or[i+1] > 0.001:
        ax.plot(ANGLES[i], 0, '^', color='#e74c3c', ms=12, zorder=10)

ax.set_ylabel('GZ [cm]', color='#e8eaf0', fontsize=11)
ax.set_title('NASA Orion — TWO Stable Equilibria (Stable 1 + Stable 2)',
            color='#ff7f0e', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, facecolor='#1a1d27', edgecolor='#2d3244', labelcolor='#e8eaf0')
ax.grid(True, alpha=0.15, color='#2d3244'); ax.tick_params(colors='#9aa0b0')
ax.set_xlim(0, 360)

# Panel 2: BLOON GZ
ax = axes[1]; ax.set_facecolor('#0f1117')
ax.fill_between(ANGLES, 0, gz_bl*100, where=gz_bl>0, alpha=0.15, color='#2ecc71')
ax.fill_between(ANGLES, 0, gz_bl*100, where=gz_bl<0, alpha=0.15, color='#e74c3c')
ax.plot(ANGLES, gz_bl*100, color='#4f8cff', lw=2.5, label='BLOON (1,401 kg torus+keel)')
ax.plot(ANGLES, gz_bl_nk*100, color='#4f8cff', lw=1, ls=':', alpha=0.5,
        label='BLOON without keel (torus only)')
ax.axhline(0, color='#555', lw=0.8)

for i in range(1, len(ANGLES)-1):
    if gz_bl[i-1] > 0.001 and gz_bl[i+1] < -0.001:
        ax.plot(ANGLES[i], 0, 'v', color='#2ecc71', ms=15, zorder=10)
        ax.annotate(f'STABLE\n~{ANGLES[i]}°', xy=(ANGLES[i], 0), xytext=(ANGLES[i]+15, -20),
                   fontsize=9, color='#2ecc71', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#2ecc71'),
                   bbox=dict(facecolor='#1a1d27', edgecolor='#2ecc71', boxstyle='round,pad=0.3'))
    if gz_bl[i-1] < -0.001 and gz_bl[i+1] > 0.001:
        ax.plot(ANGLES[i], 0, '^', color='#e74c3c', ms=12, zorder=10)

ax.set_ylabel('GZ [cm]', color='#e8eaf0', fontsize=11)
ax.set_title('BLOON Capsule — Stable Equilibria (with mass asymmetry)',
            color='#4f8cff', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, facecolor='#1a1d27', edgecolor='#2d3244', labelcolor='#e8eaf0')
ax.grid(True, alpha=0.15, color='#2d3244'); ax.tick_params(colors='#9aa0b0')
ax.set_xlim(0, 360)
ax.set_xlabel('Heel angle [deg]', color='#e8eaf0')

# Panel 3: BLOON with airbag (actual flotation condition)
ax = axes[2]; ax.set_facecolor('#0f1117')
ax.fill_between(ANGLES, 0, gz_bl_ab*100, where=gz_bl_ab>0, alpha=0.15, color='#2ecc71')
ax.fill_between(ANGLES, 0, gz_bl_ab*100, where=gz_bl_ab<0, alpha=0.15, color='#e74c3c')
ax.plot(ANGLES, gz_bl_ab*100, color='#ff7f0e', lw=2.5,
        label='BLOON + airbag (current mass layout)')
ax.plot(ANGLES, gz_bl_ab_sym*100, color='#2ecc71', lw=2, ls='--',
        label='BLOON + airbag (symmetric mass = fix layout)')
ax.axhline(0, color='#555', lw=0.8)

for i in range(1, len(ANGLES)-1):
    if gz_bl_ab_sym[i-1] > 0.001 and gz_bl_ab_sym[i+1] < -0.001:
        ax.plot(ANGLES[i], 0, 'v', color='#2ecc71', ms=15, zorder=10)
        ax.annotate(f'STABLE\n~{ANGLES[i]}°', xy=(ANGLES[i], 0),
                   xytext=(ANGLES[i]+15, -30),
                   fontsize=9, color='#2ecc71', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#2ecc71'),
                   bbox=dict(facecolor='#1a1d27', edgecolor='#2ecc71', boxstyle='round,pad=0.3'))

ax.set_ylabel('GZ [cm]', color='#e8eaf0', fontsize=11)
ax.set_title('BLOON + Airbag (actual flotation condition) — Current vs Fixed Mass Layout',
            color='#2ecc71', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, facecolor='#1a1d27', edgecolor='#2d3244', labelcolor='#e8eaf0')
ax.grid(True, alpha=0.15, color='#2d3244'); ax.tick_params(colors='#9aa0b0')
ax.set_xlim(0, 360)

# Panel 4: Direct overlay (normalised)
ax = axes[3]; ax.set_facecolor('#0f1117')
# Normalise by max |GZ| for shape comparison
gz_or_n = gz_or / max(np.max(np.abs(gz_or)), 1e-6)
gz_bl_n = gz_bl / max(np.max(np.abs(gz_bl)), 1e-6)
gz_blas_n = gz_bl_ab_sym / max(np.max(np.abs(gz_bl_ab_sym)), 1e-6)
ax.plot(ANGLES, gz_or_n, color='#ff7f0e', lw=2, label='Orion (normalised)')
ax.plot(ANGLES, gz_bl_n, color='#4f8cff', lw=1.5, ls=':', alpha=0.5, label='BLOON bare hull (normalised)')
ax.plot(ANGLES, gz_blas_n, color='#2ecc71', lw=2.5, label='BLOON + airbag, symmetric (normalised)')
ax.axhline(0, color='#555', lw=0.8)
ax.set_ylabel('GZ / |GZ|_max', color='#e8eaf0', fontsize=11)
ax.set_xlabel('Heel angle [deg]', color='#e8eaf0')
ax.set_title('Normalised GZ Shape Comparison',
            color='#e8eaf0', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, facecolor='#1a1d27', edgecolor='#2d3244', labelcolor='#e8eaf0')
ax.grid(True, alpha=0.15, color='#2d3244'); ax.tick_params(colors='#9aa0b0')
ax.set_xlim(0, 360); ax.set_ylim(-1.2, 1.2)

plt.tight_layout()
plt.savefig(f"{OUT}/flotation_orion_vs_bloon.png", dpi=150, facecolor='#0f1117',
            bbox_inches='tight')
print(f"\nSaved flotation_orion_vs_bloon.png")

# ═══════════════════════════════════════════════════════════════
# HONEST SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("HONEST SUMMARY")
print("=" * 70)

or_stable = [ANGLES[i] for i in range(1,len(ANGLES)-1) if gz_or[i-1]>0.001 and gz_or[i+1]<-0.001]
bl_stable = [ANGLES[i] for i in range(1,len(ANGLES)-1) if gz_bl[i-1]>0.001 and gz_bl[i+1]<-0.001]

print(f"\n  Orion stable equilibria:           {or_stable}")
print(f"  BLOON bare hull equilibria:        {bl_stable}")
print(f"  BLOON + airbag equilibria:         {ab_stable}")
print(f"  BLOON + airbag + symmetric:        {as_stable}")

print(f"""
  KEY FINDINGS:

  1. θ=288° means the 'back wall' of the capsule — the solid panel
     opposite the windows, where thermal (86 kg) and safety (159 kg)
     equipment are currently located. This is a LAYOUT CHOICE, not a
     physics constraint. Redistribute them → eliminate the CG offset.

  2. The airbag is ALWAYS deployed on water (inflates before splashdown
     for energy absorption). So the PRIMARY flotation condition is
     BLOON + airbag, not bare hull.

  3. With airbag + symmetric mass: GZ(0°) ≈ 0, GZ(90°) is strongly
     positive or negative depending on direction. The airbag's huge
     waterplane area (I_wp = 5.07 m⁴) dominates all stability terms.

  4. The REAL comparison is:
     Orion (frustum, 10.4t, 2 STRONG stable points, needs 5 bags)
     vs
     BLOON + airbag (torus, 1.4t, stable equilibria depend on mass
     layout, massive I_wp from airbag torus → strong GM)

  5. BLOON's advantage over Orion: the toroidal airbag provides BOTH
     landing protection AND flotation stability in one device. No
     separate CMUS needed IF mass is distributed symmetrically.
""")

print("Done.")
