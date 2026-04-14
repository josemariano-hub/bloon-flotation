#!/usr/bin/env python3
"""
Orion CM flotation deep-dive: prove WHY it needs CMUS.
Run the same point-cloud hydrostatics on the Orion frustum shape,
show both stable equilibria, compute the energy barrier between them,
and size the bags needed to overcome it.
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
# ORION GEOMETRY
# ═══════════════════════════════════════════════════════════════
OR_R_BASE = 2.51;  OR_R_TOP = 0.85;  OR_H = 3.30;  OR_M = 10400.0
# TWO CG cases: symmetric (CG on axis) and with offset
OR_CG_Z_SYM = 1.50  # from heat-shield base

def orion_r(z):
    if z < 0 or z > OR_H: return 0.0
    return OR_R_BASE + (OR_R_TOP - OR_R_BASE) * z / OR_H

# ═══════════════════════════════════════════════════════════════
# POINT CLOUD + GZ (reusable)
# ═══════════════════════════════════════════════════════════════
def make_pts(rfunc, z_lo, z_hi, cg_z, cg_y=0.0, nz=150, na=36):
    z_arr = np.linspace(z_lo, z_hi, nz); dz = z_arr[1]-z_arr[0]
    da = 2*np.pi/na; alphas = (np.arange(na)+0.5)*da
    pp, ww = [], []
    for zw in z_arr:
        rm = rfunc(zw)
        if rm < 1e-4: continue
        nr = max(3, int(rm/0.04)); dr = rm/nr
        r_arr = (np.arange(nr)+0.5)*dr
        R, A = np.meshgrid(r_arr, alphas)
        pp.append(np.column_stack([(R*np.cos(A)).ravel(), (R*np.sin(A)).ravel()-cg_y,
                                    np.full(R.size, zw)-cg_z]))
        ww.append((R*dr*da*dz).ravel())
    return np.vstack(pp), np.concatenate(ww)

def rot_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def gz_at(pts, wts, mass, phi_rad):
    V_req = mass / RHO
    pr = pts @ rot_x(phi_rad).T
    zr = pr[:, 2].copy()
    try:
        h = brentq(lambda h: float(np.sum(wts[zr+h<=0]))-V_req, -10, 10, xtol=1e-4)
    except: return 0.0, 0.0, 0.0
    mask = zr+h <= 0
    Vs = float(np.sum(wts[mask]))
    if Vs < 1e-6: return h, 0.0, 0.0
    pw = pr[mask].copy(); pw[:,2] += h
    cob = np.sum(pw*wts[mask,None], axis=0)/Vs
    return h, cob[1], cob[2]  # h_cg, GZ=cob_y, cob_z

ANGLES = np.arange(0, 361, 2)

# ═══════════════════════════════════════════════════════════════
# CASE 1: Symmetric Orion (CG on axis)
# ═══════════════════════════════════════════════════════════════
print("="*70)
print("ORION DEEP DIVE — Why It Needs CMUS")
print("="*70)

print("\n--- Case A: Symmetric Orion (CG on axis) ---")
pts_s, wts_s = make_pts(orion_r, 0, OR_H, OR_CG_Z_SYM, 0.0)
V = float(np.sum(wts_s))
print(f"  Volume = {V:.1f} m³, displaced needed = {OR_M/RHO:.1f} m³ ({OR_M/RHO/V*100:.0f}%)")

gz_sym = np.array([gz_at(pts_s, wts_s, OR_M, np.radians(d))[1] for d in ANGLES])
print(f"  GZ(0°)   = {gz_sym[0]:+.4f} m  ← Stable 1 (upright)")
print(f"  GZ(45°)  = {gz_sym[22]:+.4f} m")
print(f"  GZ(90°)  = {gz_sym[45]:+.4f} m")
print(f"  GZ(135°) = {gz_sym[67]:+.4f} m")
print(f"  GZ(180°) = {gz_sym[90]:+.4f} m  ← Stable 2 (inverted)")
print(f"  GZ_max   = {np.max(gz_sym):+.4f} m at {ANGLES[np.argmax(gz_sym)]}°")

# Find all zero crossings
equil = []
for i in range(1, len(ANGLES)):
    if gz_sym[i-1] * gz_sym[i] < 0:
        # Linear interpolation for zero crossing
        a0, a1 = ANGLES[i-1], ANGLES[i]
        g0, g1 = gz_sym[i-1], gz_sym[i]
        a_cross = a0 + (a1-a0) * (-g0)/(g1-g0)
        slope = (g1-g0)/(a1-a0)
        kind = "STABLE" if slope < 0 else "UNSTABLE"
        equil.append((a_cross, kind, slope))
        print(f"  Equilibrium at {a_cross:.1f}° — {kind} (dGZ/dφ = {slope:.4f})")

# Energy barrier from Stable 2 to Stable 1
# = integral of M*g*|GZ| over the path from Stable 2 to Stable 1 where GZ opposes motion
print("\n  Energy barrier between Stable 1 and Stable 2:")
# Find the unstable equilibrium between them (the "hill" in the energy landscape)
# Going from 180° toward 0°: GZ must be integrated
dphi = np.radians(2)
# Integrate |M*g*GZ| where GZ opposes the motion from 180° toward 0°
# From 180° decreasing: the capsule needs GZ > 0 to roll toward 0°
# If GZ < 0 in some range, that's a barrier
E_barrier_s1_to_s2 = 0.0
E_barrier_s2_to_s1 = 0.0
for i in range(90, 0, -1):  # 180° down to 0°
    # Going from 180° to 0° (decreasing phi): need negative torque (GZ < 0)
    # If GZ > 0 in this range, it pushes BACK toward 180° (barrier)
    if gz_sym[i] > 0:
        E_barrier_s2_to_s1 += OR_M * G * gz_sym[i] * dphi
for i in range(0, 90):  # 0° up to 180°
    # Going from 0° to 180° (increasing phi): need positive torque (GZ > 0)
    # If GZ < 0, it pushes back toward 0° (barrier)
    if gz_sym[i] < 0:
        E_barrier_s1_to_s2 += OR_M * G * abs(gz_sym[i]) * dphi

print(f"    Stable 1 → Stable 2: {E_barrier_s1_to_s2:.0f} J")
print(f"    Stable 2 → Stable 1: {E_barrier_s2_to_s1:.0f} J")
print(f"    Peak GZ opposing S2→S1 recovery: {np.max(gz_sym[:91]):+.4f} m")
print(f"    Peak opposing torque: {OR_M*G*np.max(gz_sym[:91]):.0f} N·m")

# ═══════════════════════════════════════════════════════════════
# CMUS SIZING: What the 5 bags must overcome
# ═══════════════════════════════════════════════════════════════
print("\n--- CMUS Bag Sizing ---")
# At Stable 2 (180°), GZ ≈ 0. But to START righting, the bags must
# create a torque that exceeds the peak restoring GZ on the path to Stable 1.
peak_opposing_gz = np.max(gz_sym[:91])  # max GZ in 0°-180° range
peak_opposing_torque = OR_M * G * peak_opposing_gz
print(f"  Peak opposing torque (keeps capsule in Stable 2): {peak_opposing_torque:.0f} N·m")
print(f"  This occurs at {ANGLES[np.argmax(gz_sym[:91])]}° heel")

# 5 CMUS bags on top of capsule, R ≈ 1.0 m from axis, Z ≈ 3.2 m from base
# When inverted, bags are at Z_world ≈ CG_z - (3.2 - CG_z) = 2*1.5 - 3.2 = -0.2
# below waterline. Buoyancy acts at y ≈ ±1.0 from axis (bags are around perimeter)
CMUS_R_OFFSET = 1.0   # radial offset of bags from axis
CMUS_Z_BODY = 3.0     # Z from base (near apex)
# At ~130° heel (worst case): bag position in world frame
phi_worst = np.radians(ANGLES[np.argmax(gz_sym[:91])])
y_bag = CMUS_R_OFFSET * np.cos(0) * np.cos(phi_worst) - (CMUS_Z_BODY - OR_CG_Z_SYM) * np.sin(phi_worst)
z_bag = CMUS_R_OFFSET * np.cos(0) * np.sin(phi_worst) + (CMUS_Z_BODY - OR_CG_Z_SYM) * np.cos(phi_worst)
print(f"  At {ANGLES[np.argmax(gz_sym[:91])]}° heel: bag y_world = {y_bag:.2f} m, z_world offset = {z_bag:.2f} m")

# Required buoyancy per bag to exceed peak opposing torque
# Total_torque = 5 * F_bag * arm_effective
# arm_effective ≈ |y_bag| at worst angle
arm = max(abs(y_bag), 0.5)
F_needed_total = peak_opposing_torque / arm * 1.5  # 1.5x margin
F_per_bag = F_needed_total / 5
V_per_bag = F_per_bag / (RHO * G)
r_per_bag = (3*V_per_bag/(4*np.pi))**(1/3)

print(f"  Required total buoyancy: {F_needed_total:.0f} N ({F_needed_total/G:.0f} kgf)")
print(f"  Per bag: {F_per_bag:.0f} N → V = {V_per_bag:.3f} m³ → r = {r_per_bag:.2f} m (d = {2*r_per_bag:.2f} m)")
print(f"  5 bags total volume: {5*V_per_bag:.2f} m³")

# ═══════════════════════════════════════════════════════════════
# CASE 2: Orion with CG offset (reentry trim)
# ═══════════════════════════════════════════════════════════════
print("\n--- Case B: Orion with CG offset (0.3 m for reentry L/D) ---")
pts_o, wts_o = make_pts(orion_r, 0, OR_H, OR_CG_Z_SYM, 0.3)
gz_off = np.array([gz_at(pts_o, wts_o, OR_M, np.radians(d))[1] for d in ANGLES])
print(f"  GZ(0°)   = {gz_off[0]:+.4f} m")
print(f"  GZ(180°) = {gz_off[90]:+.4f} m")
equil_off = []
for i in range(1, len(ANGLES)):
    if gz_off[i-1]*gz_off[i] < 0:
        a0,a1 = ANGLES[i-1],ANGLES[i]; g0,g1=gz_off[i-1],gz_off[i]
        a_c = a0+(a1-a0)*(-g0)/(g1-g0)
        sl = (g1-g0)/(a1-a0)
        kind = "STABLE" if sl<0 else "UNSTABLE"
        equil_off.append((a_c, kind))
        print(f"  Equilibrium at {a_c:.1f}° — {kind}")

# ═══════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 12))
fig.patch.set_facecolor('#0f1117')

# Panel 1: Symmetric Orion — clean Stable 1 + Stable 2
ax = axes[0]; ax.set_facecolor('#0f1117')
ax.fill_between(ANGLES, 0, gz_sym*100, where=gz_sym>0, alpha=0.15, color='#2ecc71')
ax.fill_between(ANGLES, 0, gz_sym*100, where=gz_sym<0, alpha=0.15, color='#e74c3c')
ax.plot(ANGLES, gz_sym*100, color='#ff7f0e', lw=2.5, label='Orion CM (symmetric CG)')
ax.axhline(0, color='#555', lw=0.8)

# Annotate stable points
for a_c, kind, sl in equil:
    if kind == "STABLE":
        label = "Stable 1\n(upright)" if a_c < 90 else "Stable 2\n(inverted)"
        ax.annotate(label, xy=(a_c, 0), xytext=(a_c+20, 25 if a_c<90 else -35),
                   fontsize=10, color='#2ecc71', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2),
                   bbox=dict(facecolor='#1a1d27', edgecolor='#2ecc71', boxstyle='round,pad=0.3'))

# Energy barrier annotation
idx_peak = np.argmax(gz_sym[:91])
ax.annotate(f'Energy barrier\nGZ = {gz_sym[idx_peak]*100:+.0f} cm\n'
           f'Torque = {OR_M*G*gz_sym[idx_peak]:.0f} N·m\n'
           f'THIS is why Orion\nneeds 5 CMUS bags',
           xy=(ANGLES[idx_peak], gz_sym[idx_peak]*100),
           xytext=(ANGLES[idx_peak]+40, gz_sym[idx_peak]*100+15),
           fontsize=9, color='#f1c40f', fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='#f1c40f', lw=2),
           bbox=dict(facecolor='#1a1d27', edgecolor='#f1c40f', boxstyle='round,pad=0.4'))

# Shade the barrier region
barrier_mask = (ANGLES >= 0) & (ANGLES <= 180) & (gz_sym > 0)
ax.fill_between(ANGLES, 0, gz_sym*100, where=barrier_mask,
               alpha=0.2, color='#f1c40f', hatch='///', label='Energy barrier (S2→S1)')

ax.set_ylabel('GZ [cm]', color='#e8eaf0', fontsize=11)
ax.set_title('NASA Orion — Symmetric: Two Strong Stable Equilibria',
            color='#ff7f0e', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, facecolor='#1a1d27', edgecolor='#2d3244', labelcolor='#e8eaf0')
ax.grid(True, alpha=0.15, color='#2d3244'); ax.tick_params(colors='#9aa0b0')
ax.set_xlim(0, 360)

# Panel 2: With CG offset
ax = axes[1]; ax.set_facecolor('#0f1117')
ax.fill_between(ANGLES, 0, gz_off*100, where=gz_off>0, alpha=0.15, color='#2ecc71')
ax.fill_between(ANGLES, 0, gz_off*100, where=gz_off<0, alpha=0.15, color='#e74c3c')
ax.plot(ANGLES, gz_off*100, color='#ff7f0e', lw=2.5,
        label='Orion CM (CG offset 0.3 m for reentry L/D)')
ax.plot(ANGLES, gz_sym*100, color='#ff7f0e', lw=1, ls=':', alpha=0.4,
        label='Symmetric (reference)')
ax.axhline(0, color='#555', lw=0.8)

for a_c, kind in equil_off:
    if kind == "STABLE":
        ax.plot(a_c, 0, 'v', color='#2ecc71', ms=14, zorder=10)
        ax.annotate(f'Stable ~{a_c:.0f}°', xy=(a_c, 0), xytext=(a_c+15, -25),
                   fontsize=9, color='#2ecc71', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#2ecc71'))

ax.set_ylabel('GZ [cm]', color='#e8eaf0', fontsize=11)
ax.set_xlabel('Heel angle [deg]', color='#e8eaf0')
ax.set_title('NASA Orion — With CG Offset: Equilibria Shift but Both Remain Strong',
            color='#ff7f0e', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, facecolor='#1a1d27', edgecolor='#2d3244', labelcolor='#e8eaf0')
ax.grid(True, alpha=0.15, color='#2d3244'); ax.tick_params(colors='#9aa0b0')
ax.set_xlim(0, 360)

plt.tight_layout()
plt.savefig(f"{OUT}/flotation_orion_deepdive.png", dpi=150, facecolor='#0f1117',
            bbox_inches='tight')
print(f"\nSaved flotation_orion_deepdive.png")
print("\nDone.")
