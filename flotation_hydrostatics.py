#!/usr/bin/env python3
"""
Bloon Capsule Flotation Stability Analysis — Hydrostatics
==========================================================
Computes GZ curves, metacentric height, and stability verdict
for the BLOON HTPV-180 capsule floating in seawater.

Six analysis cases:
  1. Actual mass 1319.7 kg, no airbag, symmetric crew
  2. Actual mass, airbag inflated, symmetric crew
  3. Actual mass, no airbag, worst-case asymmetric crew
  4. Actual mass, airbag inflated, worst-case asymmetric crew
  5. Design mass 1500 kg, no airbag, symmetric crew
  6. Design mass, airbag inflated, symmetric crew

Geometry from: BLOON_CAPSULE_COMPLETE.md (Z2I-DD-010-A)
Airbag from:   AIRBAG_SPEC.md (Z2I-DD-004-A)
Volume check:  geometry_verification.py (htpv180_volume)

Author: Claude / Z2I CAD project
Date:   2026-04-13
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time, sys

# ═══════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════
RHO_WATER = 1025.0    # kg/m³  seawater
G_ACC     = 9.81      # m/s²

# ═══════════════════════════════════════════════════════════════
# CAPSULE GEOMETRY  [metres]
# ═══════════════════════════════════════════════════════════════
# Body frame origin: torus centre (Z_world = 1.100 m above ground datum)
R_CROWN  = 0.600      # Crown-circle radius
A_O      = 0.900      # Extrados semi-axis (circular arc)
A_I      = 0.475      # Intrados horizontal semi-axis (elliptical)
B_VERT   = 0.900      # Vertical semi-axis (both surfaces)
Z_CENTER = 1.100      # Torus centre height in world frame

Z_TOP_W  = Z_CENTER + B_VERT   # 2.000 m  (capsule top, world)
Z_BOT_W  = Z_CENTER - B_VERT   # 0.200 m  (capsule bottom, world)

# Keel cone
KEEL_TIP_Z  = -0.871   # Keel nose tip (lowest point, world)
KEEL_BASE_Z =  0.200   # Keel max-radius height = torus bottom (world)
KEEL_BASE_R =  0.600   # Max keel radius at base
KEEL_NOSE_R =  0.300   # Spherical nose-cap radius

# Airbag torus (inflated geometry)
AB_R_MAJ  = 1.220      # Major radius
AB_R_MIN  = 0.420      # Minor radius
AB_Z_CTR  = -0.180     # Torus centre height (world)
AB_Z_TOP  =  0.240     # Top of airbag (world)
AB_Z_BOT  = -0.600     # Bottom of airbag (world)

# ═══════════════════════════════════════════════════════════════
# HULL PROFILE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _r_ext(z_w):
    """Extrados outer radius at world height z_w (scalar)."""
    dz = z_w - Z_CENTER
    if abs(dz) > B_VERT:
        return 0.0
    ct = np.sqrt(1.0 - (dz / B_VERT) ** 2)
    return R_CROWN + A_O * ct

def _r_keel(z_w):
    """Keel-cone radius at world height z_w (scalar).
    Linear cone from tip to base; spherical nose ignored for volume."""
    if z_w < KEEL_TIP_Z or z_w > KEEL_BASE_Z:
        return 0.0
    frac = (z_w - KEEL_TIP_Z) / (KEEL_BASE_Z - KEEL_TIP_Z)
    return KEEL_BASE_R * frac

def hull_r(z_w):
    """Max outer radius of sealed hull (torus + keel) at world height z_w."""
    return max(_r_ext(z_w), _r_keel(z_w))


# Vectorised versions for arrays
def hull_r_vec(z_arr):
    """Vectorised hull outer radius."""
    out = np.zeros_like(z_arr)
    # Torus part
    dz = z_arr - Z_CENTER
    mask_t = np.abs(dz) <= B_VERT
    ct = np.sqrt(np.clip(1.0 - (dz[mask_t] / B_VERT) ** 2, 0, 1))
    out[mask_t] = np.maximum(out[mask_t], R_CROWN + A_O * ct)
    # Keel part
    mask_k = (z_arr >= KEEL_TIP_Z) & (z_arr <= KEEL_BASE_Z)
    frac = (z_arr[mask_k] - KEEL_TIP_Z) / (KEEL_BASE_Z - KEEL_TIP_Z)
    out[mask_k] = np.maximum(out[mask_k], KEEL_BASE_R * frac)
    return out

# ═══════════════════════════════════════════════════════════════
# MASS BUDGET
# ═══════════════════════════════════════════════════════════════

@dataclass
class MassItem:
    name: str
    mass: float   # kg
    x: float      # m, body frame (origin = torus centre)
    y: float
    z: float

def _crew(symmetric: bool) -> List[MassItem]:
    """Return crew mass items.  Asymmetric = 4 passengers clustered at +Y."""
    r = 1.200
    if symmetric:
        angles = [122, 191, 223, 293]   # assigned seat azimuths (deg)
    else:
        angles = [75, 85, 95, 105]      # all near +Y for worst-case X-heel
    items = []
    for th in angles:
        t = np.radians(th)
        items.append(MassItem(f"Pax@{th}", 80.0, r*np.cos(t), r*np.sin(t), 0.0))
    # Pilot at theta=324
    tp = np.radians(324)
    items.append(MassItem("Pilot", 81.0, r*np.cos(tp), r*np.sin(tp), 0.0))
    return items

def build_mass_items(symmetric_crew=True, design_mass=False) -> List[MassItem]:
    items: List[MassItem] = []

    # ── Structure 305.8 kg ──
    items.append(MassItem("Shell outer",        118.2, 0, 0,  0.000))
    items.append(MassItem("Shell roof",          18.3, 0, 0,  0.550))
    items.append(MassItem("Shell floor",         19.5, 0, 0, -0.550))
    items.append(MassItem("Shell trans+cyl",     17.55,0, 0,  0.000))
    items.append(MassItem("Weld filler",          2.0, 0, 0,  0.000))

    for th in [36, 108, 180, 252]:
        t = np.radians(th)
        items.append(MassItem(f"Win@{th}", 22.5/4, 1.35*np.cos(t), 1.35*np.sin(t), 0.0))

    th_h = np.radians(324)
    items.append(MassItem("Hatch", 8.7, 0.80*np.cos(th_h), 0.80*np.sin(th_h), 0.700))

    # Keel CG: solid cone CG at 1/4 height from base
    keel_h = KEEL_BASE_Z - KEEL_TIP_Z
    keel_cg_w = KEEL_TIP_Z + 0.75 * keel_h          # world
    items.append(MassItem("Keel", 35.9, 0, 0, keel_cg_w - Z_CENTER))

    items.append(MassItem("Floor", 15.0, 0, 0, -0.600))

    for i, th in enumerate([122, 191, 223, 293, 324]):
        t = np.radians(th)
        items.append(MassItem(f"Seat{i}", 1.5, 1.20*np.cos(t), 1.20*np.sin(t), 0.0))

    items.append(MassItem("Airbag stowed", 6.2, 0, 0, AB_Z_CTR - Z_CENTER))
    items.append(MassItem("Interior",     28.2, 0, 0,  0.0))
    items.append(MassItem("Suspension",    7.5, 0, 0,  0.686))

    # ── Subsystems 1013.85 kg ──
    items.append(MassItem("Power",       117.8,  0, 0, -0.750))

    th_tm = np.radians(288)
    items.append(MassItem("Thermal", 86.35, 1.10*np.cos(th_tm), 1.10*np.sin(th_tm), 0.0))

    items.append(MassItem("LifeSupport",  95.4, 0, 0, -0.750))
    items.append(MassItem("Comms",        22.5, 0, 0,  1.000))
    items.append(MassItem("Avionics",      7.8, 1.10, 0, -0.200))
    items.append(MassItem("CabinEquip",  100.0, 0, 0,  0.000))

    th_se = np.radians(288)
    items.append(MassItem("Safety", 159.0, 0.90*np.cos(th_se), 0.90*np.sin(th_se), 0.0))

    items.append(MassItem("Parachute", 104.0, 0, 0, 1.000))

    # Crew
    items.extend(_crew(symmetric_crew))

    # Design-mass margin
    actual = sum(it.mass for it in items)
    if design_mass:
        margin = 1500.0 - actual
        if margin > 0:
            items.append(MassItem("Margin", margin, 0, 0, 0.0))

    return items


def cg_and_inertia(items: List[MassItem]):
    """Return (total_mass, cg_xyz, inertia_3x3) about CG."""
    M = sum(it.mass for it in items)
    cx = sum(it.mass * it.x for it in items) / M
    cy = sum(it.mass * it.y for it in items) / M
    cz = sum(it.mass * it.z for it in items) / M
    cg = np.array([cx, cy, cz])

    I = np.zeros((3, 3))
    for it in items:
        d = np.array([it.x, it.y, it.z]) - cg
        m = it.mass
        I[0, 0] += m * (d[1]**2 + d[2]**2)
        I[1, 1] += m * (d[0]**2 + d[2]**2)
        I[2, 2] += m * (d[0]**2 + d[1]**2)
        I[0, 1] -= m * d[0] * d[1]
        I[0, 2] -= m * d[0] * d[2]
        I[1, 2] -= m * d[1] * d[2]
    I[1, 0] = I[0, 1]; I[2, 0] = I[0, 2]; I[2, 1] = I[1, 2]
    return M, cg, I


# ═══════════════════════════════════════════════════════════════
# VOLUME-POINT CLOUD  (for arbitrary-angle buoyancy computation)
# ═══════════════════════════════════════════════════════════════

def make_body_points(include_airbag=False, n_z=250, n_alpha=48):
    """Generate interior volume-element points for the hull (+airbag).

    Points are in BODY frame (origin = torus centre, Z up).
    Returns (pts (N,3), wts (N,) volume weights in m³).
    """
    z_lo = KEEL_TIP_Z - Z_CENTER            # body frame
    z_hi = (Z_TOP_W - Z_CENTER) + 0.001
    if include_airbag:
        z_lo = min(z_lo, AB_Z_BOT - Z_CENTER)

    z_arr = np.linspace(z_lo, z_hi, n_z)
    dz = z_arr[1] - z_arr[0]
    da = 2.0 * np.pi / n_alpha
    alphas = (np.arange(n_alpha) + 0.5) * da

    all_p, all_w = [], []

    for zb in z_arr:
        zw = zb + Z_CENTER                   # world height
        rh = hull_r(zw)                       # hull outer radius

        # Airbag annulus at this height
        rab_in = rab_out = 0.0
        if include_airbag:
            dz_ab = zw - AB_Z_CTR
            if abs(dz_ab) < AB_R_MIN:
                dr = np.sqrt(AB_R_MIN**2 - dz_ab**2)
                rab_in  = AB_R_MAJ - dr
                rab_out = AB_R_MAJ + dr

        r_max = max(rh, rab_out)
        if r_max < 1e-4:
            continue

        n_r = max(5, int(r_max / 0.025))     # ~25 mm radial spacing
        dr = r_max / n_r
        r_arr = (np.arange(n_r) + 0.5) * dr  # cell centres

        # Meshgrid (n_alpha × n_r)
        R, A = np.meshgrid(r_arr, alphas)     # both shape (n_alpha, n_r)

        inside_hull   = R <= rh
        inside_airbag = (R >= rab_in) & (R <= rab_out) if rab_out > 0 else np.zeros_like(R, dtype=bool)
        inside = inside_hull | inside_airbag

        if not np.any(inside):
            continue

        X = (R * np.cos(A))[inside]
        Y = (R * np.sin(A))[inside]
        Z = np.full(X.shape, zb)
        W = (R * dr * da * dz)[inside]

        all_p.append(np.column_stack([X, Y, Z]))
        all_w.append(W)

    pts = np.vstack(all_p)
    wts = np.concatenate(all_w)
    return pts, wts


# ═══════════════════════════════════════════════════════════════
# BUOYANCY CORE
# ═══════════════════════════════════════════════════════════════

def _submerged_vol(z_rot, wts, h_cg):
    """Volume below waterline (z_world ≤ 0) given rotated-z and CG height."""
    return float(np.sum(wts[z_rot + h_cg <= 0.0]))

def find_equilibrium(pts, wts, cg_body, rot, total_mass):
    """Find CG height above waterline for hydrostatic equilibrium.

    Parameters
    ----------
    pts, wts : hull volume points/weights in body frame
    cg_body  : (3,) CG in body frame
    rot      : (3,3) rotation matrix body→world
    total_mass : kg

    Returns
    -------
    h_cg     : CG height above waterline (m)
    V_sub    : submerged volume (m³)
    cob      : (3,) centre of buoyancy in world frame
    """
    V_req = total_mass / RHO_WATER

    # Pre-rotate (same for every draft trial)
    pts_rot = (pts - cg_body) @ rot.T        # world-oriented, origin at CG
    z_rot   = pts_rot[:, 2].copy()           # z-world = z_rot + h_cg

    res = lambda h: _submerged_vol(z_rot, wts, h) - V_req

    # Quick bracket check
    V_tot = float(np.sum(wts))
    if V_tot < V_req:
        print(f"  WARNING  total body volume {V_tot:.3f} m³ < required {V_req:.3f} m³")

    h = brentq(res, -4.0, 4.0, xtol=1e-5, maxiter=80)

    # CoB at equilibrium
    zw = z_rot + h
    mask = zw <= 0.0
    V_sub = float(np.sum(wts[mask]))
    pw = pts_rot[mask] + np.array([0.0, 0.0, h])
    cob = np.sum(pw * wts[mask, None], axis=0) / V_sub

    return h, V_sub, cob


def rot_x(phi):
    """Rotation matrix about X-axis by angle phi (rad)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


# ═══════════════════════════════════════════════════════════════
# GZ CURVE + METACENTRIC HEIGHT
# ═══════════════════════════════════════════════════════════════

def gz_curve(pts, wts, cg_body, total_mass, angles_deg):
    """Compute righting arm GZ for each heel angle (rotation about X-axis).

    GZ = y_B - y_G  in world frame (y_G = 0 by construction).
    Positive GZ → restoring (stable).
    """
    gz = np.empty(len(angles_deg))
    drafts = np.empty(len(angles_deg))

    for i, deg in enumerate(angles_deg):
        R = rot_x(np.radians(deg))
        h, V, cob = find_equilibrium(pts, wts, cg_body, R, total_mass)
        gz[i] = cob[1]          # GZ = B_y  (G_y = 0)
        drafts[i] = h
    return gz, drafts


def metacentric_height(z_waterline, V_disp, z_B, z_G, include_airbag=False):
    """GM at upright equilibrium.  Positive → initially stable."""
    r_h = hull_r(z_waterline)

    if include_airbag:
        dz_ab = z_waterline - AB_Z_CTR
        if abs(dz_ab) < AB_R_MIN:
            dr = np.sqrt(AB_R_MIN**2 - dz_ab**2)
            r_out = AB_R_MAJ + dr
            r_in  = AB_R_MAJ - dr
            # Union waterplane: hull disk + airbag annulus
            if r_h >= r_out:
                I_wp = np.pi / 4 * r_h**4
            elif r_h >= r_in:
                I_wp = np.pi / 4 * r_out**4
            else:
                I_wp = np.pi / 4 * (r_h**4 + r_out**4 - r_in**4)
        else:
            I_wp = np.pi / 4 * r_h**4
    else:
        I_wp = np.pi / 4 * r_h**4

    BM = I_wp / V_disp
    BG = z_G - z_B                           # positive when G above B
    GM = BM - BG
    return GM, BM, BG, I_wp


# ═══════════════════════════════════════════════════════════════
# ANALYSIS RUNNER
# ═══════════════════════════════════════════════════════════════

HEEL_ANGLES = np.arange(0, 361, 2)           # 0 … 360 deg, 2-deg steps

@dataclass
class CaseResult:
    label: str
    mass: float
    cg: np.ndarray
    inertia: np.ndarray
    gz: np.ndarray
    drafts: np.ndarray
    GM: float
    BM: float
    BG: float
    z_wl: float
    V_sub: float
    cob_upright: np.ndarray

def run_case(label, symmetric_crew, design_mass, include_airbag) -> CaseResult:
    t0 = time.time()
    print(f"\n{'─'*60}")
    print(f"  Case: {label}")
    print(f"{'─'*60}")

    items = build_mass_items(symmetric_crew, design_mass)
    M, cg, I = cg_and_inertia(items)
    print(f"  Total mass      = {M:.1f} kg")
    print(f"  CG (body frame) = ({cg[0]:+.4f}, {cg[1]:+.4f}, {cg[2]:+.4f}) m")
    print(f"  CG world-Z      = {cg[2] + Z_CENTER:.3f} m")
    print(f"  Ixx={I[0,0]:.1f}  Iyy={I[1,1]:.1f}  Izz={I[2,2]:.1f} kg·m²")

    pts, wts = make_body_points(include_airbag)
    V_tot = float(np.sum(wts))
    V_req = M / RHO_WATER
    print(f"  Hull volume     = {V_tot:.3f} m³  (points: {len(wts):,})")
    print(f"  Displ. required = {V_req:.3f} m³")

    # Upright equilibrium
    h0, V0, cob0 = find_equilibrium(pts, wts, cg, rot_x(0), M)
    z_wl = -h0 + cg[2] + Z_CENTER            # waterline in world-Z ≈ 0 – h_cg
    # More precisely: waterline is at z_world=0, CG at z_world=h0
    # CG world = h0;  waterline world = 0
    # waterline in capsule ground-datum coords: need to convert
    # The body-frame origin (torus centre) is at world z = h0 - cg[2]
    z_wl_world = 0.0                          # by definition (water surface)
    z_origin_world = h0 - cg[2]              # torus-centre in world
    z_wl_body = -h0 + cg[2]                  # waterline in body frame = -(h0 - cg[2])
    z_wl_capsule = z_wl_body + Z_CENTER      # waterline in capsule ground-datum
    print(f"  CG height above WL = {h0:.4f} m")
    print(f"  Waterline (capsule datum) ≈ {z_wl_capsule:.3f} m")
    print(f"  CoB world = ({cob0[0]:+.4f}, {cob0[1]:+.4f}, {cob0[2]:+.4f})")
    print(f"  V_sub check = {V0:.4f} m³  (target {V_req:.4f})")

    # Metacentric height
    # z_B, z_G measured from keel (or simply use world-frame z-values)
    z_B_world = cob0[2]
    z_G_world = h0                            # CG is at z_world = h0
    GM, BM, BG, Iwp = metacentric_height(
        z_wl_capsule, V0, z_B_world, z_G_world, include_airbag)
    print(f"  I_wp = {Iwp:.4f} m⁴    BM = {BM:.4f} m")
    print(f"  BG = {BG:.4f} m   GM = {GM:+.4f} m  "
          f"{'STABLE' if GM > 0 else 'UNSTABLE'}")

    # GZ curve
    print(f"  Computing GZ curve ({len(HEEL_ANGLES)} angles) …", end="", flush=True)
    gz, drafts = gz_curve(pts, wts, cg, M, HEEL_ANGLES)
    dt = time.time() - t0
    print(f"  done  ({dt:.1f}s)")

    # Stability summary
    gz_max = np.max(gz)
    gz_min = np.min(gz)
    angle_max = HEEL_ANGLES[np.argmax(gz)]
    pos_range = HEEL_ANGLES[gz > 0]
    print(f"  GZ_max = {gz_max:+.4f} m  at {angle_max}°")
    print(f"  GZ range positive: "
          f"{pos_range[0]}°–{pos_range[-1]}°" if len(pos_range) else "NONE")

    return CaseResult(label, M, cg, I, gz, drafts, GM, BM, BG,
                      z_wl_capsule, V0, cob0)


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
STYLES = ['-', '-', '--', '--', ':', ':']

def plot_gz(results: List[CaseResult], path: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Full 360° GZ
    ax = axes[0]
    for i, r in enumerate(results):
        ax.plot(HEEL_ANGLES, r.gz * 100, color=COLORS[i], ls=STYLES[i],
                lw=2, label=r.label)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('GZ  [cm]')
    ax.set_title('BLOON Capsule — Righting Arm (GZ) vs Heel Angle')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Zoomed 0–180°
    ax = axes[1]
    mask = HEEL_ANGLES <= 180
    for i, r in enumerate(results):
        ax.plot(HEEL_ANGLES[mask], r.gz[mask] * 100, color=COLORS[i],
                ls=STYLES[i], lw=2, label=r.label)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Heel angle  [deg]')
    ax.set_ylabel('GZ  [cm]')
    ax.set_title('Righting Arm — 0° to 180°')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved {path}")


def plot_waterlines(results: List[CaseResult], path: str):
    """Cross-section view showing waterline at several heel angles."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Meridional profile (r, z) in capsule ground-datum frame
    theta_p = np.linspace(-np.pi/2, np.pi/2, 300)
    r_ext = R_CROWN + A_O * np.cos(theta_p)
    z_ext = Z_CENTER + B_VERT * np.sin(theta_p)
    r_int = R_CROWN - A_I * np.cos(theta_p)
    z_int = Z_CENTER + B_VERT * np.sin(theta_p)

    # Keel profile
    z_keel = np.linspace(KEEL_TIP_Z, KEEL_BASE_Z, 100)
    r_keel = KEEL_BASE_R * (z_keel - KEEL_TIP_Z) / (KEEL_BASE_Z - KEEL_TIP_Z)

    # Airbag profile (half cross-section)
    phi_ab = np.linspace(0, 2*np.pi, 200)
    r_ab = AB_R_MAJ + AB_R_MIN * np.cos(phi_ab)
    z_ab = AB_Z_CTR + AB_R_MIN * np.sin(phi_ab)

    for ax_i, (case_idx, title) in enumerate([
            (0, "No Airbag, Symmetric"),
            (1, "With Airbag, Symmetric"),
            (2, "No Airbag, Asymmetric")]):
        if case_idx >= len(results):
            break
        ax = axes[ax_i]
        res = results[case_idx]

        # Draw hull cross-section
        ax.plot(r_ext, z_ext, 'b-', lw=1.5)
        ax.plot(-r_ext, z_ext, 'b-', lw=1.5)
        ax.plot(r_int, z_int, 'b--', lw=0.8, alpha=0.5)
        ax.plot(-r_int, z_int, 'b--', lw=0.8, alpha=0.5)
        ax.fill_between(np.concatenate([r_ext, r_int[::-1]]),
                        np.concatenate([z_ext, z_int[::-1]]),
                        alpha=0.06, color='steelblue')
        ax.fill_between(np.concatenate([-r_ext, -r_int[::-1]]),
                        np.concatenate([z_ext, z_int[::-1]]),
                        alpha=0.06, color='steelblue')

        # Keel
        ax.plot(r_keel, z_keel, 'b-', lw=1.5)
        ax.plot(-r_keel, z_keel, 'b-', lw=1.5)
        ax.fill_betweenx(z_keel, -r_keel, r_keel, alpha=0.10, color='sienna')

        # Airbag (cases 1, 3)
        if case_idx in [1, 3]:
            ax.plot(r_ab, z_ab, '-', color='orange', lw=2)
            ax.plot(-r_ab, z_ab, '-', color='orange', lw=2)

        # Waterline
        wl = res.z_wl
        ax.axhline(wl, color='cyan', lw=2, ls='-', label=f'WL = {wl:.2f} m')
        # Shade water below
        ax.axhspan(-1.5, wl, alpha=0.08, color='cyan')

        # CG marker
        cg_z_w = res.cg[2] + Z_CENTER
        ax.plot(0, cg_z_w, 'r+', ms=14, mew=3, label=f'CG Z={cg_z_w:.2f} m')

        # CoB marker
        # CoB is in world frame where waterline=0; shift to capsule datum
        cob_z_datum = res.cob_upright[2] + (res.cg[2] + Z_CENTER) - res.drafts[0]
        # Actually simpler: cob z_world = res.cob_upright[2],
        # capsule-datum = z_world + offset where torus-centre is at z_world = h0 - cg_z
        # Let's just mark it relative to the hull
        ax.plot(0, cob_z_datum, 'g^', ms=10, label=f'CoB')

        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-1.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(f'{title}\nGM = {res.GM:+.3f} m  '
                     f'({"STABLE" if res.GM > 0 else "UNSTABLE"})',
                     fontsize=11)
        ax.set_xlabel('r [m]')
        ax.set_ylabel('z [m]  (capsule datum)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}")


def write_report(results: List[CaseResult], path: str):
    lines = []
    lines.append("=" * 70)
    lines.append("BLOON CAPSULE FLOTATION STABILITY REPORT")
    lines.append(f"Generated: 2026-04-13   Water: seawater {RHO_WATER} kg/m³")
    lines.append("=" * 70)

    for r in results:
        lines.append("")
        lines.append(f"Case: {r.label}")
        lines.append("-" * 50)
        lines.append(f"  Mass           = {r.mass:.1f} kg")
        lines.append(f"  CG body-frame  = ({r.cg[0]:+.4f}, {r.cg[1]:+.4f}, {r.cg[2]:+.4f}) m")
        lines.append(f"  CG world-Z     = {r.cg[2] + Z_CENTER:.3f} m")
        lines.append(f"  Waterline (capsule datum) = {r.z_wl:.3f} m")
        lines.append(f"  V_displaced    = {r.V_sub:.4f} m³")
        lines.append(f"  GM (upright)   = {r.GM:+.4f} m  "
                     f"({'STABLE' if r.GM > 0 else '** UNSTABLE **'})")
        lines.append(f"  BM             = {r.BM:.4f} m")
        lines.append(f"  BG             = {r.BG:.4f} m")
        gz = r.gz
        lines.append(f"  GZ_max         = {np.max(gz):+.4f} m  at {HEEL_ANGLES[np.argmax(gz)]}°")
        lines.append(f"  GZ_min         = {np.min(gz):+.4f} m  at {HEEL_ANGLES[np.argmin(gz)]}°")
        pos = HEEL_ANGLES[gz > 0.001]
        if len(pos) > 0:
            lines.append(f"  Positive GZ    = {pos[0]}° to {pos[-1]}°")
        else:
            lines.append(f"  Positive GZ    = NONE (unstable at all angles)")

        # Equilibrium check: is capsule self-righting from any orientation?
        # Self-righting if GZ > 0 for all angles 0 < phi < 180 (excluding 0 and 180)
        gz_0_180 = gz[1:90]   # angles 2,4,...,178
        if np.all(gz_0_180 > 0):
            lines.append(f"  Self-righting  = YES (GZ > 0 for all 0°<φ<180°)")
        else:
            # Find vanishing stability angle
            first_neg = HEEL_ANGLES[1:90][gz_0_180 <= 0]
            if len(first_neg):
                lines.append(f"  Self-righting  = NO  (GZ ≤ 0 first at {first_neg[0]}°)")
            else:
                lines.append(f"  Self-righting  = MARGINAL")

        # Inertia
        I = r.inertia
        lines.append(f"  Inertia [kg·m²]:")
        lines.append(f"    Ixx = {I[0,0]:.1f}   Iyy = {I[1,1]:.1f}   Izz = {I[2,2]:.1f}")
        lines.append(f"    Ixy = {I[0,1]:.1f}   Ixz = {I[0,2]:.1f}   Iyz = {I[1,2]:.1f}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    txt = "\n".join(lines)
    with open(path, 'w') as f:
        f.write(txt)
    print(f"  Saved {path}")
    print(txt)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("BLOON CAPSULE FLOTATION STABILITY ANALYSIS")
    print("=" * 70)

    # Volume sanity check
    z_test = np.linspace(KEEL_TIP_Z, Z_TOP_W, 5000)
    dz_t = z_test[1] - z_test[0]
    rr = hull_r_vec(z_test)
    V_hull = float(np.sum(np.pi * rr**2 * dz_t))
    print(f"\nHull volume (sealed, numerical) = {V_hull:.3f} m³")
    print(f"  Documented torus annular vol  = 9.53 m³")
    print(f"  Full-envelope vol (torus+keel) expected ~10.6 m³")

    # Airbag volume
    z_ab = np.linspace(AB_Z_BOT, AB_Z_TOP, 2000)
    dz_ab = z_ab[1] - z_ab[0]
    V_ab = 0.0
    for z in z_ab:
        dza = z - AB_Z_CTR
        if abs(dza) < AB_R_MIN:
            dr = np.sqrt(AB_R_MIN**2 - dza**2)
            r_o = AB_R_MAJ + dr
            r_i = AB_R_MAJ - dr
            V_ab += np.pi * (r_o**2 - r_i**2) * dz_ab
    V_ab_exact = 2 * np.pi**2 * AB_R_MAJ * AB_R_MIN**2
    print(f"\nAirbag volume (numerical)  = {V_ab:.3f} m³")
    print(f"Airbag volume (analytical) = {V_ab_exact:.3f} m³")

    # ── Run 6 cases ──
    cases = [
        ("1: Actual, no bag, symmetric",       True,  False, False),
        ("2: Actual, AIRBAG, symmetric",        True,  False, True),
        ("3: Actual, no bag, ASYMMETRIC crew",  False, False, False),
        ("4: Actual, AIRBAG, ASYMMETRIC crew",  False, False, True),
        ("5: Design 1500kg, no bag, symmetric", True,  True,  False),
        ("6: Design 1500kg, AIRBAG, symmetric", True,  True,  True),
    ]

    results: List[CaseResult] = []
    for label, sym, des, bag in cases:
        results.append(run_case(label, sym, des, bag))

    # ── Plots ──
    base = "/Users/josemarianolopezurdiales/Documents/CAD"
    plot_gz(results, f"{base}/flotation_GZ_curves.png")
    plot_waterlines(results, f"{base}/flotation_waterlines.png")
    write_report(results, f"{base}/flotation_report.txt")

    print("\n✓  Flotation hydrostatic analysis complete.")


if __name__ == "__main__":
    main()
