#!/Users/josemarianolopezurdiales/miniconda3/envs/chrono/bin/python
"""
Bloon Capsule Flotation Dynamics — Project Chrono
==================================================
Dynamic self-righting simulation with buoyancy forces and wave excitation.

Test cases (4 orientations × 2 airbag states = 8 runs):
  Upright, 45° heel, 90° on side, 180° inverted
  Each with airbag deflated and inflated.

Follows Chrono patterns from sim_6_chrono_bat.py:
  ChSystemNSC, EULER_IMPLICIT_LINEARIZED, BARZILAIBORWEIN solver,
  force accumulator pattern.

Author: Claude / Z2I CAD project
Date:   2026-04-13
"""

import math, time, sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pychrono as chrono

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
RHO_WATER = 1025.0
G_ACC     = 9.81

# ═══════════════════════════════════════════════════════════════
# CAPSULE GEOMETRY  [metres, same as flotation_hydrostatics.py]
# ═══════════════════════════════════════════════════════════════
R_CROWN  = 0.600;  A_O = 0.900;  A_I = 0.475;  B_VERT = 0.900
Z_CENTER = 1.100
Z_TOP_W  = Z_CENTER + B_VERT
Z_BOT_W  = Z_CENTER - B_VERT

KEEL_TIP_Z  = -0.871;  KEEL_BASE_Z = 0.200;  KEEL_BASE_R = 0.600
AB_R_MAJ = 1.220;  AB_R_MIN = 0.420;  AB_Z_CTR = -0.180
AB_Z_TOP = 0.240;  AB_Z_BOT = -0.600

# Simulation
SIM_T   = 40.0      # seconds per run
DT      = 0.001     # 1 kHz timestep
LOG_DT  = 0.02      # log every 20 ms
WAVE_H  = 1.0       # wave height (m, crest-to-trough)
WAVE_T  = 6.0       # wave period (s)

# Drag coefficients — high values model water resistance + added mass
CD_LIN  = 4000.0    # N·s/m   linear drag (water is dense)
CD_ANG  = 3000.0    # N·m·s   angular drag
K_SURF  = 5000.0    # N/m     restoring spring when body leaves water

# ═══════════════════════════════════════════════════════════════
# HULL PROFILE (same as hydrostatics)
# ═══════════════════════════════════════════════════════════════

def hull_r(z_w):
    r = 0.0
    dz = z_w - Z_CENTER
    if abs(dz) <= B_VERT:
        ct = math.sqrt(max(0.0, 1.0 - (dz / B_VERT) ** 2))
        r = max(r, R_CROWN + A_O * ct)
    if KEEL_TIP_Z <= z_w <= KEEL_BASE_Z:
        frac = (z_w - KEEL_TIP_Z) / (KEEL_BASE_Z - KEEL_TIP_Z)
        r = max(r, KEEL_BASE_R * frac)
    return r

# ═══════════════════════════════════════════════════════════════
# MASS BUDGET  (symmetric crew, actual mass)
# ═══════════════════════════════════════════════════════════════

def compute_mass_props():
    """Return (total_mass, cg_body, Ixx, Iyy, Izz) for symmetric crew."""
    # Simplified: use values computed by hydrostatics script
    # Case 1: actual mass, symmetric crew
    M   = 1400.9
    cg  = np.array([-0.0139, -0.2689, -0.0594])   # body frame (torus centre)
    Ixx = 743.0;  Iyy = 684.0;  Izz = 776.0
    return M, cg, Ixx, Iyy, Izz

# ═══════════════════════════════════════════════════════════════
# VOLUME POINT CLOUD  (coarser for dynamics)
# ═══════════════════════════════════════════════════════════════

def make_body_points(include_airbag=False, n_z=120, n_alpha=24):
    """Generate volume-weighted interior points (body frame, torus-centre origin)."""
    z_lo = KEEL_TIP_Z - Z_CENTER
    z_hi = (Z_TOP_W - Z_CENTER) + 0.001
    if include_airbag:
        z_lo = min(z_lo, AB_Z_BOT - Z_CENTER)

    z_arr = np.linspace(z_lo, z_hi, n_z)
    dz = z_arr[1] - z_arr[0]
    da = 2.0 * np.pi / n_alpha
    alphas = (np.arange(n_alpha) + 0.5) * da

    all_p, all_w = [], []
    for zb in z_arr:
        zw = zb + Z_CENTER
        rh = hull_r(zw)

        rab_in = rab_out = 0.0
        if include_airbag:
            dz_ab = zw - AB_Z_CTR
            if abs(dz_ab) < AB_R_MIN:
                dr = math.sqrt(AB_R_MIN**2 - dz_ab**2)
                rab_in  = AB_R_MAJ - dr
                rab_out = AB_R_MAJ + dr

        r_max = max(rh, rab_out)
        if r_max < 1e-4:
            continue

        n_r = max(3, int(r_max / 0.04))       # ~40 mm spacing
        dr = r_max / n_r
        r_arr = (np.arange(n_r) + 0.5) * dr

        R, A = np.meshgrid(r_arr, alphas)
        inside_hull = R <= rh
        inside_ab = (R >= rab_in) & (R <= rab_out) if rab_out > 0 else np.zeros_like(R, bool)
        inside = inside_hull | inside_ab
        if not np.any(inside):
            continue

        X = (R * np.cos(A))[inside]
        Y = (R * np.sin(A))[inside]
        Z = np.full(X.shape, zb)
        W = (R * dr * da * dz)[inside]
        all_p.append(np.column_stack([X, Y, Z]))
        all_w.append(W)

    return np.vstack(all_p), np.concatenate(all_w)


# ═══════════════════════════════════════════════════════════════
# QUATERNION UTILITIES
# ═══════════════════════════════════════════════════════════════

def quat_to_mat(q):
    """PyChrono ChQuaterniond → 3×3 numpy rotation matrix."""
    e0, e1, e2, e3 = q.e0, q.e1, q.e2, q.e3
    return np.array([
        [1-2*(e2*e2+e3*e3), 2*(e1*e2-e0*e3),   2*(e1*e3+e0*e2)],
        [2*(e1*e2+e0*e3),   1-2*(e1*e1+e3*e3), 2*(e2*e3-e0*e1)],
        [2*(e1*e3-e0*e2),   2*(e2*e3+e0*e1),   1-2*(e1*e1+e2*e2)]
    ])

def euler_from_quat(q):
    """Extract roll-pitch-yaw (XYZ intrinsic) from PyChrono quaternion, degrees."""
    R = quat_to_mat(q)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    else:
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# ═══════════════════════════════════════════════════════════════
# BUOYANCY COMPUTATION
# ═══════════════════════════════════════════════════════════════

class BuoyancyComputer:
    """Precomputes hull points; evaluates buoyancy force each timestep."""

    def __init__(self, pts_body, wts, cg_body):
        self.pts_cg = pts_body - cg_body      # points centred on CG
        self.wts    = wts
        self.V_tot  = float(np.sum(wts))

    def compute(self, pos_world, rot_mat, z_water=0.0):
        """Return (F_buoy_world (3,), cob_world (3,), V_sub)."""
        # Transform to world
        pts_w = self.pts_cg @ rot_mat.T
        pts_w[:, 0] += pos_world[0]
        pts_w[:, 1] += pos_world[1]
        pts_w[:, 2] += pos_world[2]

        mask = pts_w[:, 2] <= z_water
        V_sub = float(np.sum(self.wts[mask]))

        if V_sub < 1e-8:
            return np.zeros(3), np.array(pos_world), 0.0

        cob = np.sum(pts_w[mask] * self.wts[mask, None], axis=0) / V_sub
        F = np.array([0.0, 0.0, RHO_WATER * G_ACC * V_sub])
        return F, cob, V_sub


# ═══════════════════════════════════════════════════════════════
# SINGLE SIMULATION RUN
# ═══════════════════════════════════════════════════════════════

def run_sim(label, initial_heel_deg, include_airbag, enable_waves=False):
    """Run one dynamic simulation.  Returns log dict."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  heel0={initial_heel_deg}°  airbag={'ON' if include_airbag else 'OFF'}"
          f"  waves={'ON' if enable_waves else 'OFF'}")
    print(f"{'─'*60}")

    M, cg_body, Ixx, Iyy, Izz = compute_mass_props()

    # Point cloud
    pts, wts = make_body_points(include_airbag)
    buoy = BuoyancyComputer(pts, wts, cg_body)
    print(f"  Points: {len(wts):,}   V_total={buoy.V_tot:.3f} m³")

    # ── Chrono system ──
    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -G_ACC))
    system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    sv = chrono.CastToChIterativeSolverVI(system.GetSolver())
    sv.SetMaxIterations(200)
    sv.SetTolerance(1e-10)

    # Ground (fixed reference)
    gnd = chrono.ChBody()
    gnd.SetFixed(True)
    gnd.SetMass(1e6)
    system.AddBody(gnd)

    # Capsule body
    cap = chrono.ChBody()
    cap.SetName("capsule")
    cap.SetMass(M)
    cap.SetInertiaXX(chrono.ChVector3d(Ixx, Iyy, Izz))

    # Initial position: CG above waterline (use hydrostatics equilibrium as hint)
    h_eq = 0.56 if not include_airbag else 1.35       # from hydrostatics
    cap.SetPos(chrono.ChVector3d(0, 0, h_eq))

    # Initial orientation
    phi0 = math.radians(initial_heel_deg)
    cap.SetRot(chrono.QuatFromAngleX(phi0))

    acc = cap.AddAccumulator()
    system.AddBody(cap)

    # ── Simulation ──
    N_steps = int(SIM_T / DT)
    N_log   = max(1, int(LOG_DT / DT))
    N_print = max(1, int(2.0 / DT))

    log = {'t': [], 'roll': [], 'pitch': [], 'yaw': [],
           'z': [], 'V_sub': [], 'Fz': []}

    print(f"  Running {SIM_T}s at {1/DT:.0f} Hz ({N_steps} steps) …")
    t0_wall = time.time()

    for step in range(N_steps):
        t = step * DT

        # Wave surface
        z_water = 0.0
        if enable_waves:
            z_water = (WAVE_H / 2) * math.sin(2 * math.pi * t / WAVE_T)

        # Clear accumulator
        cap.EmptyAccumulator(acc)

        # Current state
        pos  = cap.GetPos()
        quat = cap.GetRot()
        Rmat = quat_to_mat(quat)
        pos_np = np.array([pos.x, pos.y, pos.z])

        # Buoyancy
        F_buoy, cob, V_sub = buoy.compute(pos_np, Rmat, z_water)

        if V_sub > 1e-6:
            # Clamp buoyancy to 3× weight to prevent force spikes
            F_max = 3.0 * M * G_ACC
            Fz = min(float(F_buoy[2]), F_max)
            cap.AccumulateForce(
                acc,
                chrono.ChVector3d(0.0, 0.0, Fz),
                chrono.ChVector3d(float(cob[0]), float(cob[1]), float(cob[2])),
                False)   # point in world frame
        else:
            # Body out of water: gentle restoring spring toward waterline
            eq_z = h_eq * 0.5   # pull back toward approximate waterline
            Fz_ret = -K_SURF * (pos.z - eq_z)
            cap.AccumulateForce(acc, chrono.ChVector3d(0, 0, Fz_ret),
                                chrono.ChVector3d(0, 0, 0), True)

        # Linear drag (proportional to submerged fraction for realism)
        frac_sub = min(1.0, V_sub / buoy.V_tot) if buoy.V_tot > 0 else 0.0
        cd_eff = CD_LIN * max(0.1, frac_sub)
        v = cap.GetPosDt()
        cap.AccumulateForce(
            acc,
            chrono.ChVector3d(-cd_eff * v.x, -cd_eff * v.y, -cd_eff * v.z),
            chrono.ChVector3d(0, 0, 0), True)

        # Angular drag via force couples
        cd_ang_eff = CD_ANG * max(0.1, frac_sub)
        w = cap.GetAngVelParent()
        arm = 1.0
        tx = -cd_ang_eff * w.x / (2 * arm)
        ty = -cd_ang_eff * w.y / (2 * arm)
        tz = -cd_ang_eff * w.z / (2 * arm)
        cap.AccumulateForce(acc, chrono.ChVector3d(0, 0, tx),
                            chrono.ChVector3d(0, arm, 0), True)
        cap.AccumulateForce(acc, chrono.ChVector3d(0, 0, -tx),
                            chrono.ChVector3d(0, -arm, 0), True)
        cap.AccumulateForce(acc, chrono.ChVector3d(0, 0, -ty),
                            chrono.ChVector3d(arm, 0, 0), True)
        cap.AccumulateForce(acc, chrono.ChVector3d(0, 0, ty),
                            chrono.ChVector3d(-arm, 0, 0), True)
        cap.AccumulateForce(acc, chrono.ChVector3d(tz, 0, 0),
                            chrono.ChVector3d(0, arm, 0), True)
        cap.AccumulateForce(acc, chrono.ChVector3d(-tz, 0, 0),
                            chrono.ChVector3d(0, -arm, 0), True)

        # Step
        system.DoStepDynamics(DT)

        # Divergence check
        pz = cap.GetPos().z
        if math.isnan(pz) or abs(pz) > 20:
            print(f"  *** Divergence at t={t:.3f}s, z={pz}. Stopping.")
            break

        # Log
        if step % N_log == 0:
            roll, pitch, yaw = euler_from_quat(cap.GetRot())
            log['t'].append(t)
            log['roll'].append(roll)
            log['pitch'].append(pitch)
            log['yaw'].append(yaw)
            log['z'].append(cap.GetPos().z)
            log['V_sub'].append(V_sub)
            log['Fz'].append(float(F_buoy[2]))

        if step % N_print == 0 and step > 0:
            roll, _, _ = euler_from_quat(cap.GetRot())
            elapsed = time.time() - t0_wall
            print(f"    t={t:5.1f}s  roll={roll:+7.1f}°  z={cap.GetPos().z:+.3f}m"
                  f"  V_sub={V_sub:.3f}m³  wall={elapsed:.1f}s")

    dt_wall = time.time() - t0_wall
    print(f"  Done in {dt_wall:.1f}s  ({N_steps/dt_wall:.0f} steps/s)")

    # Final state
    if log['t']:
        print(f"  Final roll = {log['roll'][-1]:+.1f}°   z = {log['z'][-1]:+.3f}m")

    return log


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_dynamics(all_logs, path):
    """Multi-panel plot: roll angle vs time for all runs."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels_short = ['Upright', '45° heel', '90° side', '180° inverted']

    for panel, (airbag_label, start_idx) in enumerate(
            [("No Airbag", 0), ("With Airbag", 4)]):
        ax = axes[panel]
        for i in range(4):
            idx = start_idx + i
            if idx >= len(all_logs):
                break
            log = all_logs[idx]
            if not log['t']:
                continue
            ax.plot(log['t'], log['roll'], color=colors[i], lw=1.5,
                    label=labels_short[i])
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel('Roll angle [deg]')
        ax.set_title(f'Dynamic Response — {airbag_label}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-200, 200)

    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved {path}")


def plot_wave_response(wave_logs, path):
    """Roll angle under wave excitation."""
    fig, ax = plt.subplots(figsize=(14, 5))
    styles = [('-', '#1f77b4', 'No Airbag'), ('-', '#ff7f0e', 'With Airbag')]
    for i, log in enumerate(wave_logs):
        if not log['t']:
            continue
        ls, col, lab = styles[i]
        ax.plot(log['t'], log['roll'], ls=ls, color=col, lw=1.5, label=lab)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Roll angle [deg]')
    ax.set_title(f'Wave Response (H={WAVE_H}m, T={WAVE_T}s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("BLOON CAPSULE FLOTATION DYNAMICS — PROJECT CHRONO")
    print("=" * 70)

    base = "/Users/josemarianolopezurdiales/Documents/CAD"
    all_logs = []

    # 8 calm-water runs
    for airbag in [False, True]:
        tag = "AIRBAG" if airbag else "no_bag"
        for heel in [0, 45, 90, 180]:
            label = f"{tag}_{heel}deg"
            log = run_sim(label, heel, airbag, enable_waves=False)
            all_logs.append(log)

    plot_dynamics(all_logs, f"{base}/flotation_chrono_dynamics.png")

    # 2 wave-excitation runs (upright start)
    wave_logs = []
    for airbag in [False, True]:
        tag = "wave_" + ("AIRBAG" if airbag else "no_bag")
        log = run_sim(tag, 0, airbag, enable_waves=True)
        wave_logs.append(log)

    plot_wave_response(wave_logs, f"{base}/flotation_chrono_wave_response.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    labels = []
    for ab in ["No bag", "Airbag"]:
        for h in [0, 45, 90, 180]:
            labels.append(f"{ab} {h}°")
    for i, log in enumerate(all_logs):
        if i < len(labels) and log['t']:
            final_r = log['roll'][-1]
            print(f"  {labels[i]:20s}  final roll = {final_r:+7.1f}°")

    print("\n✓  Flotation dynamics complete.")


if __name__ == "__main__":
    main()
