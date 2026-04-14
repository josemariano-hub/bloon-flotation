#!/usr/bin/env python3
"""
Generate MP4 animations of BLOON capsule flotation scenarios.
Uses matplotlib 3D + ffmpeg. Runs in ~30 seconds.

Videos:
  vid_selfrighting.mp4   - Self-righting from 90° heel
  vid_inverted.mp4       - Inverted tumbling (180° start)
  vid_airbag_waves.mp4   - Airbag riding waves
  vid_beauty_orbit.mp4   - Beauty orbit around floating capsule
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time, os

OUT = "/Users/josemarianolopezurdiales/Documents/CAD"

# ── Geometry (metres) ──
R_CROWN = 0.600; A_O = 0.900; A_I = 0.475; B_VERT = 0.900; Z_CENTER = 1.100
KEEL_TIP = -0.871; KEEL_BASE = 0.200; KEEL_R = 0.600
AB_RMAJ = 1.220; AB_RMIN = 0.420; AB_ZC = -0.180

FPS = 20
DUR = 8  # seconds per video
NFRAMES = FPS * DUR

# ═══════════════════════════════════════════════════════════════
# MESH GENERATION
# ═══════════════════════════════════════════════════════════════

def capsule_mesh(n_theta=48, n_phi=30):
    """Generate capsule torus surface as (X, Y, Z) arrays for plot_surface."""
    # Meridional profile: extrados + intrados
    phi_ext = np.linspace(-np.pi/2, np.pi/2, n_phi)
    phi_int = np.linspace(np.pi/2, -np.pi/2, n_phi)

    r_prof = np.concatenate([
        R_CROWN + A_O * np.cos(phi_ext),
        R_CROWN - A_I * np.cos(phi_int)
    ])
    z_prof = np.concatenate([
        Z_CENTER + B_VERT * np.sin(phi_ext),
        Z_CENTER + B_VERT * np.sin(phi_int)
    ])

    theta = np.linspace(0, 2*np.pi, n_theta)
    T, P = np.meshgrid(theta, np.arange(len(r_prof)))

    R = np.tile(r_prof, (n_theta, 1)).T
    Z = np.tile(z_prof, (n_theta, 1)).T
    X = R * np.cos(T)
    Y = R * np.sin(T)
    return X, Y, Z

def keel_mesh(n_theta=48, n_z=15):
    z_arr = np.linspace(KEEL_TIP, KEEL_BASE, n_z)
    theta = np.linspace(0, 2*np.pi, n_theta)
    T, Zi = np.meshgrid(theta, z_arr)
    frac = (Zi - KEEL_TIP) / (KEEL_BASE - KEEL_TIP)
    R = KEEL_R * frac
    X = R * np.cos(T)
    Y = R * np.sin(T)
    return X, Y, Zi

def airbag_mesh(n_theta=48, n_phi=20):
    theta = np.linspace(0, 2*np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    T, P = np.meshgrid(theta, phi)
    R = AB_RMAJ + AB_RMIN * np.cos(P)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z = AB_ZC + AB_RMIN * np.sin(P)
    return X, Y, Z

def water_plane(sz=4.0, n=2):
    x = np.linspace(-sz, sz, n)
    y = np.linspace(-sz, sz, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    return X, Y, Z

# ═══════════════════════════════════════════════════════════════
# ROTATION
# ═══════════════════════════════════════════════════════════════

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def transform_mesh(X, Y, Z, R_mat, offset):
    """Apply rotation about CG then translate."""
    shape = X.shape
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)  # (3, N)
    pts = R_mat @ pts
    pts[0] += offset[0]; pts[1] += offset[1]; pts[2] += offset[2]
    return pts[0].reshape(shape), pts[1].reshape(shape), pts[2].reshape(shape)

# ═══════════════════════════════════════════════════════════════
# VIDEO RENDERING
# ═══════════════════════════════════════════════════════════════

def make_video(filename, motion_func, include_airbag=False, orbit=False,
               title="", cam_elev=18, cam_azim=-60):
    """Render a video.
    motion_func(t) -> (roll_rad, pitch_rad, yaw_rad, z_offset, z_water)
    """
    print(f"  Rendering {filename} ({NFRAMES} frames @ {FPS}fps) …", end="", flush=True)
    t0 = time.time()

    # Pre-generate meshes (in body frame, origin shifted so CG near 0)
    cg_z = Z_CENTER - 0.06  # CG Z in capsule datum
    Xc, Yc, Zc = capsule_mesh()
    Zc -= cg_z
    Xk, Yk, Zk = keel_mesh()
    Zk -= cg_z
    if include_airbag:
        Xa, Ya, Za = airbag_mesh()
        Za -= cg_z
    Xw, Yw, Zw = water_plane(5.0)

    fig = plt.figure(figsize=(12.8, 7.2), facecolor='#0a1628')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0a1628')

    def draw_frame(frame):
        ax.clear()
        t = frame / FPS

        roll, pitch, yaw, z_off, z_water = motion_func(t)
        R = rot_x(roll)

        # Transform meshes
        off = np.array([0.0, 0.0, z_off])
        xc, yc, zc = transform_mesh(Xc, Yc, Zc, R, off)
        xk, yk, zk = transform_mesh(Xk, Yk, Zk, R, off)

        # Water
        Zw_t = np.full_like(Xw, z_water)
        ax.plot_surface(Xw, Yw, Zw_t, alpha=0.35, color='#1a6b8a', zorder=0)

        # Capsule hull
        ax.plot_surface(xc, yc, zc, alpha=0.85, color='#c8ccd4',
                       edgecolor='#8890a0', linewidth=0.15, zorder=5)
        # Keel
        ax.plot_surface(xk, yk, zk, alpha=0.9, color='#b8956a',
                       edgecolor='#907050', linewidth=0.2, zorder=4)
        # Airbag
        if include_airbag:
            xa, ya, za = transform_mesh(Xa, Ya, Za, R, off)
            ax.plot_surface(xa, ya, za, alpha=0.85, color='#ee6b00',
                           edgecolor='#cc5500', linewidth=0.15, zorder=3)

        # Waterline ring on capsule (visual aid)
        th = np.linspace(0, 2*np.pi, 100)
        r_wl = 1.2
        ax.plot(r_wl*np.cos(th), r_wl*np.sin(th),
                np.full(100, z_water), color='cyan', lw=1.5, alpha=0.5, zorder=10)

        # Camera
        if orbit:
            az = cam_azim + t * 30  # 30 deg/s rotation
        else:
            az = cam_azim
        ax.view_init(elev=cam_elev, azim=az)

        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-2.5, 3.5)
        ax.set_axis_off()
        ax.set_title(title, color='white', fontsize=14, fontweight='bold',
                     pad=-10, y=0.98)

        # Time label
        ax.text2D(0.02, 0.02, f"t = {t:.1f} s", transform=ax.transAxes,
                  color='#8090a0', fontsize=11, family='monospace')

    writer = FFMpegWriter(fps=FPS, bitrate=3000,
                          extra_args=['-pix_fmt', 'yuv420p'])
    anim = FuncAnimation(fig, draw_frame, frames=NFRAMES, interval=1000/FPS)
    anim.save(f"{OUT}/{filename}", writer=writer)
    plt.close(fig)
    dt = time.time() - t0
    print(f" done ({dt:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# MOTION PROFILES
# ═══════════════════════════════════════════════════════════════

def motion_selfrighting(t):
    """Start at 90°, damped oscillation to 0°."""
    tau = 3.5
    omega = 2 * np.pi / 4.5
    roll = np.radians(90) * np.exp(-t / tau) * np.cos(omega * t)
    z = 0.56 + 0.08 * np.exp(-t / 2) * np.sin(omega * t)
    return roll, 0, 0, z, 0.0

def motion_inverted(t):
    """Start at 180°, slow drift then tumble."""
    if t < 4:
        roll = np.pi - np.radians(3) * t + np.radians(5) * np.sin(2 * np.pi * t / 3)
        z = 0.5 + 0.05 * np.sin(2 * np.pi * t / 2)
    elif t < 5.5:
        # Rapid tumble
        progress = (t - 4) / 1.5
        roll = np.pi - progress * np.pi * 1.2
        z = 0.5 + 0.3 * progress
    else:
        # Oscillating near upright
        tt = t - 5.5
        roll = np.radians(40) * np.exp(-tt / 2) * np.cos(2 * np.pi * tt / 3)
        z = 0.56 + 0.1 * np.exp(-tt / 2) * np.sin(2 * np.pi * tt / 3)
    return roll, 0, 0, z, 0.0

def motion_airbag_waves(t):
    """Airbag deployed, riding waves. Very stable."""
    roll = np.radians(7 + 2 * np.sin(2 * np.pi * t / 8))
    z_water = 0.5 * np.sin(2 * np.pi * t / 6)
    z = 1.35 + 0.4 * np.sin(2 * np.pi * t / 6)  # capsule heaves with water
    return roll, 0, 0, z, z_water

def motion_beauty(t):
    """Static floating with airbag, camera orbits."""
    roll = np.radians(7)
    return roll, 0, 0, 1.35, 0.0

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING FLOTATION ANIMATION VIDEOS")
    print("=" * 60)

    make_video("vid_selfrighting.mp4", motion_selfrighting,
               include_airbag=False,
               title="Self-Righting from 90\u00b0 (No Airbag)",
               cam_elev=15, cam_azim=-55)

    make_video("vid_inverted.mp4", motion_inverted,
               include_airbag=False,
               title="Inverted Start (180\u00b0) \u2014 Tumbling Recovery",
               cam_elev=12, cam_azim=-65)

    make_video("vid_airbag_waves.mp4", motion_airbag_waves,
               include_airbag=True,
               title="Airbag Deployed \u2014 Wave Response (H=1m, T=6s)",
               cam_elev=18, cam_azim=-50)

    make_video("vid_beauty_orbit.mp4", motion_beauty,
               include_airbag=True, orbit=True,
               title="BLOON Capsule \u2014 Flotation with Airbag",
               cam_elev=20, cam_azim=0)

    print("\n\u2713  All videos generated.")
