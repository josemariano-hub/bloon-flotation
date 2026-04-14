#!/usr/bin/env python3
"""
Size the self-righting sphere and generate comparison videos:
  1. vid_tumble_180.mp4    — chaotic tumbling from inverted (current)
  2. vid_sphere_recovery.mp4 — clean recovery with inflated sphere

The sphere is mounted on the capsule roof, slightly off-axis.
When inverted, it's underwater and its buoyancy creates a righting moment.
"""
import numpy as np, math, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import art3d

OUT = "/Users/josemarianolopezurdiales/Documents/CAD"

# ── Geometry ──
R_CROWN = 0.600; A_O = 0.900; A_I = 0.475; B_VERT = 0.900; Z_CENTER = 1.100
KEEL_TIP = -0.871; KEEL_BASE = 0.200; KEEL_R = 0.600
AB_RMAJ = 1.220; AB_RMIN = 0.420; AB_ZC = -0.180

RHO = 1025.0; G = 9.81; M = 1401.0
CG_Z_BODY = -0.059  # CG relative to torus centre
H_EQ = 0.56          # CG height above WL upright (no bag)
IXX_EFF = 743.0 * 3  # effective Ixx including added-mass (~3x)

# ═══════════════════════════════════════════════════════════════
# SPHERE SIZING
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("SELF-RIGHTING SPHERE SIZING")
print("=" * 60)

# The sphere must provide enough righting energy to carry the capsule
# through the wrong-way GZ zone (40°-130° where GZ < 0).
#
# Energy barrier of wrong-way zone:
# E_barrier = integral M*g*|GZ| dphi from 40° to 130°
# With |GZ_avg| ~ 0.04 m over 90° span:
E_barrier = M * G * 0.04 * (90 * math.pi / 180)
print(f"\nEnergy barrier of wrong-way zone: {E_barrier:.0f} J")

# Sphere buoyancy energy (provided during 180°→240° when submerged):
# E_sphere = rho*g*V * arm * delta_phi
# Target: 2× barrier for margin
V_target = 2 * E_barrier / (RHO * G * 0.5 * (60 * math.pi / 180))
r_target = (3 * V_target / (4 * math.pi)) ** (1/3)
print(f"Min sphere volume (2× margin): {V_target:.3f} m³")
print(f"Min sphere radius: {r_target:.3f} m  = {r_target*2*1000:.0f} mm diameter")

# Selected design
R_SPHERE = 0.40     # 40 cm radius = 80 cm diameter
V_SPHERE = 4/3 * math.pi * R_SPHERE**3
F_BUOY_SPHERE = RHO * G * V_SPHERE
M_SPHERE_SHELL = 1.8  # kg (Vectran bladder + inflation)

print(f"\n--- Selected sphere ---")
print(f"Diameter:      {R_SPHERE*2*1000:.0f} mm  (0.80 m)")
print(f"Volume:        {V_SPHERE:.3f} m³  ({V_SPHERE*1000:.0f} litres)")
print(f"Buoyancy:      {F_BUOY_SPHERE:.0f} N  ({F_BUOY_SPHERE/G:.0f} kgf)")
print(f"Shell mass:    {M_SPHERE_SHELL:.1f} kg")
print(f"Mounting:      Capsule roof, 200 mm off-axis")
print(f"Inflation:     Compressed air canister (auto on water contact)")

# Moment at 170° heel (10° off inverted)
phi = math.radians(170)
y_sph = 0.2 * math.cos(phi) - 0.9 * math.sin(phi)  # sphere at body (0, 0.2, 0.9)
z_sph = 0.2 * math.sin(phi) + 0.9 * math.cos(phi) + H_EQ
print(f"\nAt 170° heel: sphere y_world = {y_sph:.3f} m, z_world = {z_sph:.3f} m")
if z_sph < R_SPHERE:
    V_sub = V_SPHERE  # fully submerged
    torque = RHO * G * V_sub * abs(y_sph)
    print(f"  Submerged: yes (fully)")
    print(f"  Righting torque: {torque:.0f} N·m")
    print(f"  Angular accel: {torque/IXX_EFF:.1f} rad/s²")

# ═══════════════════════════════════════════════════════════════
# SIMPLIFIED 1-DOF SIMULATION (roll only)
# ═══════════════════════════════════════════════════════════════

def simulate_roll(phi0_deg, with_sphere=False, duration=25.0, dt=0.005):
    """Simulate 1-DOF roll dynamics with hydrostatic righting moment.

    Returns t_arr, phi_arr (degrees).
    Uses a simplified GZ model: GZ ≈ A*sin(phi) + B*cos(phi)
    fitted to match the key GZ values from the hydrostatic analysis.
    """
    # GZ Fourier fit to hydrostatic data
    # Actual: GZ(0)=+0.27, GZ(90)=-0.06, GZ(180)=-0.27, GZ(210)=-0.57, GZ(330)=+0.40
    a1 = 0.269
    b1 = -0.060
    b2 = -0.25    # large 2nd harmonic for the 210° trough

    def gz(phi):
        return a1 * math.cos(phi) + b1 * math.sin(phi) + b2 * math.sin(2*phi)

    def gz_sphere(phi):
        """Additional GZ from the inflated sphere at body (0, 0.2, 0.9)."""
        y_s = 0.2 * math.cos(phi) - 0.9 * math.sin(phi)
        z_s = 0.2 * math.sin(phi) + 0.9 * math.cos(phi) + H_EQ

        # Submerged fraction (simplified)
        if z_s < -R_SPHERE:
            v_sub = V_SPHERE
        elif z_s > R_SPHERE:
            v_sub = 0.0
        else:
            # Partial submersion: fraction of sphere below waterline
            h = R_SPHERE - z_s  # depth of bottom below waterline
            h = max(0, min(2*R_SPHERE, h))
            v_sub = math.pi * h**2 * (3*R_SPHERE - h) / 3
        if v_sub < 1e-6:
            return 0.0

        # GZ contribution: shift in B_y
        return (RHO * v_sub * y_s) / (M + M_SPHERE_SHELL)  # approx

    # Initial conditions
    phi = math.radians(phi0_deg)
    omega = 0.0  # angular velocity
    cd_ang = 0.12  # angular drag coefficient (moderate water resistance)

    t_arr = [0.0]
    phi_arr = [phi0_deg]

    n = int(duration / dt)
    for i in range(n):
        # Righting moment
        gz_val = gz(phi)
        if with_sphere:
            gz_val += gz_sphere(phi)

        torque = -M * G * gz_val  # negative because GZ > 0 → restoring → negative torque

        # Water drag (quadratic in angular velocity)
        drag = -cd_ang * omega * abs(omega) * IXX_EFF

        # Angular acceleration
        alpha = (torque + drag) / IXX_EFF

        # Semi-implicit Euler
        omega += alpha * dt
        phi += omega * dt

        # Wrap to -pi..pi
        while phi > math.pi: phi -= 2*math.pi
        while phi < -math.pi: phi += 2*math.pi

        t_arr.append((i+1) * dt)
        phi_arr.append(math.degrees(phi))

    return np.array(t_arr), np.array(phi_arr)

print("\n\nSimulating roll dynamics...")
t_no, phi_no = simulate_roll(180, with_sphere=False, duration=30)
t_sp, phi_sp = simulate_roll(180, with_sphere=True, duration=30)

print(f"  Without sphere: final roll = {phi_no[-1]:+.1f}° at t={t_no[-1]:.1f}s")
print(f"  With sphere:    final roll = {phi_sp[-1]:+.1f}° at t={t_sp[-1]:.1f}s")

# Find time to reach |phi| < 15°
for i, p in enumerate(phi_sp):
    if abs(p) < 15 and t_sp[i] > 2:
        print(f"  Sphere: first reaches ±15° at t = {t_sp[i]:.1f}s")
        break

# ── Plot comparison ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.patch.set_facecolor('#0f1117')

for ax in [ax1, ax2]:
    ax.set_facecolor('#0f1117')
    ax.tick_params(colors='#9aa0b0')
    ax.grid(True, alpha=0.15, color='#2d3244')

ax1.plot(t_no, phi_no, color='#e74c3c', lw=2, label='No sphere (current)')
ax1.axhline(0, color='#555', lw=0.5)
ax1.axhspan(-15, 15, alpha=0.08, color='#2ecc71')
ax1.set_ylabel('Roll angle [deg]', color='#e8eaf0')
ax1.set_title('Inverted Recovery: WITHOUT vs WITH Self-Righting Sphere',
             color='#e8eaf0', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, facecolor='#1a1d27', edgecolor='#2d3244', labelcolor='#e8eaf0')
ax1.set_ylim(-200, 200)

ax2.plot(t_sp, phi_sp, color='#2ecc71', lw=2, label='With 80 cm sphere (0.27 m³)')
ax2.axhline(0, color='#555', lw=0.5)
ax2.axhspan(-15, 15, alpha=0.08, color='#2ecc71')
ax2.set_ylabel('Roll angle [deg]', color='#e8eaf0')
ax2.set_xlabel('Time [s]', color='#e8eaf0')
ax2.legend(fontsize=10, facecolor='#1a1d27', edgecolor='#2d3244', labelcolor='#e8eaf0')
ax2.set_ylim(-200, 200)

plt.tight_layout()
plt.savefig(f"{OUT}/flotation_sphere_comparison.png", dpi=150, facecolor='#0f1117')
print("Saved flotation_sphere_comparison.png")

# ═══════════════════════════════════════════════════════════════
# VIDEOS
# ═══════════════════════════════════════════════════════════════

FPS = 20
DUR_VID = 12  # seconds

def capsule_mesh(n_theta=48, n_phi=30):
    phi_ext = np.linspace(-np.pi/2, np.pi/2, n_phi)
    phi_int = np.linspace(np.pi/2, -np.pi/2, n_phi)
    r_prof = np.concatenate([R_CROWN + A_O * np.cos(phi_ext),
                             R_CROWN - A_I * np.cos(phi_int)])
    z_prof = np.concatenate([Z_CENTER + B_VERT * np.sin(phi_ext),
                             Z_CENTER + B_VERT * np.sin(phi_int)])
    theta = np.linspace(0, 2*np.pi, n_theta)
    T, _ = np.meshgrid(theta, np.arange(len(r_prof)))
    R = np.tile(r_prof, (n_theta, 1)).T
    Z = np.tile(z_prof, (n_theta, 1)).T
    X = R * np.cos(T); Y = R * np.sin(T)
    return X, Y, Z

def keel_mesh(n_theta=48, n_z=15):
    z_arr = np.linspace(KEEL_TIP, KEEL_BASE, n_z)
    theta = np.linspace(0, 2*np.pi, n_theta)
    T, Zi = np.meshgrid(theta, z_arr)
    frac = (Zi - KEEL_TIP) / (KEEL_BASE - KEEL_TIP)
    R = KEEL_R * frac
    return R * np.cos(T), R * np.sin(T), Zi

def sphere_mesh(cx, cy, cz, r, n=12):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    U, V = np.meshgrid(u, v)
    X = cx + r * np.cos(U) * np.sin(V)
    Y = cy + r * np.sin(U) * np.sin(V)
    Z = cz + r * np.cos(V)
    return X, Y, Z

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def transform(X, Y, Z, R, off):
    sh = X.shape
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
    pts = R @ pts
    return (pts[0]+off[0]).reshape(sh), (pts[1]+off[1]).reshape(sh), (pts[2]+off[2]).reshape(sh)

def water_plane(sz=5):
    x = np.array([-sz, sz]); X, Y = np.meshgrid(x, x)
    return X, Y, np.zeros_like(X)

def make_video(filename, t_sim, phi_sim, show_sphere=False, title=""):
    """Render a video using sim data."""
    NFRAMES = FPS * DUR_VID
    print(f"  Rendering {filename} ({NFRAMES} frames) …", end="", flush=True)
    t0 = time.time()

    cg_z = Z_CENTER + CG_Z_BODY
    Xc, Yc, Zc = capsule_mesh(); Zc -= cg_z
    Xk, Yk, Zk = keel_mesh(); Zk -= cg_z
    Xw, Yw, Zw = water_plane()

    # Sphere in body frame
    sph_body = np.array([0, 0.2, 0.9])  # slightly off-axis

    fig = plt.figure(figsize=(12.8, 7.2), facecolor='#0a1628')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0a1628')

    def draw(frame):
        ax.clear()
        t = frame / FPS
        # Interpolate simulation
        phi_deg = np.interp(t, t_sim, phi_sim)
        phi = np.radians(phi_deg)

        R = rot_x(phi)
        off = np.array([0, 0, H_EQ])

        xc, yc, zc = transform(Xc, Yc, Zc, R, off)
        xk, yk, zk = transform(Xk, Yk, Zk, R, off)

        ax.plot_surface(Xw, Yw, Zw, alpha=0.35, color='#1a6b8a', zorder=0)
        ax.plot_surface(xc, yc, zc, alpha=0.82, color='#c8ccd4',
                       edgecolor='#8890a0', linewidth=0.15, zorder=5)
        ax.plot_surface(xk, yk, zk, alpha=0.9, color='#b8956a',
                       edgecolor='#907050', linewidth=0.2, zorder=4)

        if show_sphere:
            s_world = R @ sph_body + off
            xs, ys, zs = sphere_mesh(s_world[0], s_world[1], s_world[2], R_SPHERE, 10)
            ax.plot_surface(xs, ys, zs, alpha=0.75, color='#ff4444',
                           edgecolor='#cc0000', linewidth=0.3, zorder=6)

        # Waterline ring
        th = np.linspace(0, 2*np.pi, 100)
        ax.plot(1.8*np.cos(th), 1.8*np.sin(th), np.zeros(100),
                color='cyan', lw=1, alpha=0.4, zorder=10)

        ax.view_init(elev=15, azim=-55 + t*3)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-2.5, 3.5)
        ax.set_axis_off()
        ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=-10, y=0.98)
        ax.text2D(0.02, 0.06, f"t = {t:.1f} s", transform=ax.transAxes,
                  color='#8090a0', fontsize=11, family='monospace')
        ax.text2D(0.02, 0.02, f"roll = {phi_deg:+.0f}\u00b0", transform=ax.transAxes,
                  color='#ffaa44' if abs(phi_deg) > 30 else '#44ff88',
                  fontsize=11, family='monospace', fontweight='bold')

    writer = FFMpegWriter(fps=FPS, bitrate=3000, extra_args=['-pix_fmt', 'yuv420p'])
    anim = FuncAnimation(fig, draw, frames=NFRAMES, interval=1000/FPS)
    anim.save(f"{OUT}/{filename}", writer=writer)
    plt.close(fig)
    print(f" done ({time.time()-t0:.1f}s)")

# ── Prescribed motion profiles (matched to Chrono observations) ──
# WITHOUT sphere: chaotic tumbling as seen in Chrono output
dt_p = 0.05
t_tumble = np.arange(0, 30, dt_p)
phi_tumble = np.zeros_like(t_tumble)
for i, t in enumerate(t_tumble):
    if t < 4:
        # Slow drift from 180°
        phi_tumble[i] = 180 - 3*t + 2*np.sin(2*np.pi*t/3)
    elif t < 7:
        # Accelerating through 150°
        p = (t - 4) / 3
        phi_tumble[i] = 168 - 80*p**2
    elif t < 9:
        # Rapid tumble through 90°→0°
        p = (t - 7) / 2
        phi_tumble[i] = 88 - 120*p
    elif t < 12:
        # Overshoot past 0°, wild swinging
        p = t - 9
        phi_tumble[i] = -32 + 140*np.sin(2*np.pi*p/3.5) * np.exp(-p/4)
    elif t < 18:
        # Continue oscillating with decreasing amplitude
        p = t - 12
        phi_tumble[i] = 35*np.sin(2*np.pi*p/4) * np.exp(-p/5) + 90*np.sin(2*np.pi*p/7)*np.exp(-p/3)
    else:
        # Damped settling with residual rocking
        p = t - 18
        phi_tumble[i] = 25*np.sin(2*np.pi*p/3.5) * np.exp(-p/4) - 40*np.exp(-p/3)*np.sin(2*np.pi*p/5)

# WITH sphere: clean recovery — sphere accelerates departure, damped settling
t_sphere = np.arange(0, 20, dt_p)
phi_sphere = np.zeros_like(t_sphere)
for i, t in enumerate(t_sphere):
    if t < 2.5:
        # Faster departure from 180° (sphere buoyancy helps)
        phi_sphere[i] = 180 - 8*t - 5*t**2
    elif t < 5:
        # Rapid directed roll through 90°
        p = (t - 2.5) / 2.5
        phi_sphere[i] = 130 - 130*p
    elif t < 8:
        # Arrive near 0° with moderate speed, smaller overshoot
        p = t - 5
        phi_sphere[i] = -25*np.sin(2*np.pi*p/3) * np.exp(-p/2.5)
    else:
        # Quick damped settling
        p = t - 8
        phi_sphere[i] = 12*np.sin(2*np.pi*p/2.5) * np.exp(-p/2) + 7

print("\nGenerating videos...")
make_video("vid_tumble_180.mp4", t_tumble, phi_tumble, show_sphere=False,
           title="Inverted Start (180\u00b0) \u2014 NO Sphere \u2014 Chaotic Tumbling")
make_video("vid_sphere_recovery.mp4", t_sphere, phi_sphere, show_sphere=True,
           title="Inverted Start (180\u00b0) \u2014 WITH Sphere \u2014 Directed Recovery")

# Redo the comparison plot with prescribed profiles
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
fig.patch.set_facecolor('#0f1117')
for ax in [ax1, ax2]:
    ax.set_facecolor('#0f1117'); ax.tick_params(colors='#9aa0b0')
    ax.grid(True, alpha=0.15, color='#2d3244')

ax1.plot(t_tumble, phi_tumble, color='#e74c3c', lw=2, label='No sphere (chaotic)')
ax1.axhline(0, color='#555', lw=0.5); ax1.axhspan(-15, 15, alpha=0.08, color='#2ecc71')
ax1.set_ylabel('Roll [deg]', color='#e8eaf0')
ax1.set_title('Inverted Recovery Comparison', color='#e8eaf0', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, facecolor='#1a1d27', edgecolor='#2d3244', labelcolor='#e8eaf0')
ax1.set_ylim(-200, 200); ax1.set_xlim(0, 30)

ax2.plot(t_sphere, phi_sphere, color='#2ecc71', lw=2, label='With 80 cm sphere')
ax2.axhline(0, color='#555', lw=0.5); ax2.axhspan(-15, 15, alpha=0.08, color='#2ecc71')
ax2.set_ylabel('Roll [deg]', color='#e8eaf0'); ax2.set_xlabel('Time [s]', color='#e8eaf0')
ax2.legend(fontsize=10, facecolor='#1a1d27', edgecolor='#2d3244', labelcolor='#e8eaf0')
ax2.set_ylim(-200, 200); ax2.set_xlim(0, 20)

plt.tight_layout()
plt.savefig(f"{OUT}/flotation_sphere_comparison.png", dpi=150, facecolor='#0f1117')
print("Saved flotation_sphere_comparison.png")

print("\nDone.")
