"""
Bloon Capsule Flotation Visualization — Blender 5.0
====================================================
Renders the capsule floating in seawater at equilibrium states
from the hydrostatic analysis.

Run:  /Applications/Blender.app/Contents/MacOS/Blender --background --python flotation_blender_viz.py

Scenes rendered:
  1. Upright, no airbag
  2. Upright, with airbag
  3. 45° heel, no airbag
  4. Inverted (180°), no airbag
  5. Composite grid

Author: Claude / Z2I CAD project
Date:   2026-04-13
"""

import bpy
import bmesh
import math
import numpy as np
from mathutils import Vector, Euler

OUT = "/Users/josemarianolopezurdiales/Documents/CAD"

# ═══════════════════════════════════════════════════════════════
# GEOMETRY PARAMETERS (same as hydrostatics, metres)
# ═══════════════════════════════════════════════════════════════
R_CROWN  = 0.600;  A_O = 0.900;  A_I = 0.475;  B_VERT = 0.900
Z_CENTER = 1.100
KEEL_TIP_Z  = -0.871;  KEEL_BASE_Z = 0.200;  KEEL_BASE_R = 0.600
AB_R_MAJ = 1.220;  AB_R_MIN = 0.420;  AB_Z_CTR = -0.180

# Equilibrium data from hydrostatics (CG height above waterline)
# Case 1 (no bag): h_cg = 0.562, CG body z = -0.059
# Case 2 (airbag): h_cg = 1.346, CG body z = -0.059
EQ_NO_BAG  = {'h_cg': 0.562, 'cg_z': -0.059}
EQ_AIRBAG  = {'h_cg': 1.346, 'cg_z': -0.059}

# ═══════════════════════════════════════════════════════════════
# MESH BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_capsule_mesh(name="Capsule"):
    """Create torus-shaped capsule hull + keel cone mesh."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    n_theta = 72     # azimuthal segments
    n_phi   = 60     # poloidal segments per half-profile

    # Generate meridional profile: extrados then intrados (closed loop)
    profile_r = []
    profile_z = []

    # Extrados: phi from -pi/2 to +pi/2
    for i in range(n_phi + 1):
        phi = -math.pi/2 + math.pi * i / n_phi
        r = R_CROWN + A_O * math.cos(phi)
        z = Z_CENTER + B_VERT * math.sin(phi)
        profile_r.append(r)
        profile_z.append(z)

    # Intrados: phi from +pi/2 to -pi/2 (reversed, closing the profile)
    for i in range(1, n_phi):
        phi = math.pi/2 - math.pi * i / n_phi
        r = R_CROWN - A_I * math.cos(phi)
        z = Z_CENTER + B_VERT * math.sin(phi)
        profile_r.append(r)
        profile_z.append(z)

    n_prof = len(profile_r)

    # Revolve profile around Z-axis
    verts = []
    for j in range(n_theta):
        theta = 2 * math.pi * j / n_theta
        ct, st = math.cos(theta), math.sin(theta)
        for i in range(n_prof):
            r = profile_r[i]
            x = r * ct
            y = r * st
            z = profile_z[i]
            v = bm.verts.new((x, y, z))
            verts.append(v)

    bm.verts.ensure_lookup_table()

    # Create faces (quads connecting adjacent rings)
    for j in range(n_theta):
        j_next = (j + 1) % n_theta
        for i in range(n_prof - 1):
            v00 = verts[j * n_prof + i]
            v01 = verts[j * n_prof + i + 1]
            v10 = verts[j_next * n_prof + i]
            v11 = verts[j_next * n_prof + i + 1]
            bm.faces.new([v00, v01, v11, v10])

    # Close the profile loop (last point connects to first)
    for j in range(n_theta):
        j_next = (j + 1) % n_theta
        v00 = verts[j * n_prof + n_prof - 1]
        v01 = verts[j * n_prof + 0]
        v10 = verts[j_next * n_prof + n_prof - 1]
        v11 = verts[j_next * n_prof + 0]
        bm.faces.new([v00, v01, v11, v10])

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def build_keel_mesh(name="Keel"):
    """Create keel cone mesh."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    n_theta = 72
    n_z = 30
    z_arr = np.linspace(KEEL_TIP_Z, KEEL_BASE_Z, n_z)

    verts_by_ring = []
    for z in z_arr:
        frac = (z - KEEL_TIP_Z) / (KEEL_BASE_Z - KEEL_TIP_Z)
        r = KEEL_BASE_R * frac
        ring = []
        if r < 0.001:
            # Tip: single vertex
            v = bm.verts.new((0, 0, z))
            ring.append(v)
        else:
            for j in range(n_theta):
                theta = 2 * math.pi * j / n_theta
                v = bm.verts.new((r * math.cos(theta), r * math.sin(theta), z))
                ring.append(v)
        verts_by_ring.append(ring)

    bm.verts.ensure_lookup_table()

    # Connect rings with faces
    for k in range(len(verts_by_ring) - 1):
        ring_lo = verts_by_ring[k]
        ring_hi = verts_by_ring[k + 1]

        if len(ring_lo) == 1:
            # Fan from tip vertex to first ring
            tip = ring_lo[0]
            for j in range(len(ring_hi)):
                j_next = (j + 1) % len(ring_hi)
                bm.faces.new([tip, ring_hi[j], ring_hi[j_next]])
        else:
            for j in range(n_theta):
                j_next = (j + 1) % n_theta
                bm.faces.new([ring_lo[j], ring_lo[j_next],
                              ring_hi[j_next], ring_hi[j]])

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def build_airbag_mesh(name="Airbag"):
    """Create airbag torus mesh using Blender primitive."""
    bpy.ops.mesh.primitive_torus_add(
        align='WORLD',
        location=(0, 0, AB_Z_CTR),
        major_radius=AB_R_MAJ,
        minor_radius=AB_R_MIN,
        major_segments=72,
        minor_segments=36)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def build_ocean(name="Ocean", size=20):
    """Create ocean plane at z=0."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    return obj


# ═══════════════════════════════════════════════════════════════
# MATERIALS
# ═══════════════════════════════════════════════════════════════

def mat_capsule():
    mat = bpy.data.materials.new("CapsuleShell")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.82, 1)
    bsdf.inputs["Metallic"].default_value = 0.7
    bsdf.inputs["Roughness"].default_value = 0.25
    return mat

def mat_keel():
    mat = bpy.data.materials.new("KeelFoam")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.76, 0.65, 0.45, 1)
    bsdf.inputs["Roughness"].default_value = 0.8
    return mat

def mat_airbag():
    mat = bpy.data.materials.new("AirbagOrange")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    # RAL 2004 Pure Orange ≈ (0.93, 0.42, 0.0)
    bsdf.inputs["Base Color"].default_value = (0.93, 0.42, 0.0, 1)
    bsdf.inputs["Roughness"].default_value = 0.5
    return mat

def mat_ocean():
    mat = bpy.data.materials.new("OceanWater")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.05, 0.25, 0.35, 1)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.1
    bsdf.inputs["Alpha"].default_value = 0.7
    mat.blend_method = 'BLEND' if hasattr(mat, 'blend_method') else None
    return mat


# ═══════════════════════════════════════════════════════════════
# SCENE SETUP
# ═══════════════════════════════════════════════════════════════

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # Remove orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

def setup_lighting():
    # Key light (sun)
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(45), 0, math.radians(45))

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-4, 4, 6))
    fill = bpy.context.active_object
    fill.data.energy = 200
    fill.data.size = 5

    # World background
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.6, 0.75, 0.9, 1)
        bg.inputs["Strength"].default_value = 0.8


def setup_camera(location, target=(0, 0, 0)):
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.active_object
    cam.name = "RenderCam"

    # Point at target
    direction = Vector(target) - Vector(location)
    rot = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot.to_euler()

    bpy.context.scene.camera = cam
    return cam


def setup_render(width=1920, height=1080):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'


# ═══════════════════════════════════════════════════════════════
# SCENE BUILDERS
# ═══════════════════════════════════════════════════════════════

def position_capsule(capsule_obj, keel_obj, airbag_obj, eq_data, heel_deg=0):
    """Position capsule at hydrostatic equilibrium.

    The capsule is built in capsule-datum coords (Z=0 at ground).
    For flotation: waterline at z_world=0, CG at z_world=h_cg.
    Offset = h_cg - (Z_CENTER + cg_z)  moves torus centre so CG is at h_cg.
    """
    h_cg = eq_data['h_cg']
    cg_z = eq_data['cg_z']
    # Torus centre needs to be at z_world = h_cg - cg_z
    z_offset = h_cg - cg_z - Z_CENTER

    # Group objects
    objs = [capsule_obj, keel_obj]
    if airbag_obj:
        objs.append(airbag_obj)

    for obj in objs:
        obj.location.z += z_offset

    if heel_deg != 0:
        # Rotate about X-axis through CG
        pivot = Vector((0, 0, h_cg))
        angle = math.radians(heel_deg)

        for obj in objs:
            # Translate so pivot is at origin
            obj.location -= pivot
            # Rotate
            rot = Euler((angle, 0, 0))
            obj.location.rotate(rot)
            obj.rotation_euler.x += angle
            # Translate back
            obj.location += pivot


def build_scene(include_airbag=False, heel_deg=0):
    """Build a complete scene and return the capsule objects."""
    eq = EQ_AIRBAG if include_airbag else EQ_NO_BAG

    # Hull
    capsule = build_capsule_mesh("Capsule")
    capsule.data.materials.append(mat_capsule())
    capsule.modifiers.new("Smooth", 'SUBSURF')
    capsule.modifiers["Smooth"].levels = 1

    # Keel
    keel = build_keel_mesh("Keel")
    keel.data.materials.append(mat_keel())

    # Airbag
    airbag = None
    if include_airbag:
        airbag = build_airbag_mesh("Airbag")
        airbag.data.materials.append(mat_airbag())

    # Position at equilibrium
    position_capsule(capsule, keel, airbag, eq, heel_deg)

    # Ocean
    ocean = build_ocean("Ocean", size=20)
    ocean.data.materials.append(mat_ocean())

    return capsule, keel, airbag, ocean


# ═══════════════════════════════════════════════════════════════
# RENDER SCENES
# ═══════════════════════════════════════════════════════════════

def render_scene(filename, cam_loc, cam_target=(0, 0, 0.3)):
    setup_lighting()
    setup_camera(cam_loc, cam_target)
    setup_render(1920, 1080)

    filepath = f"{OUT}/{filename}"
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    print(f"  Rendered: {filepath}")


def main():
    print("=" * 60)
    print("BLOON CAPSULE FLOTATION VISUALIZATION")
    print("=" * 60)

    # ── Scene 1: Upright, no airbag ──
    print("\n  Scene 1: Upright, no airbag")
    clear_scene()
    build_scene(include_airbag=False, heel_deg=0)
    render_scene("flotation_viz_upright.png",
                 cam_loc=(4.5, -3.5, 1.5), cam_target=(0, 0, 0.5))

    # ── Scene 2: Upright, with airbag ──
    print("\n  Scene 2: Upright, with airbag")
    clear_scene()
    build_scene(include_airbag=True, heel_deg=0)
    render_scene("flotation_viz_upright_airbag.png",
                 cam_loc=(4.5, -3.5, 1.5), cam_target=(0, 0, 0.8))

    # ── Scene 3: 45° heel, no airbag ──
    print("\n  Scene 3: 45 deg heel, no airbag")
    clear_scene()
    build_scene(include_airbag=False, heel_deg=45)
    render_scene("flotation_viz_heeled_45.png",
                 cam_loc=(4.5, -3.5, 1.5), cam_target=(0, 0, 0.5))

    # ── Scene 4: Inverted (180°), no airbag ──
    print("\n  Scene 4: Inverted, no airbag")
    clear_scene()
    build_scene(include_airbag=False, heel_deg=180)
    render_scene("flotation_viz_inverted.png",
                 cam_loc=(4.5, -3.5, 1.5), cam_target=(0, 0, 0.5))

    # ── Scene 5: Side view comparison ──
    print("\n  Scene 5: Side view, with airbag")
    clear_scene()
    build_scene(include_airbag=True, heel_deg=0)
    render_scene("flotation_viz_side_airbag.png",
                 cam_loc=(5.0, 0, 0.5), cam_target=(0, 0, 0.5))

    print("\n  All renders complete.")


if __name__ == "__main__":
    main()
