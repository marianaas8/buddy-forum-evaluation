import bpy
import bmesh
import json
import sys
import os
import math
from mathutils import Vector
from mathutils.bvhtree import BVHTree

PROP_TARGET_MAX_DIM = 0.25   # Three.js normalises each prop so its longest axis = 0.25 world units
# During the throwing gesture the prop is at its peak (hold phase = full size, multiplier = 1.0).
# The 0.18 multiplier only applies at the very start of the pop-in, before the prop grows.
SEARCH_PROP_MULTIPLIER = 1.0
CONTACT_D_THRESHOLD = 0.15  # metres — bone within this distance of prop surface = "touching"
MAX_VERTS_FOR_INSIDE_TEST = 200  # cap prop vertices sampled per frame for body-intersect check


def get_armature_and_largest_mesh(objects):
    armatures = [o for o in objects if o.type == 'ARMATURE']
    meshes = sorted([o for o in objects if o.type == 'MESH'],
                    key=lambda m: len(m.data.vertices), reverse=True)
    return (armatures[0] if armatures else None), (meshes[0] if meshes else None)


def build_combined_bvh(mesh_objects):
    """Build a single BVHTree from all mesh objects combined (all parts of Buddy)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bm_combined = bmesh.new()
    for obj in mesh_objects:
        try:
            ev = obj.evaluated_get(depsgraph)
            me = ev.to_mesh()
            bm_part = bmesh.new()
            bm_part.from_mesh(me)
            bm_part.transform(obj.matrix_world)
            bm_combined.from_mesh(ev.to_mesh())   # re-read into combined
            # use bmesh join: copy verts/edges/faces from bm_part into bm_combined
            bm_combined.free()
            bm_combined = bmesh.new()
            break  # fallback — use loop below instead
        except Exception:
            pass
        finally:
            try: ev.to_mesh_clear()
            except Exception: pass

    # Proper implementation: iterate and merge
    bm_combined = bmesh.new()
    for obj in mesh_objects:
        try:
            ev = obj.evaluated_get(depsgraph)
            me = ev.to_mesh()
            bm_part = bmesh.new()
            bm_part.from_mesh(me)
            bm_part.transform(obj.matrix_world)
            # Append all geometry into bm_combined
            vert_map = {}
            for v in bm_part.verts:
                nv = bm_combined.verts.new(v.co)
                vert_map[v.index] = nv
            bm_combined.verts.ensure_lookup_table()
            for f in bm_part.faces:
                try:
                    bm_combined.faces.new([vert_map[v.index] for v in f.verts])
                except Exception:
                    pass
            bm_part.free()
            ev.to_mesh_clear()
        except Exception:
            pass

    bvh = BVHTree.FromBMesh(bm_combined)
    bm_combined.free()
    return bvh


def find_searching_action():
    """Return the best searching-animation action from bpy.data.actions.
    Prioritises the loop actions (search_01/02/03) since those contain the
    throwing gestures where prop-body intersection is most likely to occur.
    Falls back to any action containing 'search', then to the longest action.
    """
    if not bpy.data.actions:
        return None
    # Prefer the specific loop actions by exact name
    for loop_name in ["search_01", "search_02", "search_03"]:
        for action in bpy.data.actions:
            if action.name.lower() == loop_name:
                return action
    # Fall back: any action whose name contains a search keyword
    for kw in ["search_start", "search_end_forum", "search"]:
        for action in bpy.data.actions:
            if kw in action.name.lower():
                return action
    return max(bpy.data.actions, key=lambda a: a.frame_range[1] - a.frame_range[0])


def find_all_loop_actions():
    """Return all three search loop actions (search_01, search_02, search_03)."""
    loops = []
    for loop_name in ["search_01", "search_02", "search_03"]:
        for action in bpy.data.actions:
            if action.name.lower() == loop_name:
                loops.append(action)
                break
    return loops if loops else None


SAMPLE_FRAMES = 30   # number of frames sampled across the animation


def get_sample_frames(start_frame, end_frame):
    """Return the list of frames sampled across the animation (same step used everywhere)."""
    step = max(1, (end_frame - start_frame) // SAMPLE_FRAMES)
    return list(range(start_frame, end_frame + 1, step))


def extract_bone_trajectory(armature_obj, bone_name, start_frame, end_frame):
    """Return list of world-space positions for bone_name sampled across ~30 frames."""
    positions = []
    for f in get_sample_frames(start_frame, end_frame):
        bpy.context.scene.frame_set(f)
        bpy.context.evaluated_depsgraph_get().update()
        pb = armature_obj.pose.bones.get(bone_name)
        if pb:
            positions.append((armature_obj.matrix_world @ pb.matrix).translation.copy())
    return positions


def extract_bone_matrices(armature_obj, bone_name, start_frame, end_frame):
    """Return list of world-space 4x4 matrices for bone_name sampled across ~30 frames."""
    matrices = []
    for f in get_sample_frames(start_frame, end_frame):
        bpy.context.scene.frame_set(f)
        bpy.context.evaluated_depsgraph_get().update()
        pb = armature_obj.pose.bones.get(bone_name)
        if pb:
            matrices.append((armature_obj.matrix_world @ pb.matrix).copy())
    return matrices


def extract_avatar_bvh_per_frame(all_avatar_mesh_objs, start_frame, end_frame):
    """Build a combined BVHTree from ALL Buddy mesh parts at each sampled frame."""
    bvh_list = []
    for f in get_sample_frames(start_frame, end_frame):
        bpy.context.scene.frame_set(f)
        bvh_list.append(build_combined_bvh(all_avatar_mesh_objs))
    return bvh_list


def compute_kinematic_metrics(positions, fps=24):
    """MAD and jitter score from bone world-space trajectory."""
    sentinel = {"mad_score": -1.0, "jitter_score": -1.0,
                "avg_velocity_l2": -1.0, "avg_acceleration_l2": -1.0}
    if len(positions) < 3:
        return sentinel
    dt = 1.0 / fps
    velocities = [(positions[i] - positions[i - 1]) / dt for i in range(1, len(positions))]
    accelerations = [(velocities[i] - velocities[i - 1]) / dt for i in range(1, len(velocities))]
    avg_vel = sum(v.length for v in velocities) / len(velocities)
    if not accelerations:
        return {"mad_score": 0.0, "jitter_score": 0.0,
                "avg_velocity_l2": round(avg_vel, 4), "avg_acceleration_l2": 0.0}
    avg_acc = sum(a.length for a in accelerations) / len(accelerations)
    acc_mags = [a.length for a in accelerations]
    mad = (sum(abs(acc_mags[i] - acc_mags[i - 1]) for i in range(1, len(acc_mags)))
           / (len(acc_mags) - 1)) if len(acc_mags) > 1 else 0.0
    mean_acc = sum(acc_mags) / len(acc_mags)
    jitter = math.sqrt(sum((a - mean_acc) ** 2 for a in acc_mags) / len(acc_mags))
    return {
        "mad_score": round(mad, 4),
        "jitter_score": round(jitter, 4),
        "avg_velocity_l2": round(avg_vel, 4),
        "avg_acceleration_l2": round(avg_acc, 4),
    }


def compute_hoi_metrics(armature_obj, avatar_mesh_obj, bone_positions, prop_obj, prop_root_pos,
                        hand_positions_list=None, avatar_bvh_per_frame=None,
                        bone_matrices=None, mid_bone_matrix=None,
                        all_avatar_meshes=None, loop_eval_data=None):
    """
    HOI metrics between Buddy and a prop placed at DEF-Prop_0.

    prop_root_pos       — world-space position where the prop root was placed (mid_bone).

    hand_positions_list — optional list of per-hand position lists (e.g. [left_traj, right_traj]).
                          When provided, D2O / L2P / Contact Rate are computed using, for each
                          frame, the hand that is closest to the prop (minimum across all hands).
                          Bone Clearance always uses bone_positions (DEF-Prop_0) regardless.

    d2o_mean/min      — dist from nearest-hand trajectory to prop surface (prop at mid_bone)
    l2p_mean/min      — dist from nearest-hand trajectory to prop bbox center
    contact_rate      — % of frames where nearest hand is within CONTACT_D_THRESHOLD of prop surface
    bone_to_avatar    — clearance of DEF-Prop_0 (prop attachment bone) from Buddy's body mesh
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # --- Prop world-space geometry (prop placed at prop_root_pos) ---
    prop_eval = prop_obj.evaluated_get(depsgraph)
    prop_mesh = prop_eval.to_mesh()

    bm_prop = bmesh.new()
    bm_prop.from_mesh(prop_mesh)
    bm_prop.transform(prop_obj.matrix_world)

    # Offsets of each prop vertex from the root position — reused at each bone pos
    prop_offsets = [v.co.copy() - prop_root_pos for v in bm_prop.verts]
    if len(prop_offsets) > MAX_VERTS_FOR_INSIDE_TEST:
        step = max(1, len(prop_offsets) // MAX_VERTS_FOR_INSIDE_TEST)
        prop_offsets = prop_offsets[::step][:MAX_VERTS_FOR_INSIDE_TEST]

    prop_bvh = BVHTree.FromBMesh(bm_prop)
    bm_prop.free()
    prop_eval.to_mesh_clear()

    bbox_world = [prop_obj.matrix_world @ Vector(c) for c in prop_obj.bound_box]
    prop_min = Vector((min(c.x for c in bbox_world), min(c.y for c in bbox_world), min(c.z for c in bbox_world)))
    prop_max = Vector((max(c.x for c in bbox_world), max(c.y for c in bbox_world), max(c.z for c in bbox_world)))
    prop_center = (prop_min + prop_max) * 0.5

    # --- Avatar BVH (world space, static — all Buddy mesh parts combined) ---
    avatar_bvh = None
    if all_avatar_meshes:
        avatar_bvh = build_combined_bvh(all_avatar_meshes)
    elif avatar_mesh_obj:
        av_eval = avatar_mesh_obj.evaluated_get(depsgraph)
        av_mesh = av_eval.to_mesh()
        bm_av = bmesh.new()
        bm_av.from_mesh(av_mesh)
        bm_av.transform(avatar_mesh_obj.matrix_world)
        avatar_bvh = BVHTree.FromBMesh(bm_av)
        bm_av.free()
        av_eval.to_mesh_clear()

    # --- Determine per-frame positions to use for D2O / L2P / Contact Rate ---
    # If hand trajectories are provided, use the closest hand per frame.
    # Otherwise fall back to the prop-bone (DEF-Prop_0) trajectory.
    n_frames = len(bone_positions)
    if hand_positions_list and all(len(h) == n_frames for h in hand_positions_list):
        hoi_positions = []
        for i in range(n_frames):
            best_pos = None
            best_d = float('inf')
            for hand_traj in hand_positions_list:
                hp = hand_traj[i]
                hit = prop_bvh.find_nearest(hp)
                if hit[0] is not None:
                    d = (hp - hit[0]).length
                    if d < best_d:
                        best_d = d
                        best_pos = hp
            if best_pos is not None:
                hoi_positions.append(best_pos)
    else:
        hoi_positions = bone_positions

    # --- D2O + Contact Rate: nearest-hand trajectory → prop surface ---
    d2o_vals = []
    for hp in hoi_positions:
        hit = prop_bvh.find_nearest(hp)
        if hit[0] is not None:
            d2o_vals.append((hp - hit[0]).length)
    d2o_mean = round(sum(d2o_vals) / len(d2o_vals), 4) if d2o_vals else -1.0
    d2o_min  = round(min(d2o_vals), 4) if d2o_vals else -1.0
    contact_rate = (round(sum(1 for d in d2o_vals if d < CONTACT_D_THRESHOLD) / len(d2o_vals) * 100, 2)
                    if d2o_vals else 0.0)

    # --- L2P: nearest-hand trajectory → prop center ---
    l2p_vals = [(hp - prop_center).length for hp in hoi_positions]
    l2p_mean = round(sum(l2p_vals) / len(l2p_vals), 4) if l2p_vals else -1.0
    l2p_min  = round(min(l2p_vals), 4) if l2p_vals else -1.0

    # --- Bone-to-avatar clearance (measures how close the prop attachment bone gets to Buddy's body) ---
    bone_to_avatar_min  = -1.0
    bone_to_avatar_mean = -1.0
    if avatar_bvh and bone_positions:
        b2a_vals = []
        for bp in bone_positions:
            hit_pt, _, _, _ = avatar_bvh.find_nearest(bp)
            if hit_pt is not None:
                b2a_vals.append((bp - hit_pt).length)
        if b2a_vals:
            bone_to_avatar_min  = round(min(b2a_vals), 4)
            bone_to_avatar_mean = round(sum(b2a_vals) / len(b2a_vals), 4)

    # --- Prop body intersection rate ---
    # For each sampled frame, use the avatar BVH AT THAT FRAME (animated pose) to check
    # whether any prop vertex — translated to the bone position of that frame — is inside
    # Buddy's body. Uses ray casting (parity test) which is robust for non-watertight meshes.
    RAY_DIR = Vector((1.0, 0.0, 0.0))   # fixed ray direction for parity test

    def _is_inside(bvh, point):
        """Ray-casting parity test: odd number of intersections → inside mesh."""
        count = 0
        origin = point.copy()
        for _ in range(64):   # cap iterations to avoid infinite loop
            loc, _, _, _ = bvh.ray_cast(origin, RAY_DIR)
            if loc is None:
                break
            count += 1
            origin = loc + RAY_DIR * 1e-4   # advance just past the hit surface
        return (count % 2) == 1

    # Convert prop offsets to bone-local space using the midframe bone matrix.
    # At runtime the prop is parented to the bone and inherits full rotation + translation,
    # so we must apply the per-frame bone matrix (not just translation) to get correct positions.
    prop_local_coords = None
    if mid_bone_matrix is not None and prop_offsets:
        try:
            mid_inv = mid_bone_matrix.inverted()
            prop_local_coords = []
            for offset in prop_offsets:
                v_world_mid = prop_root_pos + offset
                v_local = mid_inv @ v_world_mid.to_4d()
                prop_local_coords.append(v_local)
        except Exception:
            prop_local_coords = None

    def _intersect_rate_for_loop(lbm, lbvh):
        """Compute body intersection rate for a single loop action."""
        if not lbm or not lbvh or len(lbm) != len(lbvh):
            return -1.0
        use_local = bool(prop_local_coords and len(lbm) > 0)
        intersect_frames = 0
        for i, (bm_i, fbvh) in enumerate(zip(lbm, lbvh)):
            frame_has_intersection = False
            verts = [(bm_i @ lc).to_3d() for lc in prop_local_coords] if use_local else []
            for v in verts:
                if _is_inside(fbvh, v):
                    frame_has_intersection = True
                    break
            if frame_has_intersection:
                intersect_frames += 1
        return round(intersect_frames / len(lbm) * 100, 2)

    # Compute body intersection per loop (search_01, search_02, search_03)
    loop_rates = []
    if loop_eval_data and prop_local_coords:
        for lbm, lbvh in loop_eval_data:
            r = _intersect_rate_for_loop(lbm, lbvh)
            if r >= 0:
                loop_rates.append(r)

    if loop_rates:
        body_intersect_rate = round(sum(loop_rates) / len(loop_rates), 2)
        body_intersect_loop1 = loop_rates[0] if len(loop_rates) > 0 else -1.0
        body_intersect_loop2 = loop_rates[1] if len(loop_rates) > 1 else -1.0
        body_intersect_loop3 = loop_rates[2] if len(loop_rates) > 2 else -1.0
    else:
        # Fallback to main action if no loop data
        body_intersect_rate = -1.0
        body_intersect_loop1 = body_intersect_loop2 = body_intersect_loop3 = -1.0
        bvh_frames = avatar_bvh_per_frame if avatar_bvh_per_frame else ([avatar_bvh] * len(bone_positions) if avatar_bvh else None)
        if bvh_frames and bone_positions and prop_local_coords and len(bvh_frames) == len(bone_positions):
            intersect_frames = 0
            use_local = bool(bone_matrices and len(bone_matrices) == len(bone_positions))
            for i, (bp, frame_bvh) in enumerate(zip(bone_positions, bvh_frames)):
                frame_has_intersection = False
                verts = [(bone_matrices[i] @ lc).to_3d() for lc in prop_local_coords] if use_local else [bp + off for off in prop_offsets]
                for v in verts:
                    if _is_inside(frame_bvh, v):
                        frame_has_intersection = True
                        break
                if frame_has_intersection:
                    intersect_frames += 1
            body_intersect_rate = round(intersect_frames / len(bone_positions) * 100, 2)

    return {
        "d2o_mean": d2o_mean,
        "d2o_min": d2o_min,
        "l2p_mean": l2p_mean,
        "l2p_min": l2p_min,
        "contact_rate": contact_rate,
        "bone_to_avatar_min":  bone_to_avatar_min,
        "bone_to_avatar_mean": bone_to_avatar_mean,
        "body_intersect_rate": body_intersect_rate,
        "body_intersect_loop1": body_intersect_loop1,
        "body_intersect_loop2": body_intersect_loop2,
        "body_intersect_loop3": body_intersect_loop3,
    }


def evaluate_props(buddy_glb, prop_glb_paths, bone_name, output_json, hand_bone_names=None):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    try:
        bpy.ops.import_scene.gltf(filepath=buddy_glb)
    except Exception as e:
        with open(output_json, 'w') as f:
            json.dump({"error": f"Buddy GLB import failed: {e}"}, f)
        return

    buddy_obj_names = set(o.name for o in bpy.context.scene.objects)
    armature_obj, avatar_mesh_obj = get_armature_and_largest_mesh(list(bpy.context.scene.objects))
    # Collect ALL Buddy mesh parts for combined BVH (body + arms + hands + accessories)
    all_buddy_meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']

    if armature_obj is None:
        with open(output_json, 'w') as f:
            json.dump({"error": "No armature found in buddy GLB."}, f)
        return

    action = find_searching_action()
    if action is None:
        with open(output_json, 'w') as f:
            json.dump({"error": "No animation found in buddy GLB."}, f)
        return

    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    armature_obj.animation_data.action = action

    start_f = int(action.frame_range[0])
    end_f = int(action.frame_range[1])
    bpy.context.scene.frame_start = start_f
    bpy.context.scene.frame_end = end_f

    if armature_obj.pose.bones.get(bone_name) is None:
        for pb in armature_obj.pose.bones:
            if 'prop' in pb.name.lower():
                bone_name = pb.name
                break

    if armature_obj.pose.bones.get(bone_name) is None:
        sample = [b.name for b in armature_obj.pose.bones][:15]
        with open(output_json, 'w') as f:
            json.dump({"error": f"Bone '{bone_name}' not found. Sample bones: {sample}"}, f)
        return

    bone_positions = extract_bone_trajectory(armature_obj, bone_name, start_f, end_f)
    bone_matrices  = extract_bone_matrices(armature_obj, bone_name, start_f, end_f)
    kinematic = compute_kinematic_metrics(bone_positions)

    # Midframe bone matrix — used to convert prop vertices to bone-local space
    mid_idx = len(bone_matrices) // 2
    mid_bone_matrix = bone_matrices[mid_idx] if bone_matrices else None

    # Extract combined avatar BVH (all Buddy mesh parts) at each sampled frame (main action)
    avatar_bvh_per_frame = None
    if all_buddy_meshes:
        avatar_bvh_per_frame = extract_avatar_bvh_per_frame(all_buddy_meshes, start_f, end_f)

    # Pre-compute per-loop data for body intersection (search_01, search_02, search_03)
    # For each loop action: bone matrices + avatar BVHs at that action's frames
    loop_actions = find_all_loop_actions()
    loop_eval_data = []   # list of (bone_matrices, avatar_bvh_per_frame) per loop
    for loop_action in (loop_actions or []):
        armature_obj.animation_data.action = loop_action
        lf_start = int(loop_action.frame_range[0])
        lf_end   = int(loop_action.frame_range[1])
        lbm = extract_bone_matrices(armature_obj, bone_name, lf_start, lf_end)
        lbvh = extract_avatar_bvh_per_frame(all_buddy_meshes, lf_start, lf_end) if all_buddy_meshes else None
        loop_eval_data.append((lbm, lbvh))
    # Restore main action
    armature_obj.animation_data.action = action

    # Extract hand bone trajectories for HOI metrics (D2O / L2P / Contact Rate)
    hand_positions_list = None
    if hand_bone_names:
        hand_trajs = []
        for hbn in hand_bone_names:
            if armature_obj.pose.bones.get(hbn):
                traj = extract_bone_trajectory(armature_obj, hbn, start_f, end_f)
                if len(traj) == len(bone_positions):
                    hand_trajs.append(traj)
        if hand_trajs:
            hand_positions_list = hand_trajs

    mid_f = start_f + (end_f - start_f) // 2
    mid_bone = bone_positions[len(bone_positions) // 2] if bone_positions else Vector((0, 0, 0))
    prop_results = []

    for prop_path in prop_glb_paths:
        if not os.path.exists(prop_path):
            prop_results.append({"prop_path": prop_path, "error": "File not found"})
            continue

        try:
            bpy.ops.import_scene.gltf(filepath=prop_path)
        except Exception as e:
            prop_results.append({"prop_path": prop_path, "error": f"Import failed: {e}"})
            continue

        new_meshes = [o for o in bpy.context.scene.objects
                      if o.name not in buddy_obj_names and o.type == 'MESH']
        new_roots = [o for o in bpy.context.scene.objects
                     if o.name not in buddy_obj_names and o.parent is None]

        if not new_meshes:
            for o in list(bpy.context.scene.objects):
                if o.name not in buddy_obj_names:
                    bpy.data.objects.remove(o, do_unlink=True)
            prop_results.append({"prop_path": prop_path, "error": "No mesh in prop GLB"})
            continue

        # --- Compute prop scale matching Three.js attachDynamicProp logic ---
        # Three.js: scale = (0.25 / maxDim) / boneWorldScale * 0.18  (for search props)
        # Step 1: measure prop bounding box at unit scale
        bpy.context.view_layer.update()
        from mathutils import Matrix
        temp_mesh = sorted(new_meshes, key=lambda o: len(o.data.vertices), reverse=True)[0]
        bbox_world = [temp_mesh.matrix_world @ Vector(c) for c in temp_mesh.bound_box]
        prop_min = Vector((min(c.x for c in bbox_world), min(c.y for c in bbox_world), min(c.z for c in bbox_world)))
        prop_max = Vector((max(c.x for c in bbox_world), max(c.y for c in bbox_world), max(c.z for c in bbox_world)))
        prop_size = prop_max - prop_min
        max_dim = max(prop_size.x, prop_size.y, prop_size.z)

        # Step 2: compute bone world scale (to cancel it out, as Three.js does)
        prop_bone = armature_obj.pose.bones.get(bone_name)
        bone_ws = (armature_obj.matrix_world @ prop_bone.matrix).to_scale() if prop_bone else Vector((1, 1, 1))

        # Step 3: final scale = (0.25 / maxDim) * (1 / boneWorldScale) * 0.18
        if max_dim > 0:
            norm_scale = PROP_TARGET_MAX_DIM / max_dim
            sx = norm_scale * (1.0 / bone_ws.x if bone_ws.x != 0 else 1) * SEARCH_PROP_MULTIPLIER
            sy = norm_scale * (1.0 / bone_ws.y if bone_ws.y != 0 else 1) * SEARCH_PROP_MULTIPLIER
            sz = norm_scale * (1.0 / bone_ws.z if bone_ws.z != 0 else 1) * SEARCH_PROP_MULTIPLIER
        else:
            sx = sy = sz = SEARCH_PROP_MULTIPLIER

        for root in new_roots:
            root.location = mid_bone
            root.scale = (sx, sy, sz)

        bpy.context.scene.frame_set(mid_f)
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get().update()

        prop_mesh_obj = sorted(new_meshes, key=lambda o: len(o.data.vertices), reverse=True)[0]

        hoi = compute_hoi_metrics(armature_obj, avatar_mesh_obj, bone_positions,
                                   prop_mesh_obj, mid_bone,
                                   hand_positions_list=hand_positions_list,
                                   avatar_bvh_per_frame=avatar_bvh_per_frame,
                                   bone_matrices=bone_matrices,
                                   mid_bone_matrix=mid_bone_matrix,
                                   all_avatar_meshes=all_buddy_meshes,
                                   loop_eval_data=loop_eval_data)

        prop_results.append({"prop_path": os.path.basename(prop_path), **hoi})

        for o in list(bpy.context.scene.objects):
            if o.name not in buddy_obj_names:
                bpy.data.objects.remove(o, do_unlink=True)
        buddy_obj_names = set(o.name for o in bpy.context.scene.objects)

    with open(output_json, 'w') as f:
        json.dump({
            "buddy_glb": os.path.basename(buddy_glb),
            "bone_name": bone_name,
            "action_name": action.name,
            "frame_range": [start_f, end_f],
            "bone_samples": len(bone_positions),
            "kinematic": kinematic,
            "props": prop_results,
        }, f, indent=2)


if __name__ == "__main__":
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1:]
        hand_bone_names = json.loads(args[4]) if len(args) > 4 else None
        evaluate_props(args[0], json.loads(args[1]), args[2], args[3],
                       hand_bone_names=hand_bone_names)
    except (ValueError, IndexError) as e:
        print(f"Usage: blender --background --python blender_prop_eval.py -- "
              f"buddy_glb prop_glb_paths_json bone_name output_json [hand_bone_names_json]")
        print(f"Error: {e}")
