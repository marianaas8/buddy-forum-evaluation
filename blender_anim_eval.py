import bpy
import bmesh
import json
import sys
import os
import math
import random
from mathutils import Vector, bvhtree

# Function to get the evaluated mesh for the current frame, which includes all modifiers and animations applied
def get_evaluated_mesh(obj, depsgraph):
    eval_obj = obj.evaluated_get(depsgraph)
    return eval_obj.to_mesh()

def compute_edge_metrics_tracked(bm, tracked_indices):
    """Calculate edge length statistics using stable edge indices across frames."""
    if not tracked_indices or len(bm.edges) == 0: return 0.0, 0.0
    
    valid_indices = [i for i in tracked_indices if i < len(bm.edges)]
    if not valid_indices: return 0.0, 0.0
    
    edge_lengths = [bm.edges[i].calc_length() for i in valid_indices]
    avg_length = sum(edge_lengths) / len(edge_lengths)
    variance = sum((l - avg_length) ** 2 for l in edge_lengths) / len(edge_lengths)
    
    return avg_length, variance ** 0.5

def compute_mesh_volume(bm):
    try: return bm.calc_volume()
    except AttributeError: return 0.0

def detect_mesh_clipping_tracked(bm, tracked_indices, threshold=0.0001):
    """Detect self-intersections by checking consistent face indices across frames."""
    if not tracked_indices or len(bm.faces) == 0: return 0.0
    
    valid_indices = [i for i in tracked_indices if i < len(bm.faces)]
    if not valid_indices: return 0.0
    
    bvh = bvhtree.BVHTree.FromBMesh(bm)
    overlapping_faces = 0
    
    for i in valid_indices:
        face = bm.faces[i]
        face_center = face.calc_center_median()
        face_normal = face.normal
        test_point = face_center + face_normal * threshold
        
        nearby = bvh.find_nearest(test_point)
        if nearby and nearby[2] != face.index:
            overlapping_faces += 1
            
    return overlapping_faces / len(valid_indices)

def check_watertight(bm):
    for edge in bm.edges:
        if len(edge.link_faces) != 2: return False
    return True

def compute_velocity_acceleration_l2(positions, timestamps):
    """Calculate average L2 (euclidean distance) norms of velocity and acceleration, and a jitter score based on acceleration variance."""
    if len(positions) < 2: return 0.0, 0.0, 0.0
    velocities = []
    for i in range(1, len(positions)):
        dt = timestamps[i] - timestamps[i-1]
        if dt > 0: velocities.append((positions[i] - positions[i-1]) / dt)
    
    avg_vel_l2 = sum(v.length for v in velocities) / len(velocities) if velocities else 0.0
    
    accelerations = []
    for i in range(1, len(velocities)):
        dt = timestamps[i+1] - timestamps[i] if i+1 < len(timestamps) else (timestamps[i] - timestamps[i-1])
        if dt > 0: accelerations.append((velocities[i] - velocities[i-1]) / dt)
    
    avg_acc_l2 = sum(a.length for a in accelerations) / len(accelerations) if accelerations else 0.0
    
    if accelerations:
        acc_lengths = [a.length for a in accelerations]
        mean_acc = sum(acc_lengths) / len(acc_lengths)
        jitter_score = math.sqrt(sum((a - mean_acc) ** 2 for a in acc_lengths) / len(acc_lengths))
    else: jitter_score = 0.0
    
    return avg_vel_l2, avg_acc_l2, jitter_score

def setup_camera_and_light():
    for obj in bpy.context.scene.objects:
        if obj.type in ['CAMERA', 'LIGHT']:
            bpy.data.objects.remove(obj, do_unlink=True)

    light_data = bpy.data.lights.new(name="Light", type='SUN')
    light_data.energy = 5.0
    light_obj = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(45), math.radians(45), 0)

    cam_data = bpy.data.cameras.new('Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_obj.location = (0, -4, 2)
    cam_obj.rotation_euler = (math.radians(70), 0, 0)

def render_frame(render_path):
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.filepath = render_path
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    try: bpy.ops.render.render(write_still=True)
    except Exception as e: print(f"Render failed: {e}")

def evaluate_animation(glb_path, output_json):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    try: bpy.ops.import_scene.gltf(filepath=glb_path)
    except Exception as e:
        with open(output_json, 'w') as f: json.dump({"error": f"GLB import error: {str(e)}"}, f)
        return
        
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
    
    if not meshes or not armatures or len(bpy.data.actions) == 0:
        with open(output_json, 'w') as f: json.dump({"error": "No skeletal animation found in GLB."}, f)
        return

    mesh_obj = sorted(meshes, key=lambda m: len(m.data.vertices), reverse=True)[0]
    start_frame = int(min([action.frame_range[0] for action in bpy.data.actions]))
    end_frame = int(max([action.frame_range[1] for action in bpy.data.actions]))
    total_frames = end_frame - start_frame
    mid_frame = start_frame + (total_frames // 2)
    
    if total_frames <= 1:
        with open(output_json, 'w') as f: json.dump({"error": "Animation too short."}, f)
        return

    setup_camera_and_light()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene

    # ---------------------------------------------------------
    # BASE FRAME - FIXED INITIAL REFERENCE (DETERMINISTIC)
    # ---------------------------------------------------------
    scene.frame_set(start_frame)
    depsgraph.update()
    base_mesh = get_evaluated_mesh(mesh_obj, depsgraph)
    
    if len(base_mesh.vertices) == 0:
        with open(output_json, 'w') as f: json.dump({"error": "Mesh has 0 vertices."}, f)
        return
    
    bm_base = bmesh.new()
    bm_base.from_mesh(base_mesh)
    
    # Force Blender to ensure edge and face lookup tables before using random.sample
    bm_base.edges.ensure_lookup_table()
    bm_base.faces.ensure_lookup_table()
    
    # Randomly sample 1500 edges, 400 faces, and 150 vertices for tracking across frames (or all if fewer than those)
    tracked_edges = random.sample(range(len(bm_base.edges)), min(1500, len(bm_base.edges)))
    tracked_faces = random.sample(range(len(bm_base.faces)), min(400, len(bm_base.faces)))
    tracked_vertices = random.sample(range(len(base_mesh.vertices)), min(150, len(base_mesh.vertices)))
    
    base_edge_length, base_edge_stretch = compute_edge_metrics_tracked(bm_base, tracked_edges) # Compute base edge length and stretch variance for the tracked edges in the base frame
    is_watertight_base = check_watertight(bm_base) # Check if the base mesh is watertight
    base_volume = compute_mesh_volume(bm_base) # Compute base mesh volume for later comparison
    bm_base.free()
    
    z_coords = [v.co.z for v in base_mesh.vertices]
    min_z = min(z_coords)
    foot_threshold = min_z + (max(z_coords) - min_z) * 0.1 # consider vertices in the lowest 10% of the mesh as potential foot vertices
    foot_indices = [v.index for v in base_mesh.vertices if v.co.z <= foot_threshold]
    prev_foot_positions = {i: base_mesh.vertices[i].co.copy() for i in foot_indices}
    
    mesh_obj.evaluated_get(depsgraph).to_mesh_clear()

    # Mid frame assessment
    scene.frame_set(mid_frame)
    depsgraph.update()
    eval_mesh_mid = get_evaluated_mesh(mesh_obj, depsgraph)
    bm_mid = bmesh.new()
    bm_mid.from_mesh(eval_mesh_mid)
    is_watertight_anim = check_watertight(bm_mid)
    bm_mid.free()
    
    render_output_path = os.path.abspath(glb_path).replace(".glb", "_anim_render.png")
    render_frame(render_output_path)
    mesh_obj.evaluated_get(depsgraph).to_mesh_clear()

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------
    skating_frames = 0
    total_acceleration = 0.0
    prev_velocities = []
    joint_positions = []
    timestamps = []
    
    max_expansion = 0.0
    max_edge_stretch = base_edge_stretch
    max_clipping_ratio = 0.0
    max_volume_variation = 0.0
    
    # Analyses a total of 10 frames evenly spaced across the animation to balance performance and insight.
    step = max(1, total_frames // 10) 
    frames_analyzed = 0
    
    for f in range(start_frame + step, end_frame + 1, step):
        scene.frame_set(f)
        depsgraph.update()
        eval_mesh = get_evaluated_mesh(mesh_obj, depsgraph)
        
        bm = bmesh.new()
        bm.from_mesh(eval_mesh)
        
        # Ensure lookup tables on the new BMesh before extracting data
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        # 1. Edge Expansion (check if the same tracked edges have stretched significantly compared to the base frame)
        current_edge_length, edge_stretch = compute_edge_metrics_tracked(bm, tracked_edges)
        if base_edge_length > 0:
            expansion_pct = ((current_edge_length - base_edge_length) / base_edge_length) * 100
            if expansion_pct > max_expansion: max_expansion = expansion_pct
        if edge_stretch > max_edge_stretch: max_edge_stretch = edge_stretch
            
        # 2. Volume Variation (check if the overall mesh volume has changed significantly compared to the base frame, which can indicate unnatural deformation)
        if base_volume > 0:
            current_volume = compute_mesh_volume(bm)
            var_pct = abs((current_volume - base_volume) / base_volume) * 100
            if var_pct > max_volume_variation: max_volume_variation = var_pct
            
        # 3. Clipping Ratio (check if any of the tracked faces are self-intersecting, which can indicate severe mesh deformation issues)
        clip_ratio = detect_mesh_clipping_tracked(bm, tracked_faces)
        if clip_ratio > max_clipping_ratio: max_clipping_ratio = clip_ratio
        
        bm.free()
            
        # 4. Foot Skating (detect if all the vertices in the foot region (10 % of the mesh) are sliding on the ground plane across frames, by tracking their positions and checking if they move significantly in the XY plane while staying close to the ground level in Z)
        is_skating = False
        for i in foot_indices:
            try:
                curr_pos = eval_mesh.vertices[i].co
                if curr_pos.z < foot_threshold and (curr_pos.xy - prev_foot_positions[i].xy).length > 0.05:
                    is_skating = True; break
                prev_foot_positions[i] = curr_pos.copy()
            except IndexError: pass
        if is_skating: skating_frames += 1
            
        # 5. Jitter (assess if the tracked vertices exhibit erratic movement by calculating their acceleration and checking for high variance in acceleration magnitudes across frames)
        valid_tracking_indices = [i for i in tracked_vertices if i < len(eval_mesh.vertices)]
        if valid_tracking_indices:
            center_of_mass = sum((eval_mesh.vertices[i].co for i in valid_tracking_indices), Vector()) / len(valid_tracking_indices)
            if prev_velocities:
                curr_vel = (center_of_mass - prev_velocities[0]['pos']).length 
                total_acceleration += abs(curr_vel - prev_velocities[-1])
                prev_velocities.append(curr_vel)
            else: prev_velocities = [{'pos': center_of_mass}, 0.0]
            
            joint_positions.append(center_of_mass.copy())
            timestamps.append(f - start_frame)
        
        mesh_obj.evaluated_get(depsgraph).to_mesh_clear()
        frames_analyzed += 1

    # ---------------------------------------------------------
    # FINALIZE METRICS
    # ---------------------------------------------------------
    metrics = {
        "max_edge_expansion_pct": round(max_expansion, 2), # Ou of all 10 frames analyzed, what was the maximum percentage increase in edge length compared to the base frame
        "foot_skating_ratio": round((skating_frames / frames_analyzed) * 100, 2) if frames_analyzed > 0 else 0, # Out of all 10 frames analyzed, what was the percentage of frames where the foot region was skating
        "jitter_acceleration": round(total_acceleration / frames_analyzed, 4) if frames_analyzed > 0 else 0, # Average change in velocity (acceleration) across frames for the tracked vertices
        "is_watertight_base": is_watertight_base, # Check if the base mesh (at frame 0) is watertight
        "is_watertight_anim": is_watertight_anim, # Check if the mesh at the mid animation frame is watertight
        "render_path": render_output_path, # Path to the rendered animation frame for visual inspection
        "mesh_clipping_ratio": round(max_clipping_ratio, 4), # Maximum ratio of self-intersecting faces among the tracked faces across the analyzed frames
        "volume_variation_pct": round(max_volume_variation, 2), # Maximum percentage change in mesh volume compared to the base frame
        "edge_stretch_variance": round(max_edge_stretch, 4), # Maximum percentage change in edge length compared to the base frame
        "avg_velocity_l2": 0.0,
        "avg_acceleration_l2": 0.0, 
        "jitter_score": 0.0 
    }
    
    # Compute velocity and acceleration metrics
    if joint_positions:
        avg_vel_l2, avg_acc_l2, jitter_score = compute_velocity_acceleration_l2(joint_positions, timestamps)
        metrics["avg_velocity_l2"] = round(avg_vel_l2, 4) # Average L2 norm of velocity for the tracked vertices across frames, which can indicate how fast the joints are moving on average
        metrics["avg_acceleration_l2"] = round(avg_acc_l2, 4) # Average L2 norm of acceleration for the tracked vertices across frames
        metrics["jitter_score"] = round(jitter_score, 4) # A measure of the variance in acceleration magnitude across frames
    
    with open(output_json, 'w') as f: json.dump(metrics, f)

if __name__ == "__main__":
    try:
        idx = sys.argv.index("--")
        evaluate_animation(sys.argv[idx + 1], sys.argv[idx + 2])
    except ValueError: pass