# render_script.py
import bpy
import sys
import math
from mathutils import Vector

# Parse arguments passed after "--"
argv = sys.argv
argv = argv[argv.index("--") + 1:]
glb_path = argv[0]
out_path = argv[1]

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import the GLB model
bpy.ops.import_scene.gltf(filepath=glb_path)

# Calculate bounding box to dynamically frame the camera
objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not objects:
    sys.exit()

bbox_corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
min_coord = Vector((min([c.x for c in bbox_corners]), min([c.y for c in bbox_corners]), min([c.z for c in bbox_corners])))
max_coord = Vector((max([c.x for c in bbox_corners]), max([c.y for c in bbox_corners]), max([c.z for c in bbox_corners])))
center = (min_coord + max_coord) / 2
size = max((max_coord - min_coord))

# Setup Camera
cam_data = bpy.data.cameras.new('Camera')
cam_obj = bpy.data.objects.new('Camera', cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Position camera based on object size
cam_obj.location = (center.x, center.y - (size * 1.5), center.z + (size * 0.3))
cam_obj.rotation_euler = (math.radians(80), 0, 0) 

# Setup standard lighting
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_data.energy = 3.0
light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
bpy.context.collection.objects.link(light_obj)
light_obj.rotation_euler = (math.radians(45), math.radians(0), math.radians(20))

# Configure Render Settings (EEVEE is faster for batch processing)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.film_transparent = True
scene.render.filepath = out_path

# Execute Render
bpy.ops.render.render(write_still=True)