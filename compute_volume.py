# compute_volume.py
import bpy
import sys

def main():
    # parse args after “--”
    argv = sys.argv
    if "--" not in argv or len(argv) < argv.index("--") + 3:
        print("Usage: blender --background --python compute_volume.py -- mesh_file density")
        sys.exit(1)
    mesh_path = argv[argv.index("--") + 1]
    density  = float(argv[argv.index("--") + 2])

    # import your mesh (adjust importer if needed)
    bpy.ops.import_scene.obj(filepath=mesh_path)

    # grab the imported mesh object
    obj = next(o for o in bpy.context.selected_objects if o.type == "MESH")

    # If you have modifiers, evaluate them first:
    # depsgraph = bpy.context.evaluated_depsgraph_get()
    # obj_eval = obj.evaluated_get(depsgraph)
    # mesh_eval = obj_eval.to_mesh()

    # calc_volume returns the absolute volume in Blender units³
    vol = obj.calc_volume(signed=False)

    # clean up the evaluated mesh
    # obj_eval.to_mesh_clear()

    # convert from m³ → cm³ (1 m³ = 1e6 cm³) then × density (g/cm³)
    mass_g = vol * 1e6 * density

    print(f"{mass_g:.6f}")

if __name__ == "__main__":
    main()