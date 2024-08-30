from pathlib import Path

import bpy
import bmesh


def import_data(context):
    props = context.scene.starster

    recons = infer_model(context)

    if props.import_as in ("VERTS", "DUPLI"):
        make_mesh(context, recons, dupli=props.import_as == "DUPLI")


def infer_model(context):
    import starster
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    props = context.scene.starster

    model_path = props.model_path
    dir = Path(bpy.path.abspath(props.directory))
    res = props.resolution

    images = []
    filepaths = []
    for file in dir.iterdir():
        if file.suffix.lower() in (".jpg", ".jpeg", ".png"):
            images.append(starster.load_image(str(file), res))
            filepaths.append(str(file))

    print("Found images:", filepaths)

    print("Reconstruct.")
    model = starster.Mast3rModel.from_pretrained(model_path).to(DEVICE)
    recons = starster.reconstruct_scene(model, images, filepaths, DEVICE)

    return recons

def make_mesh(context, recons, dupli=False):
    print("Making mesh.")

    import starster

    props = context.scene.starster
    dsize = props.dupli_size

    num_verts = starster.num_verts(recons)
    if props.import_as == "DUPLI":
        num_verts *= 4

    mesh = bpy.data.meshes.new("Starster")
    obj = bpy.data.objects.new("Starster", mesh)
    bpy.context.collection.objects.link(obj)

    # Make mesh
    bm = bmesh.new()
    for loc, col in starster.iterate_verts(recons):
        if props.import_as == "DUPLI":
            v1 = bm.verts.new((loc[0] - dsize, loc[1] - dsize, loc[2] - dsize))
            v2 = bm.verts.new((loc[0] + dsize, loc[1] - dsize, loc[2] - dsize))
            v3 = bm.verts.new((loc[0], loc[1] + dsize, loc[2] - dsize))
            v4 = bm.verts.new((loc[0], loc[1], loc[2] + dsize))
            bm.faces.new((v1, v2, v3))
            bm.faces.new((v1, v2, v4))
            bm.faces.new((v1, v3, v4))
            bm.faces.new((v2, v3, v4))
        else:
            bm.verts.new((loc[0], loc[1], loc[2]))

    bm.to_mesh(mesh)

    # Make vertex colors
    vert_colors = mesh.attributes.new(name="Color", type="FLOAT_COLOR", domain="POINT")
    i = 0
    for loc, col in starster.iterate_verts(recons):
        if props.import_as == "DUPLI":
            for j in range(4):
                vert_colors.data[i + j].color = (col[0], col[1], col[2], 1)
            i += 4
        else:
            vert_colors.data[i].color = (col[0], col[1], col[2], 1)
            i += 1
