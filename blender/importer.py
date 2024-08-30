from pathlib import Path

import bpy


def import_data(context):
    props = context.scene.starster

    recons = infer_model(context)

    if props.import_as in ("VERTS", "DUPLI"):
        make_mesh(recons, dupli=props.import_as == "DUPLI")


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

def make_mesh(recons, dupli=False):
    import starster

    print("Making mesh.")

    mesh = bpy.data.meshes.new("Starster")
    obj = bpy.data.objects.new("Starster", mesh)
    bpy.context.collection.objects.link(obj)

    mesh.vertices.add(starster.num_verts(recons))
    vert_colors = mesh.attributes.new(name="Color", type="FLOAT_COLOR", domain="POINT")

    i = 0
    for loc, col in starster.iterate_verts(recons):
        mesh.vertices[i].co = (loc[0], loc[1], loc[2])
        vert_colors.data[i].color = (col[0], col[1], col[2], 1)
        i += 1
