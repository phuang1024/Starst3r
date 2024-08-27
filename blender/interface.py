"""
Bpy interface. Operators, props, and UI.
"""

from pathlib import Path

import bpy
import bmesh


class StarsterProps(bpy.types.PropertyGroup):
    model_path: bpy.props.StringProperty(
        name="Model Path",
        description="Path to MASt3R model.",
        default="",
        subtype="FILE_PATH"
    )

    directory: bpy.props.StringProperty(
        name="Directory",
        description="Dir with source images.",
        default="",
        subtype="DIR_PATH"
    )

    resolution: bpy.props.IntProperty(
        name="Resolution",
        description="Resolution of images.",
        default=224,
        min=0
    )


class STARSTER_OT_ReconstructConfirm(bpy.types.Operator):
    """Show confirmation dialog before calling reconstruct."""
    bl_idname = "starster.reconstruct_confirm"
    bl_label = "Reconstruct"
    bl_description = "Reconstruct 3D model from 2D images using MASt3R."
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Reconstructing will take a while and will freeze Blender.")
        layout.label(text="Please save your work before proceeding.")

    def execute(self, context):
        bpy.ops.starster.reconstruct()
        return {"FINISHED"}


class STARSTER_OT_Reconstruct(bpy.types.Operator):
    bl_idname = "starster.reconstruct"
    bl_label = "Reconstruct"
    bl_description = "Reconstruct 3D model from 2D images using MASt3R."
    bl_options = {"REGISTER", "UNDO"}

    def verify_props(self, context):
        if not os.path.isfile(context.scene.starster.model_path):
            self.report({"ERROR"}, "Model file does not exist.")
            return False
        if not os.path.isdir(context.scene.starster.directory):
            self.report({"ERROR"}, "Directory does not exist.")
            return False
        return True

    def infer_model(self, context):
        import starster
        import torch
        from mast3r.model import AsymmetricMASt3R

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = context.scene.starster.model_path
        dir = Path(bpy.path.abspath(context.scene.starster.directory))
        res = context.scene.starster.resolution

        images = []
        filepaths = []
        for file in dir.iterdir():
            if file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                images.append(starster.load_image(str(file), res))
                filepaths.append(str(file))

        print("Found images:", filepaths)

        print("Reconstruct.")
        images = starster.prepare_images_for_mast3r(images)
        model = AsymmetricMASt3R.from_pretrained(model_path).to(DEVICE)
        recons = starster.reconstruct_scene(model, images, filepaths, DEVICE)

        return recons

    def make_mesh(self, context, recons):
        print("Making mesh.")

        i = 0
        while True:
            name = f"Starster.{i:03}"
            if name not in bpy.data.objects:
                break
            i += 1

        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)

        bm = bmesh.new()
        for pts in recons.pts3d:
            for pt in pts:
                bm.verts.new(pt)
        bm.to_mesh(mesh)
        bm.free()

    def execute(self, context):
        if not self.verify_props(context):
            return {"CANCELLED"}
        recons = self.infer_model(context)
        self.make_mesh(context, recons)
        return {"FINISHED"}


class BasePanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Starst3r"
    bl_options = {"DEFAULT_CLOSED"}


class STARSTER_PT_MainPanel(bpy.types.Panel, BasePanel):
    bl_idname = "STARSTER_PT_MainPanel"
    bl_label = "Starst3r"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene.starster, "model_path")
        layout.prop(scene.starster, "directory")
        layout.prop(scene.starster, "resolution")

        layout.operator("starster.reconstruct_confirm")


classes = (
    StarsterProps,

    STARSTER_OT_ReconstructConfirm,
    STARSTER_OT_Reconstruct,

    STARSTER_PT_MainPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.starster = bpy.props.PointerProperty(type=StarsterProps)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.starster
