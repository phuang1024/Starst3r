"""
Bpy interface. Operators, props, and UI.
"""

import os

import bpy

from .importer import import_main


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

    import_as: bpy.props.EnumProperty(
        name="Import As",
        description="Type of object to import as.",
        items=[
            ("VERTS", "Vertices", "Import as vertices."),
            ("DUPLI", "DupliVerts", "Create small mesh at each vert for rendering."),
            ("POINT_CLOUD", "Point Cloud", "Import as point cloud object (in experimental)."),
        ],
        default="DUPLI",
    )

    dupli_size: bpy.props.FloatProperty(
        name="Dupli Size",
        description="Size of dupli verts.",
        default=0.01,
        min=0.0
    )

    make_material: bpy.props.BoolProperty(
        name="Make Material",
        description="Make material for object.",
        default=True
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
        props = context.scene.starster

        if not os.path.isfile(props.model_path):
            self.report({"ERROR"}, "Model file does not exist.")
            return False
        if not os.path.isdir(props.directory):
            self.report({"ERROR"}, "Directory does not exist.")
            return False
        return True

    def execute(self, context):
        if not self.verify_props(context):
            return {"CANCELLED"}
        import_main(context)
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
        props = context.scene.starster

        layout.prop(props, "model_path")
        layout.prop(props, "directory")
        layout.prop(props, "resolution")
        layout.prop(props, "import_as")
        if props.import_as == "DUPLI":
            layout.prop(props, "dupli_size")
        layout.prop(props, "make_material")

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
