bl_info = {
    "name": "Starst3r 3D Reconstruction",
    "description": "3D reconstruction from 2D images using MASt3R.",
    "author": "Patrick Huang",
    "version": (0, 0, 1),
    "blender": (2, 80, 0),
    "location": "View3D > Object > Starst3r",
    "category": "Object",
}

from . import interface


def register():
    interface.register()


def unregister():
    interface.unregister()
