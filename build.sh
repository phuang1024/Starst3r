set -e

function build_python() {
    mkdir -p build
    cd build

    cp -r ../starster .
    rm starster/dust3r starster/mast3r

    cp -r ../mast3r/mast3r starster/
    cp -r ../mast3r/dust3r/dust3r starster/
    rm starster/dust3r/dust3r
    cp -r ../mast3r/dust3r/dust3r starster/dust3r/

    python setup.py sdist bdist_wheel
}

function build_blender() {
    mkdir -p build/blender
    cd build/blender

    cp -r ../../blender starster_blender
    zip -r starster.zip starster_blender
}

if [ "$1" == "python" ]; then
    build_python
elif [ "$1" == "blender" ]; then
    build_blender
else
    build_python
    build_blender
fi
