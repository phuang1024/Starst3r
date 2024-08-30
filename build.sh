set -e

function build_python() {
    mkdir -p build
    cd build

    cp ../setup.py .

    cp -r ../starster .
    rm starster/dust3r starster/mast3r

    cp -r ../mast3r/mast3r .
    cp -r ../mast3r/dust3r/dust3r .
    # Workaround for mast3r code; copy dust3r into itself.
    cp -r ../mast3r/dust3r/dust3r dust3r/
    cp -r ../mast3r/dust3r/croco/models .
    # WOrkaround; copy croco into dust3r.
    cp -r ../mast3r/dust3r/croco dust3r/
    # Workaround for missing init.
    touch models/__init__.py dust3r/croco/__init__.py dust3r/croco/models/__init__.py

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
    echo "Invalid usage."
fi
