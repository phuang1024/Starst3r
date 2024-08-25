cd build

cp -r ../starster .
rm starster/dust3r starster/mast3r

cp -r ../mast3r/mast3r starster/
cp -r ../mast3r/dust3r/dust3r starster/

python setup.py sdist bdist_wheel
