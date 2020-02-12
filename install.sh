pip uninstall -y xwp_rotation

python setup.py bdist_wheel

rm -rf xwp_rotation.egg-info/

pip install --no-index --find-links=$(pwd)/dist xwp_rotation