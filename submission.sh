#!/bin/bash
mkdir -p results

echo "#################################################"
echo "Rendering the first Mesh"
echo "#################################################"
python3 submission.py --question 0.1
echo '------------DONE------------'

echo "#################################################"
echo "Running question 1.1 360-degree Renders"
echo "#################################################"
python3 submission.py --question 1.1
echo '------------DONE------------'

echo "#################################################"
echo "Running questiion 1.2 Re-creating the Dolly Zoom"
echo "#################################################"
python3 submission.py --question 1.2
echo '------------DONE------------'

echo "#################################################"
echo "Running question 2.1 Constructing a Tetrahedron"
echo "#################################################"
python3 submission.py --question 2.1
echo '------------DONE------------'

echo "#################################################"
echo "Running question 2.2 Constructing a Cube"
echo "#################################################"
python3 submission.py --question 2.2
echo '------------DONE------------'

echo "#################################################"
echo "Running question 3 Re-texturing a mesh"
echo "#################################################"
python3 submission.py --question 3
echo '------------DONE------------'

echo "#################################################"
echo "Running question 4 Camera Transformations"
echo "#################################################"
python3 submission.py --question 4
echo '------------DONE------------'

echo "#################################################"
echo "Running question 5.1 Rendering Point Cloud from RGBD Images"
echo "#################################################"
python3 submission.py --question 5.1
echo '------------DONE------------'

echo "#################################################"
echo "Running question 5.2 Torus using Parametric Function"
echo "#################################################"
python3 submission.py --question 5.2
echo '------------DONE------------'

echo "#################################################"
echo "Running question 5.3 Torus using Implicit Surface"
echo "#################################################"
python3 submission.py --question 5.3
echo '------------DONE------------'
