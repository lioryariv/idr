# Data Convention

### Camera information and normalization
Besides RGB and mask images, IDR needs cameras information in order to run. For each scan out of the 15 DTU scans that are presented in the paper we supply two npz files:
- `cameras.npz` for fixed cameras setup
- `cameras_linear_init.npz` for trainable cameras setup

A cameras file contains for each image a projection matrix (named "world_mat_{i}"), and a normalization matrix (named "scale_mat_{i}").

#### Camera projection matrix
A 3x4 camera projection matrix, P = K[R t] projects points from 3D coordinates to image pixels by the formula: d[x; y; 1]=P[X;Y;Z;1] where K is a 3x3 calibration matrix, [R t] is 3x4 a world to camera Euclidean transformation, [X;Y;Z] is the 3D point, [x;y] is the 2D pixel coordinates of the projected point and d is the depth of the point.
The "world_mat" matrix is a concatenation of the camera projection matrix with a row vector of [0,0,0,1] (which makes it a 4x4 matrix).

#### Normalization matrix
The normalization matrix is used to normalize the cameras such that the visual hull of the observed object is approximately inside the unit sphere. 


### Preprocess
In order to generate a normalization matrix for each scan, we used the input object masks and camera projection matrices. A script that demonstrates this process is presented in: `code/preprocess/preprocess_cameras.py`.
Note: in order to run the supplied 15 scans, it is not required to run this script. 

For running the pre-processing script on all DTU scans:
```
cd ./code
python preprocess_cameras.py --dtu
```

For running the pre-process script on the linear init cameras:

```
cd ./code
python preprocess_cameras.py --dtu --use_linear_init
```

#### New data
In order to run IDR on new data [DIR PATH], you need to supply `image` and `mask` directories, as well as `cameras.npz` file containing the appropriate camera projection matrices.

For running the pre-process script to generate a `cameras.npz` file that contains the suitable normalization matrix:

```
cd ./code
python preprocess_cameras.py --source_dir [DIR PATH]
```
