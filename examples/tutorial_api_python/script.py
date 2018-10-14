# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
from PIL import Image
import numpy as np

# Remember to add your installation path here
# Option a
dir_path = os.path.dirname(os.path.realpath(__file__))
if platform == "win32":
    sys.path.append(dir_path + '/../../python/openpose/');
else:
    sys.path.append('../../python');
# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
# sys.path.append('/usr/local/python/openpose')
# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    from openpose import *
except:
    raise Exception(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = dir_path + "/../../../models/"
# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)

# Read new image
img = cv2.imread("../../../examples/media/rsz_img_8894.jpg")
# cv2.imshow("output",img)
# Output keypoints and the image with the human skeleton blended on it
keypoints, output_image = openpose.forward(img, True)

# Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
lst = keypoints[0].reshape(25, 3)[:, :2]

#np.savetxt("output2.txt", keypoints.reshape(25, 3)[:, :2], fmt='%.3e')
# Display the image
# cv2.imshow("output", output_image)
#output_image_jpg = Image.fromarray(output_image)

#output_image_jpg.save("output3.jpg")

# print(output_image.shape)
#cv2.waitKey(15)

indices = np.array([8, 1, 13, 12, 9, 19, 14, 22, 11, 10])

x_coord = lst[:,0]

y_coord = output_image.shape[0] - lst[:,1]

#indices where x_coord is origin
badx = indices[np.where(x_coord[indices] == 0)[0]]
bady = indices[np.where(y_coord[indices] == output_image.shape[0])[0]]
bad = np.intersect1d(badx, bady)

def create_vector(x, y, index1, index2, bad):
    if(np.any(bad == index1) or np.any(bad == index2)):
        return None
    return np.array([x[index1] - x[index2], y[index1] - y[index2]])

back = create_vector(x_coord, y_coord, 8, 1, bad)
lthigh = create_vector(x_coord, y_coord, 13, 12, bad)
rthigh = create_vector(x_coord, y_coord, 10, 9, bad)
lcalf = create_vector(x_coord, y_coord, 14, 13, bad)
rcalf = create_vector(x_coord, y_coord, 11, 10, bad)
lfoot = create_vector(x_coord, y_coord, 19, 14, bad)
rfoot = create_vector(x_coord, y_coord, 22, 11, bad)
neck = create_vector(x_coord, y_coord, 1, 0, bad)

#back = np.array([x_coord[8]-x_coord[1],y_coord[8]-y_coord[1]])
#lthigh = np.array([x_coord[13]-x_coord[12],y_coord[13]-y_coord[12]])
#rthigh = [x_coord[10]-x_coord[9],y_coord[10]-y_coord[9]]
#lcalf = [x_coord[14]-x_coord[13],y_coord[14]-y_coord[13]]
#rcalf = [x_coord[11]-x_coord[10],y_coord[11]-y_coord[10]]
#lfoot = [x_coord[19]-x_coord[14],y_coord[19]-y_coord[14]]
#rfoot = [x_coord[22]-x_coord[11],y_coord[22]-y_coord[11]]
#neck = [x_coord[1]-x_coord[0],y_coord[1]-y_coord[0]]
# add arms if possible

vectors = np.array([neck,back,rthigh,rcalf,lthigh,lcalf,rfoot,lfoot])

def angle_between(v1, v2):
    if(v1 is None or v2 is None):
        return -1
    return (np.arccos(np.dot(v1,-1*v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))*180/np.pi)

def distance_from_range(value, bound1, bound2):
    if (bound1 <= value <= bound2)
        return 0
    elif (value == -1)
        return false
    return max(value-bound1, value-bound2)

neckback = angle_between(vectors[0], vectors[1])
backrthigh = angle_between(vectors[1], vectors[2])
rthighrcalf = angle_between(vectors[2], vectors[3])
lthighlcalf = angle_between(vectors[4], vectors[5])
lfootlthigh = angle_between(vectors[-1], vectors[-4])
rfootrthigh = angle_between(vectors[-2], vectors[2])

output = np.array([distance_from_range(neckback, 150, 170), distance_from_range(backrthigh, 85, 95), distance_from_range(rthighrcalf, 80, 105), distance_from_range(lthighlcalf, 80, 105), distance_from_range(lfootlthigh, 0, 10), distance_from_range(rfootrthigh, 0, 10)])

#np.savetxt("output.txt", output, fmt='%.5e')
