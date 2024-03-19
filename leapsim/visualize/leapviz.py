import os, sys
from os.path import join, dirname, abspath
import time
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio
from pinocchio.visualize import MeshcatVisualizer


# parameters
BOX_SIZE = '0.10'
NUM_DISPLAYS = 30
CACHE_PREFIX = "random_rpy_cache"

# This path refers to Pinocchio source code but you can define your own directory here.
pinocchio_model_dir = join(dirname(dirname(dirname(str(abspath(__file__))))), "assets/leap_hand")
 
model_path = pinocchio_model_dir
mesh_dir = pinocchio_model_dir
urdf_model_path = join(model_path,"robot_cube.urdf")
 
# Load the urdf model
model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(urdf_model_path, mesh_dir)
print('model name: ' + model.name)
 
# Create data required by the algorithms
data, collision_data, visual_data  = pinocchio.createDatas(model, collision_model, visual_model)

viz = MeshcatVisualizer(model, collision_model, visual_model)
 
# Start a new MeshCat server and client.
# Note: the server can also be started separately using the "meshcat-server" command in a terminal:
# this enables the server to remain active after the current script ends.
#
# Option open=True pens the visualizer.
# Note: the visualizer can also be opened seperately by visiting the provided URL.
try:
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)
 
# Load the robot in the viewer.
viz.loadViewerModel()

# load joint data
cache_dir = join(dirname(dirname(str(abspath(__file__)))), "cache")
cache_file_name = "{0}_grasp_50k_s{1}.npy".format(CACHE_PREFIX, ''.join(str(BOX_SIZE).split('.')[1:]))
cached_joint_pos_data = np.load(os.path.join(cache_dir, cache_file_name))

# euler_angle_distribution
cache_obj_euler_angles = R.from_quat(cached_joint_pos_data[:, -4:]).as_euler('xyz')
plt.hist(cache_obj_euler_angles[:, 0], bins=50, density=True, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')
plt.hist(cache_obj_euler_angles[:, 1], bins=50, density=True, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')
plt.hist(cache_obj_euler_angles[:, 2], bins=50, density=True, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')
plt.show()

# visualize
selected_indexes = np.random.choice(np.arange(len(cached_joint_pos_data)), NUM_DISPLAYS, replace=False)
selected_joint_pos = cached_joint_pos_data[selected_indexes]
print("selected indexes: ", selected_indexes)
for i_j in range(len(selected_joint_pos)):
    q0 = selected_joint_pos[i_j]
    viz.display(q0)
    time.sleep(2.0)
