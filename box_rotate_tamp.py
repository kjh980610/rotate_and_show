import numpy as np
import sys, os
import yaml

from pykin import assets
from pykin.utils import plot_utils as p_utils

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.utils.mesh_utils import get_object_mesh
from pykin.utils.transform_utils import get_matrix_from_rpy
from pykin.utils.transform_utils import get_rpy_from_matrix

from pytamp.scene.scene_manager import SceneManager
from pytamp.planners.rrt_star_planner import RRTStarPlanner
from collections import OrderedDict

asset_file_path = os.path.abspath(assets.__file__ + "/../")

# 로봇 불러오기

file_path = "urdf/doosan/doosan_with_robotiq140.urdf"
robot = SingleArm(
    f_name=file_path,
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]),
    has_gripper=True,
    gripper_name="robotiq140",
)

robot.setup_link_name("base_0", "link6")
robot.init_qpos = np.array([0, 0, 0, 0, 0, 0])

custom_fpath = asset_file_path + "/config/doosan_init_params.yaml"
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]

# 씬 설정(테이블, 박스)

red_cube_mesh = get_object_mesh("ben_cube.stl",scale=[0.15, 0.1, 0.1])
table_mesh = get_object_mesh("ben_table.stl",  scale=[1.0, 1.5, 1.0])
table_height = table_mesh.bounds[1][2] - table_mesh.bounds[0][2]
b_z = table_height + abs(red_cube_mesh.bounds[0][2])

red_box_pose = Transform(
            pos=np.array([0.6, 0, table_height + abs(red_cube_mesh.bounds[0][2])+0.074]),
            rot=np.array([0,0,0])
        )
table_pose = Transform(
            pos=np.array([1.0, -0.6, 0.043])
        )


scene_mngr = SceneManager("collision", is_pyplot=True)
scene_mngr.add_object(
    name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.823, 0.71, 0.55]
)
scene_mngr.add_object(
    name="red_box",
    gtype="mesh",
    gparam=red_cube_mesh,
    h_mat=red_box_pose.h_mat,
    color=[1.0, 0.0, 0.0],
)

#초기 위치
init_qpos = np.array([0, 0, np.pi / 2, 0, np.pi / 2, 0])
scene_mngr.add_robot(robot, init_qpos)


#grasp 위치 설정
init_pose = scene_mngr.get_robot_eef_pose()
target_yaw = get_rpy_from_matrix(red_box_pose.h_mat)[2]
grasp_pose = red_box_pose.h_mat
r_mat = get_matrix_from_rpy(np.array([0, np.pi, target_yaw]))
grasp_pose[:3, :3] = r_mat
grasp_pose[:3, 3] = grasp_pose[:3, 3] + [0, 0, 0.2]



def update_joint_path(scene_mngr, target_pose,joint_path):
    planner = RRTStarPlanner(delta_distance=0.05, epsilon=0.2, gamma_RRT_star=2, dimension = robot.arm_dof)
    planner.run(scene_mngr=scene_mngr, cur_q=joint_path[-1], goal_pose=target_pose, max_iter=300)
    joint_path += planner.get_joint_path(n_step=5)

    for i in range(3) :
         joint_path.append(joint_path[-1])

joint_path =[init_qpos]
joint_pathes = OrderedDict()

att_idx=[]
det_idx=[]
pathes_num = 0

#회전 각도
theta = np.deg2rad(-70)


update_joint_path(scene_mngr, grasp_pose, joint_path)
scene_mngr.attach_object_on_gripper("red_box")

att_idx.append(len(joint_path)-3)


default_pose=init_pose + np.zeros((4,4))
default_pose[:3, :3] = r_mat


update_joint_path(scene_mngr, default_pose, joint_path)


rot_z =  get_matrix_from_rpy(np.array([0, 0, theta]))

target_pose = grasp_pose + np.zeros((4,4))
target_pose[:3, :3]=rot_z.dot(grasp_pose[:3, :3])


update_joint_path(scene_mngr, target_pose, joint_path)


scene_mngr.detach_object_from_gripper("red_box")
red_box_pose2=red_box_pose.h_mat
red_box_pose2[:3,:3]=target_pose[:3, :3]
scene_mngr.add_object(
    name="red_box",
    gtype="mesh",
    gparam=red_cube_mesh,
    h_mat=red_box_pose2,
    color=[1.0, 0.0, 0.0],
)
det_idx.append(len(joint_path)-3)


update_joint_path(scene_mngr, init_pose, joint_path)


scene_mngr.remove_object("red_box")
scene_mngr.add_object(
    name="red_box",
    gtype="mesh",
    gparam=red_cube_mesh,
    h_mat=red_box_pose.h_mat,
    color=[1.0, 0.0, 0.0],
)

fig, ax = p_utils.init_3d_figure()
scene_mngr.animation(
    ax,
    fig,
    joint_path=joint_path,
    visible_text=True,
    alpha=1,
    interval=50,
    repeat=False,
    pick_object=["red_box"],
    attach_idx= att_idx,
    detach_idx= det_idx,
    # is_save=True,
    # video_name='rotate_init0'
)