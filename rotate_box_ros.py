#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from typing import OrderedDict
import numpy as np
import rospy
from dsr_msgs.srv import OperatePytamp,Robotiq2FMove

import os
import yaml

from pykin import assets
from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.utils.mesh_utils import get_object_mesh
from pykin.utils.transform_utils import get_matrix_from_rpy
from pykin.utils.transform_utils import get_rpy_from_matrix

from pytamp.scene.scene_manager import SceneManager
from pytamp.planners.rrt_star_planner import RRTStarPlanner
from collections import OrderedDict



rospy.init_node('box_rotate', anonymous=True)

operate_robot = rospy.ServiceProxy('/operate_robot_joint',OperatePytamp)
# 0.7 : close , 0 : open 
operate_gripper = rospy.ServiceProxy('/dsr01a0912/gripper/robotiq_2f_move', Robotiq2FMove)

def get_joint_pathes():

        
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

    #로봇 초기 설정
    scene_mngr = SceneManager("collision", is_pyplot=True)
    init_qpos = np.array([0, 0, np.pi / 2, 0, np.pi / 2, 0])
    scene_mngr.add_robot(robot, init_qpos)


    # 테이블 씬 설정
    table_mesh = get_object_mesh("ben_table.stl",  scale=[1.0, 1.5, 1.0])
    table_height = table_mesh.bounds[1][2] - table_mesh.bounds[0][2]
    table_pose = Transform(
                pos=np.array([1.0, -0.6, 0.043])
            )
    scene_mngr.add_object(
        name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.823, 0.71, 0.55]
    )


    #타겟 씬 설정
    target_init_yaw = 0 #rad
    red_cube_mesh = get_object_mesh("ben_cube.stl",scale=[0.15, 0.1, 0.1])
    target_pose = Transform(
                pos=np.array([0.6, 0, table_height + abs(red_cube_mesh.bounds[0][2])+0.074]),
                rot=np.array([0,0,target_init_yaw])
            )
    scene_mngr.add_object(
        name="target",
        gtype="mesh",
        gparam=red_cube_mesh,
        h_mat=target_pose.h_mat,
        color=[1.0, 0.0, 0.0],
    )

    #타겟 회전 각도
    theta = np.deg2rad(-70) #rad


    #타겟 위치를 통해 그랩 pose 설정(z)
    init_pose = scene_mngr.get_robot_eef_pose()                     #로봇 초기 eef pose 저장
    target_yaw = get_rpy_from_matrix(target_pose.h_mat)[2]          #타겟 현재 yaw 불러오기
    grasp_pose = target_pose.h_mat                                  
    r_mat = get_matrix_from_rpy(np.array([0, np.pi, target_yaw]))   #수직으로 잡기위해 p=pi
    grasp_pose[:3, :3] = r_mat                                      #yaw는 타겟과 동일
    grasp_pose[:3, 3] = grasp_pose[:3, 3] + [0, 0, 0.2]             #pos는 z만 약간 더 높게 설정



    def update_joint_path(scene_mngr, goal_pose,joint_path):        #rrt로 path 생성
        planner = RRTStarPlanner(delta_distance=0.05, epsilon=0.2, gamma_RRT_star=2, dimension = robot.arm_dof)
        planner.run(scene_mngr=scene_mngr, cur_q=joint_path[-1], goal_pose=goal_pose, max_iter=300)
        joint_path += planner.get_joint_path(n_step=10)


    #path 초기값
    joint_path =[init_qpos]         #[[qpos(7 joint angles)],...]
    joint_pathes = OrderedDict()    #OrderedDict([('task',[joint_path]),... ])
    pathes_num = 0                  #

    #grasp
    update_joint_path(scene_mngr, grasp_pose, joint_path)
    joint_pathes.update({"grasp" : joint_path[pathes_num:-1]})
    pathes_num = len(joint_path)
    scene_mngr.attach_object_on_gripper("target")                   #씬메니져 충돌감지를 위한 attach
    

    #post_grasp - 초기 위치로 돌아가지만 물체 잡을 때의 회전 상태는 유지
    default_pose=init_pose + np.zeros((4,4))
    default_pose[:3, :3] = r_mat

    update_joint_path(scene_mngr, default_pose, joint_path)
    joint_pathes.update({"post_grasp" : joint_path[pathes_num:-1]})
    pathes_num = len(joint_path)


    #회전 후 놓을 위치 계산(pos는 잡을 때 그대로, rot은 회전행렬 곱)
    rot_z =  get_matrix_from_rpy(np.array([0, 0, theta]))
    release_pose = grasp_pose + np.zeros((4,4))
    release_pose[:3, :3]=rot_z.dot(grasp_pose[:3, :3])

    update_joint_path(scene_mngr, release_pose, joint_path)
    joint_pathes.update({"release" : joint_path[pathes_num:-1]})
    pathes_num = len(joint_path)

    #씬메니져 충돌감지를 위한 dettach, 씬에서 제거된 물체 다시 추가
    scene_mngr.detach_object_from_gripper("target")             
    target_pose2=target_pose.h_mat
    target_pose2[:3,:3]=release_pose[:3, :3]
    scene_mngr.add_object(
        name="target",
        gtype="mesh",
        gparam=red_cube_mesh,
        h_mat=target_pose2,
        color=[1.0, 0.0, 0.0],
    )

    #post_release - 초기 위치로 돌아가기
    update_joint_path(scene_mngr, init_pose, joint_path)
    joint_pathes.update({"post_release" : joint_path[pathes_num:-1]})
    pathes_num = len(joint_path)
    return joint_pathes

joint_pathes = get_joint_pathes()

global_move_time = 6
local_move_time = 3

#초기 위치로 이동
operate_gripper(0)
init_pos = np.array([ 0, 0, np.pi/2, 0, np.pi/2, 0]).reshape(-1).tolist()
operate_robot(init_pos,global_move_time)

for task, value in joint_pathes.items():
        if task == "grasp":
            grasp_joint_path = np.array(value).reshape(-1).tolist()
            operate_robot(grasp_joint_path,local_move_time)
            operate_gripper(0.7)
        elif task == "post_grasp":
            post_joint_path = np.array(value).reshape(-1).tolist()
            operate_robot(post_joint_path,local_move_time)
        elif task == "release":
            release_joint_path = np.array(value).reshape(-1).tolist()
            operate_robot(release_joint_path,local_move_time)
            operate_gripper(0)
        elif task == "post_release":
            post_release_joint_path = np.array(value).reshape(-1).tolist()
            operate_robot(post_release_joint_path,local_move_time)
            
