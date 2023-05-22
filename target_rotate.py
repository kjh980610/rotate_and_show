#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from pytamp.planners.cartesian_planner import CartesianPlanner
from collections import OrderedDict



def target_rotate(current_pose,rotate_pose) :
    rospy.init_node('box_rotate', anonymous=True)

    operate_robot = rospy.ServiceProxy('/operate_robot_joint',OperatePytamp)
    # 0.7 : close , 0 : open 
    operate_gripper = rospy.ServiceProxy('/dsr01a0912/gripper/robotiq_2f_move', Robotiq2FMove)


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
                pos=np.array([1.0, -0.6, table_mesh.bounds[0][2]])
            )
    scene_mngr.add_object(
        name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.823, 0.71, 0.55]
    )


    #타겟 씬 설정
    red_cube_mesh = get_object_mesh("ben_cube.stl",scale=[0.15, 0.1, 0.1])
    target_offset = np.array([0.6, 0, table_height + abs(red_cube_mesh.bounds[0][2])+0.073])
    current_pose = Transform(pos = current_pose.pos + target_offset, rot=current_pose.rot)
    rotate_pose = Transform(pos = rotate_pose.pos + target_offset, rot=rotate_pose.rot)

    scene_mngr.add_object(
        name="target",
        gtype="mesh",
        gparam=red_cube_mesh,
        h_mat=current_pose.h_mat,
        color=[1.0, 0.0, 0.0],
    )


    def update_rrt_path(scene_mngr, goal_pose,joint_path):        #rrt로 path 생성
        planner = RRTStarPlanner(delta_distance=0.05, epsilon=0.2, gamma_RRT_star=2, dimension = robot.arm_dof)
        planner.run(scene_mngr=scene_mngr, cur_q=joint_path[-1], goal_pose=goal_pose, max_iter=300)
        joint_path += planner.get_joint_path(n_step=10)

    def update_cartesian_path(scene_mngr,goal_pose,joint_path):        #cartesian path 생성
        planner = CartesianPlanner(n_step = 10, dimension = robot.arm_dof)
        planner.run(scene_mngr=scene_mngr, cur_q=joint_path[-1], goal_pose=goal_pose)
        joint_path += planner.get_joint_path()
         

    #path 초기값
    joint_path =[init_qpos]         #[[qpos(7 joint angles)],...]
    joint_pathes = OrderedDict()    #OrderedDict([('task',[joint_path]),... ])
    pathes_num = 0                  #


    #타겟 위치를 통해 grasp pose 설정(z)
    init_pose = scene_mngr.get_robot_eef_pose()                     #로봇 초기 eef pose 저장
    target_yaw = get_rpy_from_matrix(current_pose.h_mat)[2]          #타겟 현재 yaw 불러오기
    grasp_pose = current_pose.h_mat                                  
    r_mat = get_matrix_from_rpy(np.array([0, np.pi, target_yaw]))   #수직으로 잡기위해 p=pi
    grasp_pose[:3, :3] = r_mat                                      #yaw는 타겟과 동일
    grasp_pose[:3, 3] = grasp_pose[:3, 3] + [0, 0, 0.2]             #pos는 z만 약간 더 높게 설정

    cartesian_dis_mat = np.zeros((4,4))
    cartesian_dis_mat[2,3] = 0.3

    pre_grasp_pose = grasp_pose + cartesian_dis_mat

    #pre_grasp
    print("pre grasp")
    update_rrt_path(scene_mngr, pre_grasp_pose, joint_path)
    joint_pathes.update({"pre_grasp" : joint_path[pathes_num:-1]})
    pathes_num = len(joint_path)+1

    #grasp
    print("grasp")
    update_cartesian_path(scene_mngr, grasp_pose, joint_path)
    joint_pathes.update({"grasp" : joint_path[pathes_num:-1]})
    pathes_num = len(joint_path)+1
    scene_mngr.set_robot_eef_pose(joint_path[-1])
    scene_mngr.attach_object_on_gripper("target")                   #씬메니져 충돌감지를 위한 attach
    

    #post_grasp - 초기 위치로 돌아가지만 물체 잡을 때의 회전 상태는 유지
    default_pose=init_pose + np.zeros((4,4))
    default_pose[:3, :3] = r_mat


    print("post_grasp")
    update_cartesian_path(scene_mngr, pre_grasp_pose, joint_path)
    joint_pathes.update({"post_grasp" : joint_path[pathes_num:-1]})
    pathes_num = len(joint_path)+1


    print("default_grasp_pose")
    update_rrt_path(scene_mngr, default_pose, joint_path)
    joint_pathes.update({"default_grasp_pose" : joint_path[pathes_num:-1]})
    pathes_num = len(joint_path)+1


    target_yaw = get_rpy_from_matrix(rotate_pose.h_mat)[2]
    r_mat = get_matrix_from_rpy(np.array([0, np.pi, target_yaw]))
    release_pose = rotate_pose.h_mat
    release_pose[:3,:3] = r_mat 
    release_pose[:3, 3] = grasp_pose[:3, 3]

    pre_release_pose = release_pose + cartesian_dis_mat

    print("pre_release")
    update_rrt_path(scene_mngr, pre_release_pose, joint_path)
    joint_pathes.update({"pre_release" : joint_path[pathes_num:-1]})
    pathes_num = len(joint_path)+1


    print("release")
    update_cartesian_path(scene_mngr, release_pose, joint_path)
    joint_pathes.update({"release" : joint_path[pathes_num:-1]})
    pathes_num = len(joint_path)+1


    #씬메니져 충돌감지를 위한 dettach, 씬에서 제거된 물체 다시 추가
    scene_mngr.detach_object_from_gripper("target")
    scene_mngr.add_object(
        name="target",
        gtype="mesh",
        gparam=red_cube_mesh,
        h_mat=rotate_pose.h_mat,
        color=[1.0, 0.0, 0.0],
    )

    #post_release
    print("post_release")
    update_cartesian_path(scene_mngr, pre_release_pose, joint_path)
    joint_pathes.update({"post_release" : joint_path[pathes_num:-1]})
    pathes_num = len(joint_path)+1

    #back_init_pose - 초기 위치로 돌아가기
    print("back_init_pose")
    update_rrt_path(scene_mngr, init_pose, joint_path)
    joint_pathes.update({"post_release" : joint_path[pathes_num:-1]})
    pathes_num = len(joint_path)+1


    global_move_time = 6
    local_move_time = 3

    #초기 위치로 이동
    operate_gripper(0)
    init_pos = np.array([ 0, 0, np.pi/2, 0, np.pi/2, 0]).reshape(-1).tolist()
    operate_robot(init_pos,global_move_time)

    for task, value in joint_pathes.items():
        print(task)
        joint_path = np.array(value).reshape(-1).tolist()
        operate_robot(joint_path,local_move_time)
        if task == "grasp":
            operate_gripper(0.7)
        elif task == "release":
            operate_gripper(0)
    return 0
