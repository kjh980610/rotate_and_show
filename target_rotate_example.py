#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import target_rotate
from pykin.kinematics.transform import Transform
import numpy as np


current_pose = Transform(pos=np.array([0,0,0]), rot=np.array([0.0, 0.0, 0.0]))
target_pose = Transform(pos=np.array([0,0,0]), rot=np.array([0.0, 0.0, np.pi]))

target_rotate.target_rotate(current_pose,target_pose)
