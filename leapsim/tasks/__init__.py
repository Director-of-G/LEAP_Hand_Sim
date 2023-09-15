# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------


from .leap_hand_rot import LeapHandRot
from .leap_hand_grasp import LeapHandGrasp

# Mappings from strings to environments
isaacgym_task_map = {
    "LeapHandGrasp": LeapHandGrasp,
    "LeapHandRot": LeapHandRot,
}
