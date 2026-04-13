# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Real2sim workflow environments.
"""

import gymnasium as gym

import pathlib

_CURRENT_FILE_PATH = pathlib.Path(__file__).absolute()
REAL2SIM_ROOT_DIR = _CURRENT_FILE_PATH.parents[0] 
