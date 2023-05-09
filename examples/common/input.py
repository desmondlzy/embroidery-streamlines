# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import functools
from dataclasses import dataclass
import numpy as np
from numpy.linalg import norm
import matplotlib as mpl
from embroidery.utils import math
from embroidery.utils.math import map_range

from typing import Callable

np.set_printoptions(precision=4, linewidth=10000)

def matplotlib_settings():
	mpl.rcParams["figure.figsize"] = (12, 12)
	mpl.rcParams["figure.dpi"] = 100

matplotlib_settings()
