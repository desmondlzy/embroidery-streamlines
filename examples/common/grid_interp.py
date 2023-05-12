# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from scipy import interpolate

def interpolate_rgb_image(image):
	channels = [
		interpolate.interp2d(
			np.arange(image.shape[1]), 
			np.arange(image.shape[0]),
			image[:, :, i].reshape(image.shape[0], image.shape[1]))
		for i in range(3)
	] 

	def color_density(xk):
		values = np.stack(([channels[i](xk[0], xk[1]) for i in range(3)]), axis=-1)
		if values.size == 3:
			return values.ravel()
		else:
			return values
	
	return color_density

def interpolate_grid_values(grid_values):
	return interpolate.interp2d(
		np.arange(grid_values.shape[1]), 
		np.arange(grid_values.shape[0]),
		grid_values
	)