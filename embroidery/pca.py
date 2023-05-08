# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from sklearn.decomposition import PCA
from scipy import interpolate

from embroidery.utils.colors import cmy2rgb, rgb2cmy
from embroidery.utils import math, rng

from common.grid_interp import interpolate_rgb_image


def color_gradient_extraction_with_pca(
	image, 
	inside_indicator_func,  
	normal_space_to_image_space, 
	truncate_outside=True):

	image_interpolant = interpolate_rgb_image(image)
	color_func = lambda xk: image_interpolant(normal_space_to_image_space(xk))

	n_samples = 10000
	samples = rng.random((n_samples, 2))
	inside_mask = inside_indicator_func(samples[:, 0], samples[:, 1])
	# inside_mask = math.inside_boundary(boundary, samples, strict=False)
	samples_inside = samples[np.where(inside_mask)]
	rgb_inside = np.apply_along_axis(color_func, axis=1, arr=samples_inside)

	cmy = rgb2cmy(rgb_inside)
	cmy_mean = cmy.mean(axis=0)
	cmy_centered = cmy - cmy_mean

	pca = PCA(n_components=1)
	ts = pca.fit_transform(cmy_centered)
	comp = pca.components_[0]

	t_pc_min = np.percentile(ts, 10)
	t_pc_max = np.percentile(ts, 90)

	# remove the pixels that fall outside of the color space before parametrization
	# the fitted cmy line is 
	# 0 <= cmy_mean + t * comp <= 1
	t_hit = np.concatenate(((1 - cmy_mean) / comp, (0 - cmy_mean) / comp))
	hit_group_a = t_hit[np.nonzero(t_hit > 0)]
	hit_group_b = t_hit[np.nonzero(t_hit <= 0)]

	assert len(hit_group_a) == 3 and len(hit_group_b) == 3
	assert np.all(hit_group_a > 0), f"{cmy_mean = }, {comp = }, {hit_group_a = }, {hit_group_b = }"

	t_hit_max = hit_group_a.min()
	t_hit_min = hit_group_b.max()

	t_min = max(t_pc_min, t_hit_min)
	t_max = min(t_pc_max, t_hit_max)

	rgb_a, rgb_b = cmy2rgb(t_min * comp + cmy_mean), cmy2rgb(t_max * comp + cmy_mean)

	t_image = pca.transform(rgb2cmy(image).reshape(-1, 3) - cmy_mean)
	normalized_ts = np.clip(
		math.map_range(t_image, t_min, t_max, 0, 1),
		0, 1)

	return normalized_ts.reshape(*image.shape[:2]), rgb_a, rgb_b
