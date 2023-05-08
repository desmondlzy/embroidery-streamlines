# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from itertools import product

from embroidery.utils import math


def poisson_disc(radius_func, dim=2, num_candidates=10000, random_func=np.random.random, additional_rejecter=None):
	if additional_rejecter is None:  
		additional_rejecter = lambda p: False  # always don't reject

	candidates = random_func((num_candidates, dim))
	accepted_candidates = np.empty_like(candidates) 
	accepted_candidates_radii = np.empty((num_candidates, ))

	accepted_candidates[0] = candidates[0]
	accepted_candidates_radii[0] = radius_func(candidates[0])
	num_accepted = 1

	for p in candidates[1:]:
		distances = np.linalg.norm(p - accepted_candidates[:num_accepted], axis=1) 
		p_radius = radius_func(p)
		if np.all(distances > accepted_candidates_radii[:num_accepted]) and np.all(distances > p_radius):
			accepted_candidates[num_accepted] = p
			accepted_candidates_radii[num_accepted] = p_radius
			num_accepted += 1

	return accepted_candidates[:num_accepted]


def poisson_disc_in_boundary(radius_func, dim=2, num_candidates=10000, random_func=np.random.random, boundary_points=None):
	assert boundary_points is not None

	all_candidates = random_func((num_candidates, dim))
	boundary_mask = math.inside_boundary(boundary_points, all_candidates)
	candidates = all_candidates[boundary_mask]
	accepted_candidates = np.empty_like(candidates) 
	accepted_candidates_radii = np.empty((num_candidates, ))

	accepted_candidates[0] = candidates[0]
	accepted_candidates_radii[0] = radius_func(candidates[0])
	num_accepted = 1

	for p in candidates[1:]:
		distances = np.linalg.norm(p - accepted_candidates[:num_accepted], axis=1) 
		p_radius = radius_func(p)
		if np.all(distances > accepted_candidates_radii[:num_accepted]) and np.all(distances > p_radius):
			accepted_candidates[num_accepted] = p
			accepted_candidates_radii[num_accepted] = p_radius
			num_accepted += 1

	return accepted_candidates[:num_accepted]



def jitter_sampling(pdf, window_num, rng):
	window_size = 1 / window_num

	samples = []

	for x_offset, y_offset in product(
		np.linspace(0, 1, window_num, endpoint=False), 
		np.linspace(0, 1, window_num, endpoint=False)):

		integral = math.gauss_legendre_integration_on_grid(
			pdf, 
			x_offset, x_offset + window_size,
			y_offset, y_offset + window_size,
			deg=5)

		expected_point_num = np.abs(integral)
		fraction, integer = np.modf(expected_point_num)
		actual_point_num = int(integer + rng.binomial(1, fraction))
		cell_samples = rng.random(size=(actual_point_num, 2)) * np.array([window_size, window_size]) + np.array([x_offset, y_offset])
		samples.append(cell_samples)

	return np.vstack(samples)
