# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from scipy import interpolate
import warnings

from embroidery.utils import math, rng


def sample_points_on_interior(signal_interpolant, init_xl, init_xr, init_yl, init_yr, com_random_coeff=0.0):
	"""
	KD-tree points generation with in a given rectangle from a given signal.
	when decide to put the point if the integral evaluated to 1, place at the center of mass -- random deviation allowed by `com_random_coeff`
	"""
	assert 0 <= com_random_coeff <= 1, f"{com_random_coeff=} must be in [0, 1]"
	# plt.plot(boundary[:, 0], boundary[:, 1])
	def integrate_on_grid(x0, x1, y0, y1):
		return signal_interpolant.integral(x0, x1, y0, y1)

	def find_split(target, x_l, x_r, y_l, y_r, in_x):
		assert integrate_on_grid(x_l, x_r, y_l, y_r) > target

		if in_x:
			# find xnew such that integrate_on_grid(x_l, x_new, y_l, y_r) = target / 2 
			guess_range = (x_l, x_r)
			evaluate_guess = lambda g: integrate_on_grid(x_l, g, y_l, y_r)
		else:
			# find ynew such that integrate_on_grid(x_l, y_l, x_r, y_new) = target / 2 
			guess_range = (y_l, y_r)
			evaluate_guess = lambda g: integrate_on_grid(x_l, x_r, y_l, g)

		while True:
			guess = (guess_range[0] + guess_range[1]) / 2
			guess_integral = evaluate_guess(guess)
			if np.isclose(guess_integral, target, atol=1e-5):  # guess is right
				ans = guess
				break
			elif guess_integral > target:
				guess_range = (guess_range[0], guess)
			elif guess_integral < target:
				guess_range = (guess, guess_range[1])
		
		return ans


	def compute_samples(target, x_l, x_r, y_l, y_r):
		if target == 1:
			# return the center of mass
			com = 0.5 + rng.random() * com_random_coeff
			x_new = find_split(com, x_l, x_r, y_l, y_r, in_x=True)
			y_new = find_split(1 - com, x_l, x_r, y_l, y_r, in_x=False)
			return [(x_new, y_new)]

		target_left = target // 2
		target_right = target - target_left

		x_len = x_r - x_l
		y_len = y_r - y_l

		if x_len >= y_len:
			x_new = find_split(target_left, x_l, x_r, y_l, y_r, in_x=True)
			samples_left = compute_samples(target_left, x_l, x_new, y_l, y_r)
			samples_right = compute_samples(target_right, x_new, x_r, y_l, y_r)
		else:
			y_new = find_split(target_left, x_l, x_r, y_l, y_r, in_x=False)
			samples_left = compute_samples(target_left, x_l, x_r, y_l, y_new)
			samples_right = compute_samples(target_right, x_l, x_r, y_new, y_r)
		
		return [*samples_left, *samples_right]

	total_integral = integrate_on_grid(init_xl, init_xr, init_yl, init_yr)

	if total_integral < 1:
		warnings.warn(f"{total_integral = } < 1, no points are sampled")
		return np.empty((0, 2), dtype=float)

	total_target = int(np.round(total_integral))
	print(f"to sample {total_integral = }, {total_target = }")
	samples = np.array(compute_samples(total_target, init_xl, init_xr, init_yl, init_yr))

	return samples


def sample_points_on_line(signal_interpolant, init_t0, init_t1):
	# plt.plot(boundary[:, 0], boundary[:, 1])
	def integrate_on_line(t0, t1):
		return signal_interpolant.integral(t0, t1)

	def find_split(target, t0, t1):
		assert integrate_on_line(t0, t1) > target
		guess_range = (t0, t1)
		while True:
			guess = (guess_range[0] + guess_range[1]) / 2
			guess_integral = integrate_on_line(t0, guess)
			if np.isclose(guess_integral, target, atol=1e-5):  # guess is right
				ans = guess
				break
			elif guess_integral > target:
				guess_range = (guess_range[0], guess)
			elif guess_integral < target:
				guess_range = (guess, guess_range[1])
		
		return ans

	def compute_samples(target, t0, t1):
		if target == 1:
			# return the center of mass
			t_new = find_split(0.5, t0, t1)
			return [t_new]

		target_left = target // 2
		target_right = target - target_left

		t_new = find_split(target_left, t0, t1)
		samples_left = compute_samples(target_left, t0, t_new)
		samples_right = compute_samples(target_right, t_new, t1)
		
		return [*samples_left, *samples_right]

	total_integral = integrate_on_line(init_t0, init_t1)
	total_target = int(np.round(total_integral))
	print(f"to sample {total_integral = }, {total_target = }")
	samples = np.array(compute_samples(total_target, init_t0, init_t1))

	return samples


def sample_sources_on_closed_boundary(z_func, boundary, line_width):
	parametrized_boundary, boundary_length = math.periodic_arclength_parametrization(boundary)
	ts = np.linspace(0, boundary_length, num=int(boundary_length / 0.01))
	ns = math.curve_unit_normal(parametrized_boundary, ts)
	pts = parametrized_boundary(ts)
	zs = np.apply_along_axis(z_func, axis=1, arr=pts)
	vals = np.einsum("ij,ij->i", ns, zs)  # row-wise dot product
	pos_vals = np.maximum(vals, 0)
	boundary_signal_interpolant = interpolate.UnivariateSpline(ts, pos_vals * (1 / line_width), k=1) 
	if boundary_signal_interpolant.integral(0, boundary_length) < 1:
		return np.empty((0, 2), dtype=float)

	sampled_ts = sample_points_on_line(boundary_signal_interpolant, 0, boundary_length)
	boundary_sources = parametrized_boundary(sampled_ts)

	# push things inwards to prevent unwanted truncation of line due to close to boundary
	deviations = math.curve_unit_normal(parametrized_boundary, sampled_ts) * 0.002  
	# deviations = np.apply_along_axis(z_func, axis=1, arr=parametrized_boundary(sampled_ts)) * 0.002
	return boundary_sources + deviations
