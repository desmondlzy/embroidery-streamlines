# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import functools
import numpy as np
from numpy.linalg import norm
from embroidery.utils.math import rotation_matrix_2d, rowwise_rotate, map_range, divergence_fd, jacobian_fd, rotate90
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb

primary_color = "#1553b7ff"
secondary_color = "#ff6a00ff"

def primary_colormap():
	r, g, b = to_rgb(primary_color)
	color_n = 256
	vals = np.ones((color_n, 4))
	vals[:, 0] = np.linspace(1, r, color_n)
	vals[:, 1] = np.linspace(1, g, color_n)
	vals[:, 2] = np.linspace(1, b, color_n)
	newcmp = ListedColormap(vals)

	return newcmp

def plot_vector_function_within_unit_square(z, boundary_points=None):
	"""
	plot a vector function z in [0, 1]^2: arrow plot, divergence map, length map
	"""
	def jacobian_z(xk):
		jac = jacobian_fd(xk, z, dim=2)
		return jac

	def norm_grad(xk):
		grad = np.ravel(jacobian_fd(xk, f=lambda xk: [norm(z(xk))], dim=1))
		return grad
	
	def norm_grad_on_tangent(xk):
		tangent = z(xk)
		tangent /= norm(tangent)
		n = norm_grad(xk) 
		return np.dot(n, tangent)

	def norm_grad_on_normal(xk):
		tangent = z(xk)
		tangent /= norm(tangent)
		n = norm_grad(xk) 
		return np.dot(n, rotate90(tangent))
		

	def div_z(xk):
		jac = jacobian_z(xk)
		return jac[0, 0] + jac[1, 1]

	def curl_z(xk):
		jac = jacobian_z(xk)
		return jac[1, 0] - jac[0, 1]
	
	def first_direction_norm(xk):
		jac = jacobian_z(xk)
		n = norm(z(xk))
		return norm(jac @ z(xk) / n)

	def second_direction_norm(xk):
		jac = jacobian_z(xk)
		n = norm(z(xk))
		return norm(jac @ rotate90(z(xk)) / n)
	

	nx, ny = 80, 80
	x = np.linspace(0.0, 1, nx)
	y = np.linspace(0.0, 1, ny)
	xv, yv = np.meshgrid(x, y)
	xyv = np.dstack([xv, yv])

	quiver_x = np.linspace(0.0, 1, 10)
	quiver_y = np.linspace(0.0, 1, 10)
	quiver_xv, quiver_yv = np.meshgrid(quiver_x, quiver_y)
	quiver_xyv = np.dstack([quiver_xv, quiver_yv])
	quiver_z = np.apply_along_axis(z, 2, quiver_xyv)

	zv = np.apply_along_axis(z, 2, xyv)
	zv_norm = np.linalg.norm(zv, axis=2)
	zv_div = np.apply_along_axis(div_z, 2, xyv)
	zv_curl = np.apply_along_axis(curl_z, 2, xyv)
	zv_first = np.apply_along_axis(norm_grad_on_tangent, 2, xyv)
	zv_second = np.apply_along_axis(norm_grad_on_normal, 2, xyv)
	# zv_div = np.array([divergence_fd(np.array([x, y]), z) for x, y in zv.reshape(-1, 2)]).reshape(nx, ny).round(4)

	invgray = ListedColormap(
		np.vstack((
			np.linspace(np.array([1.0, 1.0, 1.0, 1]), np.array([0, 0, 0, 1]), 256, endpoint=True),
		)))

	fig, ax = plt.subplots(1, 3, sharex='all', sharey='all')

	a = ax[0]
	a.set_title(r"$Z$")
	a.quiver(quiver_xv, quiver_yv, quiver_z.reshape(-1, 2)[:, 0], quiver_z.reshape(-1, 2)[:, 1])

	# a = ax[0, 1]
	# a.set_title(r"curl $ Z$" + f": {zv_curl.min().round(4), zv_curl.max().round(4)}")
	# # a.quiver(xv, yv, zv.reshape(-1, 2)[:, 0], zv.reshape(-1, 2)[:, 1])
	# curl_plot = a.contourf(x, y, zv_curl.round(4), levels=30)

	a = ax[1]
	a.set_title(r"$div Z$" + f": {zv_div.min().round(4), zv_div.max().round(4)}")
	div_plot = a.contourf(x, y, zv_div.round(4), levels=np.linspace(-2, 2, num=20))

	a = ax[2]
	a.contourf(x, y, zv_norm, cmap=invgray, levels=20)
	a.set_title(r"$\|Z\|$" + f"{zv_norm.min().round(4), zv_norm.max().round(4)}")

	# a = ax[1, 1]
	# a.contourf(x, y, zv_first.round(4), levels=40)
	# a.set_title(r"$grad \|Z\| \cdot t$" + f" : {zv_first.min().round(4), zv_first.max().round(4)}")

	# a = ax[1, 2]
	# a.contourf(x, y, zv_second.round(4), levels=40)
	# a.set_title(r"$grad \|Z\| \cdot n$" + f" : {zv_second.min().round(4), zv_second.max().round(4)}")

	for a in np.ravel(ax):
		a.axis("scaled")
	if isinstance(boundary_points, np.ndarray):
		for a in np.ravel(ax):
			a.plot(boundary_points[:, 0], boundary_points[:, 1])
	
	fig.tight_layout()

	plt.show()

def save_plot_without_axes(filename):
	plt.gca().set_xticks([])
	plt.gca().set_yticks([])
	plt.savefig(filename, transparent=True, pad_inches=0, bbox_inches="tight")