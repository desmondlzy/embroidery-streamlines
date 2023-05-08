# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import List, Union
import itertools
from operator import xor
from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
import triangle
from scipy import sparse
from tqdm import tqdm

from embroidery.utils import math, rng
from embroidery.utils.generator import pairwise

@dataclass
class _SpringCurve:
	point_indices: "list[int]"  # index to points
	spring_point_indices_to_point_indices: "list[int]" # index to point_indices

	def spring_point_indices(self):
		return [self.point_indices[ind] for ind in self.spring_point_indices_to_point_indices]


@dataclass
class _EdgeAttribute:
	ideal_length = 0
	d: np.ndarray
	on_the_same_curve: bool
	effective: int

@dataclass
class SpringEnergyComponents:
	Q_align: sparse.spmatrix
	Q_space: sparse.spmatrix
	Q_dirreg: sparse.spmatrix
	Q_reg: sparse.spmatrix
	Q_fix: sparse.spmatrix
	b_align: Union[sparse.spmatrix, np.ndarray]
	b_space: Union[sparse.spmatrix, np.ndarray]
	b_dirreg: Union[sparse.spmatrix, np.ndarray]
	b_reg: Union[sparse.spmatrix, np.ndarray]
	b_fix: Union[sparse.spmatrix, np.ndarray]
	curves: List[np.ndarray]

@dataclass
class _Triplet:
	xs: List[int] = field(default_factory=list)
	ys: List[int] = field(default_factory=list)
	vs: List[float] = field(default_factory=list)

	def add(self, i, j, v):
		self.xs.append(i)
		self.ys.append(j)
		self.vs.append(v)


def spring_energy_matrices(
	streamlines, 
	line_width,
	z_func,
	boundary, 
	density_func,
	inside_indicator_func, 
	spring_density_threshold=0.2, 
	plot=False) -> SpringEnergyComponents:

	ideal_stitch_dist = line_width

	streamline_interp = [math.arclength_parametrization(l) for l in streamlines]
	streamline_lengths = [math.total_length(l) for l in streamlines]
	curves = [f(np.linspace(0 + 0.000001, l - 0.0000001, num=max(int(np.round(l / ideal_stitch_dist)), 2))) for f, l in zip(streamline_interp, streamline_lengths)]  # if a curve is shorter than two stitches, discard it.


	if boundary is not None:
		print("adding boundary points")
		boundary_interp = math.arclength_parametrization(boundary)
		boundary_length = math.total_length(boundary)
		num_subd = max(int(np.round(boundary_length / ideal_stitch_dist)), 2)
		resampled_boundary = boundary_interp(np.linspace(0, boundary_length - 0.0001, num=num_subd, endpoint=False))
		curves.append(resampled_boundary)

	points = np.vstack(curves)
	print(f"{points.shape = }")

	spring_curves = []

	# this loop translate curves from bunch of points into a indices hierarchy
	point_index_offset = 0
	for _, crv in enumerate(curves):
		l = len(crv)

		# control how many points sampled as the spring points
		ind = np.arange(l)
		rng.shuffle(ind)
		ind = np.sort(ind[:min(l, l)])
		# this_group = ind + point_index_offset

		spring_curves.append(
			_SpringCurve(point_indices=[i + point_index_offset for i in range(l)],
						spring_point_indices_to_point_indices=ind))

		point_index_offset += l

	spring_point_indices = np.concatenate([sprcrv.spring_point_indices() for sprcrv in spring_curves]).astype(int)
	spring_points = points[spring_point_indices]

	streamline_endpoint_set = set(crv.point_indices[0] for crv in spring_curves).union(set(crv.point_indices[-1] for crv in spring_curves))

	segment_endpoints = []
	for sprcrv in spring_curves:
		segment_endpoints.extend(
			[[p, q] for p, q in pairwise(sprcrv.spring_point_indices())]
		)

	connected_point_indices = set(
		tuple(pair) for pair in segment_endpoints
	) 

	tri_results = triangle.triangulate({
		"vertices": spring_points,
		"segments": np.array(segment_endpoints),
		}, "cpen")

	# new extra_points will be appended to the end of original_points
	num_extra_points = len(tri_results["vertices"]) - len(spring_points)
	print(f"delaunay finished: #V = {len(tri_results['vertices'])}, #E = {len(tri_results['edges'])}, #F = {len(tri_results['triangles'])}")
	print("triangle.triangulate add new vertices?", len(tri_results["vertices"]), len(spring_points), num_extra_points, len(tri_results["vertices"]) == len(spring_points))

	spring_points = tri_results["vertices"]
	edge_data = tri_results["edges"]

	# start setting up energy
	nV = len(spring_points)
	nE = len(edge_data)
	nDoF = 2 * nV

	print(f"preprocessing the edge attributes")
	# in case if spring system has less points than the points we are considering
	spring_system_point_index_map = { i: i for i in range(nV) }

	# precondition
	edge_attributes = []
	num_same_curve = 0
	for edge_ind, (p_ind, q_ind) in tqdm(enumerate(edge_data)):
		# this is very slow
		# on_the_same_curve = any(p_ind in sprcrv.point_indices and q_ind in sprcrv.point_indices for sprcrv in spring_curves)
		on_the_same_curve = (p_ind, q_ind) in connected_point_indices or (q_ind, p_ind) in connected_point_indices
		if on_the_same_curve:
			effective = 0

		else:
			pt_p, pt_q = spring_points[p_ind], spring_points[q_ind]
			norm_p, norm_q = density_func(*pt_p), density_func(*pt_q)
			# pq = pt_p - pt_q
			# dir_p, dir_q = dir_z(pt_p), dir_z(pt_q)
			effective = 0
			# if np.abs(norm_z(pt_p) - norm_z(pt_q)) > 0.1:
			# 	effective = 3
			
			if norm_p < spring_density_threshold or norm_q < spring_density_threshold:
				effective = 4

			# if np.abs(norm_z(pt_p)) > 1 or np.abs(norm_z(pt_q)) > 1:
			# 	effective = 4

			p_is_endpoint, q_is_endpoint = p_ind in streamline_endpoint_set, q_ind in streamline_endpoint_set
			if p_is_endpoint and q_is_endpoint: 
				effective = 5
			
			if boundary is not None and xor(p_is_endpoint, q_is_endpoint):
				mid_points = np.linspace(pt_p, pt_q, 5)[1: -1]  # sample some points on the edge
				if not np.all(inside_indicator_func(mid_points[:, 0], mid_points[:, 1])):  # if not all the points inside the boundary
					effective = 6 
			
			if boundary is not None and (p_ind >= nV - len(curves[-1]) or q_ind >= nV - len(curves[-1])):
				effective = 7

		edge_attributes.append(_EdgeAttribute(d=np.empty(0), on_the_same_curve=on_the_same_curve, effective=effective))


	if plot:
		for edge_ind, ((p_ind, q_ind), attr) in tqdm(enumerate(zip(edge_data, edge_attributes))):
			pt_p, pt_q = spring_points[p_ind], spring_points[q_ind]
			if not attr.on_the_same_curve:
				plt.plot([pt_p[0], pt_q[0]], [pt_p[1], pt_q[1]], color="green")
		for c in curves:
			plt.plot(c[:, 0], c[:, 1], color="#1553b7ff")
		plt.axis("scaled")
		plt.savefig("figures/delaunay.svg", bbox_inches="tight", pad_inches=0)
		plt.show()


	def _smoothing_distance_func(xk):
		return line_width / density_func(*xk)

	x0 = spring_points.reshape(-1)

	b_sys_align = np.zeros((nDoF, ), dtype=float)
	tri_align = _Triplet()
	
	b_sys_space = np.zeros((nDoF, ), dtype=float)
	tri_space = _Triplet()

	b_sys_dirreg = np.zeros((nDoF, ), dtype=float)
	tri_dirreg = _Triplet()

	Q_sys_reg = sparse.eye(nDoF, dtype=float, format="coo")
	b_sys_reg = -2 * x0

	# Q_sys_fixed = sparse.dok_matrix((nDoF, nDoF), dtype=float)
	# b_sys_fixed = np.zeros_like(x0)

	print("start build stiffness matrix")
	num_same_curve = 0
	num_ineffective = 0
	for edge_ind, ((p_ind, q_ind), attr) in tqdm(enumerate(zip(edge_data, edge_attributes))):
		p, q = spring_points[p_ind], spring_points[q_ind]
		z_p, z_q = z_func(p), z_func(q)
		
		# map the element stiffness matrix to the globals
		ps_ind, qs_ind = spring_system_point_index_map[p_ind], spring_system_point_index_map[q_ind]
		def _add_ele_to_triplet(el, triplet: _Triplet):
			for (sys_a, sys_b), (ele_a, ele_b) in zip(itertools.product((ps_ind, qs_ind), repeat=2), itertools.product(range(2), repeat=2)):
				triplet.add(sys_a * 2, sys_b * 2, el[ele_a * 2, ele_b * 2])
				triplet.add(sys_a * 2 + 1, sys_b * 2, el[ele_a * 2 + 1, ele_b * 2])
				triplet.add(sys_a * 2, sys_b * 2 + 1, el[ele_a * 2, ele_b * 2 + 1])
				triplet.add(sys_a * 2 + 1, sys_b * 2 + 1, el[ele_a * 2 + 1, ele_b * 2 + 1])

		# only consider inter-curve edges
		if attr.on_the_same_curve:
			num_same_curve += 1
			attr.ideal_length = norm(p - q)

			e = q - p
			ex, ey = e

			Q_ele_align = np.array([
				[1, 0, -1, 0],
				[0, 1, 0, -1],
				[-1, 0, 1, 0],
				[0, -1, 0, 1],
			])

			b_ele_align = -2 * np.array([
				-e[0],
				-e[1],
				e[0],
				e[1],
			])

			Q_ele_dirreg = np.array([
				[ex ** 2, ex * ey, 0, 0],
				[ex * ey, ey ** 2, 0, 0],
				[0, 0, ex ** 2, ex * ey],
				[0, 0, ex * ey, ey ** 2],
			])

			b_ele_dirreg = 2 * np.array([
				-ex ** 2 * p[0] - ex * ey * p[1],
				-ex * ey * p[0] - ey ** 2 * p[1],
				-ex ** 2 * q[0] - ex * ey * q[1],
				-ex * ey * q[0] - ey ** 2 * q[1],
			])

			# _add_ele_to_sys(Q_ele_align, Q_sys_align)
			_add_ele_to_triplet(Q_ele_align, tri_align)
			b_sys_align[ps_ind * 2: ps_ind * 2 + 2] += b_ele_align[0: 2]
			b_sys_align[qs_ind * 2: qs_ind * 2 + 2] += b_ele_align[2: 4]

			# _add_ele_to_sys(Q_ele_dirreg, Q_sys_dirreg)
			_add_ele_to_triplet(Q_ele_dirreg, tri_dirreg)
			b_sys_dirreg[ps_ind * 2: ps_ind * 2 + 2] += b_ele_dirreg[0: 2]
			b_sys_dirreg[qs_ind * 2: qs_ind * 2 + 2] += b_ele_dirreg[2: 4]


		else:
			if attr.effective != 0:  # not effective
				num_ineffective += 1
				continue
			# t: rest length (target value) and d: normal direction, are local to the element
			d = 0.5 * (z_p / norm(z_p)) + 0.5 * (z_q / norm(z_q))
			d = math.rotate90(d)

			t = (_smoothing_distance_func(p) + _smoothing_distance_func(q)) / 2
			t *= np.sign(np.dot(q - p, d))

			attr.ideal_length = t

			# element stiffness matrix
			dx, dy = d
			dx2, dy2, dxy = dx ** 2, dy ** 2, dx * dy
			Q_ele_space = np.array([
				[dx2, dxy, -dx2, -dxy],
				[dxy, dy2, -dxy, -dy2],
				[-dx2, -dxy, dx2, dxy],
				[-dxy, -dy2, dxy, dy2],
			]) 
			b_ele_space = np.array([
				2 * t * dx,
				2 * t * dy,
				-2 * t * dx,
				-2 * t * dy,
			]) 

			# _add_ele_to_sys(Q_ele_space, Q_sys_space)
			_add_ele_to_triplet(Q_ele_space, tri_space)
			b_sys_space[ps_ind * 2: ps_ind * 2 + 2] += b_ele_space[0: 2]
			b_sys_space[qs_ind * 2: qs_ind * 2 + 2] += b_ele_space[2: 4]
			# c += t ** 2 * w_space

	print("matrix assembling done")
	print(f"{num_ineffective = } / #E = {nE}")

	Q_sys_align = sparse.coo_matrix((tri_align.vs, (tri_align.xs, tri_align.ys)), shape=(nDoF, nDoF))
	Q_sys_dirreg = sparse.coo_matrix((tri_dirreg.vs, (tri_dirreg.xs, tri_dirreg.ys)), shape=(nDoF, nDoF))
	Q_sys_space = sparse.coo_matrix((tri_space.vs, (tri_space.xs, tri_space.ys)), shape=(nDoF, nDoF))

	if boundary is not None:
		num_fixed_points = len(curves[-1]) + num_extra_points
		Q_fix_last_n_points = sparse.coo_matrix(
			(np.ones((2 * num_fixed_points, )), 
			(
				np.arange(num_fixed_points * 2) + (len(spring_points) - num_fixed_points) * 2, 
				np.arange(num_fixed_points * 2) + (len(spring_points) - num_fixed_points) * 2, 
			)),
			shape=(nDoF, nDoF))

		b_fix_last_n_points = -2 * np.copy(x0)
		b_fix_last_n_points[0: -num_fixed_points * 2] = 0.0

	else:
		Q_fix_last_n_points = sparse.coo_matrix(([], ([], [])), shape=(nDoF, nDoF), dtype=float)
		b_fix_last_n_points = np.zeros_like(x0, dtype=float)
	
	components = SpringEnergyComponents(
		Q_align=Q_sys_align, 
		Q_space=Q_sys_space,
		Q_reg=Q_sys_reg,
		Q_fix=Q_fix_last_n_points,
		Q_dirreg=Q_sys_dirreg,
		b_align=b_sys_align, 
		b_space=b_sys_space,
		b_reg=b_sys_reg,
		b_fix=b_fix_last_n_points,
		b_dirreg=b_sys_dirreg,
		curves=curves,
		)
	
	return components
	

def weigh_matrices_and_solve(
	comps: SpringEnergyComponents, 
	w_align=1e5, 
	w_space=1e4, 
	w_dirreg=1e6, 
	w_reg=1e-4, 
	w_fix=1e8
	) -> np.ndarray:
	# weighted sum of things 
	Q = (
		w_align * comps.Q_align + 
		w_space * comps.Q_space + 
		w_dirreg * comps.Q_dirreg + 
		w_reg * comps.Q_reg + 
		w_fix * comps.Q_fix 
	)
	b = (
		w_align * comps.b_align + 
		w_space * comps.b_space + 
		w_dirreg * comps.b_dirreg + 
		w_reg * comps.b_reg + 
		w_fix * comps.b_fix 
	)

	# solve
	if sparse.issparse(Q):
		x_sol = sparse.linalg.spsolve(Q.tocsr() * 2, -b)
	else:
		x_sol = np.linalg.solve(Q * 2, -b)

	opt_points = x_sol.reshape(-1, 2)

	lines = map_points_to_curves(opt_points, comps.curves)

	return lines


def map_points_to_curves(points, curves):
	new_curves = []
	for i, offset in enumerate(np.cumsum([len(c) for c in curves])):
		new_c = points[offset - len(curves[i]): offset]
		new_curves.append(new_c)

	return new_curves
