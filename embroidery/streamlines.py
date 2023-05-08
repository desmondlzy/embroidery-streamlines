# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from numpy.linalg import norm
from scipy.spatial import Delaunay, KDTree
from scipy import sparse
import networkx
from tqdm import tqdm

from embroidery.utils import math, rng
from embroidery.utils.generator import pairwise

from dataclasses import dataclass

import triangle

import matplotlib.pyplot as plt


def _default_boundary_checker(pt):
	return np.all(pt >= 0) and np.all(pt <= 1)


def trace_streamline(z, start, step=100, step_size=0.01, boundary_checker=None):
	if boundary_checker is None:
		boundary_checker = _default_boundary_checker

	points = []
	for i in range(step):
		d = z(start)
		d_norm = np.linalg.norm(d)
		if d_norm == 0:
			break
		d /= d_norm

		mid = start + d * step_size * 0.5

		d_mid = z(mid)
		d_mid_norm = np.linalg.norm(d_mid)
		if d_mid_norm == 0:
			break

		d_mid /= d_mid_norm
		# p = div_z(start)

		points.append(start)

		if not (in_boundary := boundary_checker(start)):
			break

		# start = start + step_size * d  # euler
		start = start + step_size * d_mid  # runge kutta
	
	if len(points) >= 2:
		return np.row_stack(points)
	else:
		return None

def consecutive(data, stepsize=1):
	# https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def cut_streamlines_assignment(streamlines, sink_points, boundary_points):
	if len(sink_points) == 0:
		return [l for l in streamlines]

	assert len(sink_points) >= 1
	kdtrees = {f"l{i}" : KDTree(line) for i, line in enumerate(streamlines)}
	graph = networkx.Graph()
	graph.add_nodes_from([f"l{i}" for i in range(len(streamlines))], bipartite=0)
	graph.add_nodes_from([f"s{i}" for i in range(len(sink_points))], bipartite=1)
	for ln, tree in kdtrees.items():
		dd, ii = tree.query(sink_points)  # get point-line distance for line i
		graph.add_weighted_edges_from([(ln, f"s{si}", dd[si]) for si in range(len(sink_points))])
	
	matches = networkx.bipartite.minimum_weight_full_matching(graph) 

	cut_streamlines = []


	for ln, tree in kdtrees.items():
		line_ind = int(ln[1:])
		line = streamlines[line_ind]

		sn = matches.get(ln, None)
		if sn is None:
			cut_streamlines.append(np.copy(line))

		else:
			s_ind = int(sn[1:])
			sink = sink_points[s_ind]
			_, match_point_ind = kdtrees[ln].query(sink)  # this is repeated computation

			if match_point_ind in (0, 1):  # remove the whole line
				continue

			if match_point_ind in (len(line) - 1, ):  # match the rear, no trimming, avoid out-of-bound
				cut_streamlines.append(np.copy(line))

			else:
				trimmed_line = line[0: match_point_ind + 1]
				cut_streamlines.append(trimmed_line)

	return cut_streamlines



def crop_streamlines_outside_boundary(streamlines, inside_func):
	streamlines_in_boundary = []
	for line in streamlines:
		mask = inside_func(line[:, 0], line[:, 1])
		groups = math.consecutive(np.nonzero(mask)[0])
		if groups[0].size == 0:
			continue

		for group in groups:
			if group.size > 1:
				streamlines_in_boundary.append(line[group])

	return streamlines_in_boundary



def scale_and_resample(lines, scaling_factor, ideal_spacing):
	scaled_lines = []

	for line in lines:
		f = math.arclength_parametrization(line * scaling_factor)
		l = math.total_length(line * scaling_factor)
		num = max(2, int(l / ideal_spacing))
		resamples = f(np.linspace(0, l - 0.0001, num))
		scaled_lines.append(resamples)
	
	return scaled_lines


def truncate_streamline_by_indicator(streamline, seed_index, inside_indicator_func):
	assert seed_index == 0
	assert streamline is not None
	mask = inside_indicator_func(streamline[:, 0], streamline[:, 1]).round().astype(bool)
	# assert mask[seed_index] == True or mask[seed_index + 1] == True or mask[seed_index + 2] == True
	groups = math.consecutive(np.nonzero(mask)[0])
	for group in groups:
		if seed_index in group or seed_index + 1 or seed_index + 2 in group:
			return streamline[group]
	
	return np.empty((0, 2))


def trace_streamlines_from_sources(sources, z_func, inside_indicator_func, step=1800, step_size=0.001):
	streamlines = []
	for pt in tqdm(sources):
		branching_sign = 1
		line = trace_streamline(lambda xk: z_func(xk) * branching_sign, pt, step=step, step_size=step_size)

		if line is not None:
			line = truncate_streamline_by_indicator(line, 0, inside_indicator_func)
			if len(line) > 0:
				streamlines.append(line)
		
	return streamlines

