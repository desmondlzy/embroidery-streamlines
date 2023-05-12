# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import numpy as np
from scipy import interpolate

from embroidery.kdsampling import sample_points_on_interior, sample_points_on_line
from embroidery import smoothing, streamlines, kdsampling, connecting
from embroidery.utils import math, pairwise
from embroidery.utils.timer import catchtime

from matplotlib import pyplot as plt

primary_color = "#1553b7ff"
secondary_color = "#ff6a00ff"


def main_pipeline(
	xaxis, 
	yaxis, 
	boundary,
	holes,
	density_grid,
	direction_grid,
	inside_indicator_grid,
	relative_line_width,
	streamline_step=1800,
	streamline_step_size=0.001,
	use_boundary_for_smoothing=True,
	plot_figure=False,
	plot_save_folder=None,
	print_timings=False,
	returns_regularization_matrices=False,
	):
	"""
	The main pipeline of the core part of the algorithm, do the following sequentially

		Sampling on boundary
		Sampling on interior
		Tracing from the sources
		Cutting from the sinks
		Smoothing

	"""
	if plot_save_folder is not None and plot_figure == False:
		print("it looks like you want to plot the figure, override the plot_figure = True")
		plot_figure = True


	if plot_save_folder is not None and plot_save_folder != "show":
		print("Plotting the figures: ", plot_save_folder)
		os.makedirs(plot_save_folder, exist_ok=True)

	# make sure the boundary goes anticlockwise by computing the winding number and reverse if necessary
	# ensure the holes goes clockwise, so that lines come out of the holes
	print(f"main pipeline starts")
	print(f"{relative_line_width = }")
	print(f"prepares the grids and interpolants")
	print(f"grid shape: {density_grid.shape}")
	boundary = math.ensure_counterclockwise(boundary)
	holes = [np.flipud(math.ensure_counterclockwise(h)) for h in holes] 

	dir_u_grid = np.cos(direction_grid)
	dir_v_grid = np.sin(direction_grid)

	dir_u_interp = interpolate.RectBivariateSpline(xaxis, yaxis, dir_u_grid.transpose(), kx=1, ky=1)
	dir_v_interp = interpolate.RectBivariateSpline(xaxis, yaxis, dir_v_grid.transpose(), kx=1, ky=1)

	density_interp = interpolate.RectBivariateSpline(xaxis, yaxis, density_grid.transpose(), kx=1, ky=1)

	def directionality(xk):
		return np.array([dir_u_interp.ev(xk[0], xk[1]), dir_v_interp.ev(xk[0], xk[1])])
	
	def z_func(xk):
		return density_interp.ev(xk[0], xk[1]) * directionality(xk)

	gradient_y_grid = (np.roll(density_grid * dir_v_grid, -1, axis=0) - np.roll(density_grid * dir_v_grid, 1, axis=0)) / (2 / (density_grid.shape[0] - 1))
	gradient_x_grid = (np.roll(density_grid * dir_u_grid, -1, axis=1) - np.roll(density_grid * dir_u_grid, 1, axis=1)) / (2 / (density_grid.shape[1] - 1))
	divergence_grid = gradient_x_grid + gradient_y_grid

	assert np.all(inside_indicator_grid >= 0)

	inside_indicator_interp = interpolate.RectBivariateSpline(xaxis, yaxis, inside_indicator_grid.transpose())

	pos_div_grid = np.maximum(divergence_grid, 0) * inside_indicator_grid
	neg_div_grid = -np.minimum(divergence_grid, 0) * inside_indicator_grid

	pos_signal_interp = interpolate.RectBivariateSpline(xaxis, yaxis, pos_div_grid.transpose() * (1 / relative_line_width))
	neg_signal_interp = interpolate.RectBivariateSpline(xaxis, yaxis, neg_div_grid.transpose() * (1 / relative_line_width))

	# positive_interior_sources = sample_points_on_interior(lambda xk: np.abs(div_func.ev(xk[0], xk[1])), normalized_segment_boundary, 1 / relative_line_width)
	positive_interior_sources = sample_points_on_interior(pos_signal_interp, 0, 1, 0, 1)
	negative_interior_sources = sample_points_on_interior(neg_signal_interp, 0, 1, 0, 1)

	print(f"sampling sources and sinks")
	with catchtime("samplings"):
		boundary_sources = kdsampling.sample_sources_on_closed_boundary(z_func, boundary, relative_line_width)

	if len(holes) > 0:
		hole_sources = np.row_stack([kdsampling.sample_sources_on_closed_boundary(z_func, h, relative_line_width) for h in holes])
		positive_sources = np.row_stack((boundary_sources, positive_interior_sources, hole_sources))
	else:
		hole_sources = np.empty((0, 2))
		positive_sources = np.row_stack((boundary_sources, positive_interior_sources, ))

	negative_sources = np.row_stack((negative_interior_sources, ))

	print(f"streamline tracing")
	with catchtime("tracing"):
		init_streamlines = streamlines.trace_streamlines_from_sources(
			positive_sources, 
			z_func, 
			lambda x, y: inside_indicator_interp.ev(x, y),
			step=streamline_step,
			step_size=streamline_step_size)

	init_streamlines = [
		l for l in init_streamlines
		if len(l) >= 2]  # throw away lines that are too short (single vertex), otherwise error in spring_energy_matrices

	if plot_figure and plot_save_folder is not None:
		plt.title("Before cutting")
		plt.plot(boundary[:, 0], boundary[:, 1], color="black")
		plt.scatter(positive_interior_sources[:, 0], positive_interior_sources[:, 1], color=primary_color)
		plt.scatter(negative_interior_sources[:, 0], negative_interior_sources[:, 1], color=secondary_color)
		plt.scatter(boundary_sources[:, 0], boundary_sources[:, 1], color=primary_color)
		if len(hole_sources) > 0:
			plt.scatter(hole_sources[:, 0], hole_sources[:, 1], color=primary_color)
		for l in init_streamlines:
			assert len(l) >= 2
			plt.plot(l[:, 0], l[:, 1], color=primary_color)
	
		plt.axis("scaled")
		if plot_save_folder == "show":
			plt.show()
		else:
			plt.savefig(f"{plot_save_folder}/before-cutting.svg", bbox_inches="tight", pad_inches=0)
			plt.savefig(f"{plot_save_folder}/before-cutting.png", bbox_inches="tight", pad_inches=0)
		plt.clf()

	print(f"{len(init_streamlines) = } traced")
	# cut_streamlines = streamlines.activate_lines(init_streamlines, pos_points=None, neg_points=negative_sources)

	print(f"streamline cutting")
	with catchtime("cutting"):
		cut_streamlines = [
			l 
			for l in streamlines.cut_streamlines_assignment(init_streamlines, negative_sources, boundary) 
			if len(l) >= 2]  # throw away lines that are too short, otherwise error in spring_energy_matrices

	print(f"{len(cut_streamlines) = } traced")
	if plot_figure and plot_save_folder is not None:
		plt.title("Sources, sinks and streamlines")
		plt.plot(boundary[:, 0], boundary[:, 1], color="black")
		plt.scatter(positive_interior_sources[:, 0], positive_interior_sources[:, 1], color=primary_color)
		plt.scatter(negative_interior_sources[:, 0], negative_interior_sources[:, 1], color=secondary_color)
		plt.scatter(boundary_sources[:, 0], boundary_sources[:, 1], color=primary_color)
		if len(hole_sources) > 0:
			plt.scatter(hole_sources[:, 0], hole_sources[:, 1], color=primary_color)
		for l in cut_streamlines:
			assert len(l) >= 2
			plt.plot(l[:, 0], l[:, 1], color=primary_color)
	
		plt.axis("scaled")
		if plot_save_folder == "show":
			plt.show()
		else:
			plt.savefig(f"{plot_save_folder}/sources-sinks-streamlines.svg", bbox_inches="tight", pad_inches=0)
			plt.savefig(f"{plot_save_folder}/sources-sinks-streamlines.png", bbox_inches="tight", pad_inches=0)
		plt.clf()


	if use_boundary_for_smoothing:
		inflated_boundary = math.perturb_curve_along_normal(boundary)  # assuming boundary is counter-clockwise
		shrinked_holes = [math.perturb_curve_along_normal(h) for h in holes]  # assuming holes are clockwise
		smoothing_boundary = np.row_stack((inflated_boundary, *shrinked_holes))
	else:
		smoothing_boundary = None

	print(f"building spring_energy_matrices")
	with catchtime("spring matrices"):
		components = smoothing.spring_energy_matrices(
			cut_streamlines, 
			line_width=relative_line_width,
			z_func=z_func,
			boundary=smoothing_boundary,
			inside_indicator_func=lambda xs, ys: inside_indicator_interp.ev(xs, ys),
			density_func=lambda xs, ys: density_interp.ev(xs, ys), 
			spring_density_threshold=0.1, plot=False)


	if returns_regularization_matrices:
		return components


	print(f"starting weigh and solve")
	with catchtime("spring solve"):
		smoothed_lines_w_boundary = smoothing.weigh_matrices_and_solve(
			components,
			w_align=1e3, 
			w_space=1e1, 
			w_dirreg=1e5, 
			w_reg=1e-7, 
			w_fix=1e5
			)

	print(f"remove outsider")
	smoothed_lines = streamlines.crop_streamlines_outside_boundary(
		(smoothed_lines_w_boundary[: -1] if use_boundary_for_smoothing else smoothed_lines_w_boundary) ,  # remove the boundary added during smoothing
		lambda x, y: inside_indicator_interp.ev(x, y).round().astype(bool),
	)

	# heuristic for avoiding long connecting stitches between short stitches in areas of low density
	removal_threshold = min(2, np.mean([len(l) for l in smoothed_lines]) * 0.3)
	remove_short_lines = [l for l in smoothed_lines if len(l) > np.round(removal_threshold)]

	if plot_save_folder is not None:
		plt.plot(boundary[:, 0], boundary[:, 1], color="black")
		plt.scatter(positive_interior_sources[:, 0], positive_interior_sources[:, 1], color=primary_color)
		plt.scatter(negative_interior_sources[:, 0], negative_interior_sources[:, 1], color=secondary_color)
		plt.scatter(boundary_sources[:, 0], boundary_sources[:, 1], color=primary_color)
		if len(hole_sources) > 0:
			plt.scatter(hole_sources[:, 0], hole_sources[:, 1], color=primary_color)
		for l in remove_short_lines:
			assert len(l) >= 2
			plt.plot(l[:, 0], l[:, 1], color=primary_color)
	
		plt.axis("scaled")
		if plot_save_folder == "show":
			plt.show()
		else:
			plt.savefig(f"{plot_save_folder}/smoothing-lines.svg", bbox_inches="tight", pad_inches=0)
			plt.savefig(f"{plot_save_folder}/smoothing-lines.png", bbox_inches="tight", pad_inches=0)
		plt.clf()

	print(f"removed line count: {len(smoothed_lines) - len(remove_short_lines) = }")
	print(f"postprocessing")
	with catchtime("postprocessing"):
		resampled_lines = streamlines.scale_and_resample(remove_short_lines, 1, relative_line_width * 4)

		fg_single_line = connecting.connect_lines(
			resampled_lines, 
			lambda x, y: directionality([x, y]), 
			density_interp.ev) 


	return fg_single_line