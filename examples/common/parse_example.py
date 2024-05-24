from dataclasses import dataclass
import functools
from typing import List, Mapping, Any

import numpy as np
from scipy import interpolate
import numba
from matplotlib import pyplot as plt

from embroidery.utils import math
from embroidery.pca import color_gradient_extraction_with_pca


@numba.njit
def recover_colors(grid, color_left, color_right):
	cimg = np.empty((*grid.shape, 3))
	for i in range(grid.shape[0]):
		for j in range(grid.shape[1]):
			t = grid[i, j]
			cimg[i, j] = (1 - t) * color_left + t * color_right
	return cimg


@dataclass
class Context:
	image: np.ndarray
	direction_grid: np.ndarray
	density_grid: np.ndarray
	t_grid: np.ndarray
	t_interpolant: Any
	indicator_grid: np.ndarray
	boundary: np.ndarray
	annotation: np.ndarray
	holes: List[np.ndarray]
	color_fg: np.ndarray
	color_bg: np.ndarray
	line_fg: np.ndarray
	line_bg: np.ndarray


def interpolate_direction_on_grid(direction_data, xgrid, ygrid):
	data_x, data_y, data_u, data_v = [], [], [], []

	for da in direction_data:
		rad = np.arctan2(da[1, 1] - da[0, 1], da[1, 0] - da[0, 0])
		data_x.append(da[0, 0])
		data_y.append(da[0, 1])
		data_u.append(np.cos(rad))
		data_v.append(np.sin(rad))
	
	# from matplotlib import pyplot as plt
	# plt.quiver(data_x, data_y, data_u, data_v)
	# plt.show()

	points = np.column_stack((data_x, data_y))
	for x in (0, 1):
		for y in (0, 1):
			nearest_ind = np.argmin(np.linalg.norm(points - np.array([x, y]), axis=1))
			data_x.append(x)
			data_y.append(y)
			data_u.append(data_u[nearest_ind])
			data_v.append(data_v[nearest_ind])
	
	u_interp = interpolate.griddata(np.column_stack([data_x, data_y]), data_u, (xgrid, ygrid))
	v_interp = interpolate.griddata(np.column_stack([data_x, data_y]), data_v, (xgrid, ygrid))
	return np.arctan2(v_interp, u_interp)


def _get_annotation(json_obj, label_name):
	item_index = [obj["label"] for obj in json_obj["shapes"]].index(label_name)
	image_annotation = np.array(json_obj["shapes"][item_index]["points"])
	return image_annotation


def _analyze_direction_and_indicator(json_obj, label_name, xgrid, ygrid, img2nor):
	"""Helper for `parse_example_common`
	"""
	item_index = [obj["label"] for obj in json_obj["shapes"]].index(label_name)
	image_annotation = np.array(json_obj["shapes"][item_index]["points"])
	normal_anno =  math.ensure_counterclockwise(img2nor(image_annotation))
	direction_annotations = [np.array(shape["points"]) 
		for shape in json_obj["shapes"] 
		if shape["shape_type"] == "line" and shape["label"].startswith(label_name.replace("region", "direction"))]

	assert len(direction_annotations) > 0, f"did you annotate {label_name} with direction?"
	direction_data = np.asarray([img2nor(anno) for anno in direction_annotations])
	
	_starts = direction_data[:, 0, :]
	_ends = direction_data[:, 1, :]
	_dir = (_ends - _starts) / np.linalg.norm(_starts - _ends, axis=1)[:, np.newaxis]
	# plt.title(f"{label_name}")
	# plt.quiver(_starts[:, 0], _starts[:, 1], _dir[:, 0], _dir[:, 1])
	# plt.plot(normal_anno[:, 0], normal_anno[:, 1])
	# plt.axis("scaled")
	# plt.show()

	dir_grid = interpolate_direction_on_grid(direction_data, xgrid, ygrid)

	ps = normal_anno
	qs = np.roll(ps, -1, axis=0)

	@numba.vectorize([numba.float64(numba.float64, numba.float64)])
	def inside_func(x, y):
		num = math.signed_angle_ufunc(ps[:, 0], ps[:, 1], qs[:, 0], qs[:, 1], x, y).sum()
		return num

	inside_indicator_grid = np.round(-inside_func(xgrid, ygrid))


	return dir_grid, normal_anno, inside_indicator_grid

def _analyze_density_and_color(image, inside_indicator_grid, nor2img, xaxis, yaxis):
	xgrid, ygrid = np.meshgrid(xaxis, yaxis)
	indicator_interp = interpolate.RectBivariateSpline(xaxis, yaxis, inside_indicator_grid.transpose(), kx=1, ky=1)
	t_grid, color_fg, color_bg = color_gradient_extraction_with_pca(
		image, 
		lambda x, y: indicator_interp.ev(x, y).round().astype(bool), 
		nor2img,
		)

	# resample t_grid to the same resolution as other grids
	t_interpolant = interpolate.interp2d(
		np.arange(t_grid.shape[1]), 
		np.arange(t_grid.shape[0]), 
		t_grid)

	def _density(x, y):
		xk = np.array([x, y], dtype=float)
		img_xk = nor2img(xk)
		return math.map_range(
			1 - np.clip(t_interpolant(*img_xk), 0.001, 0.999),
			0, 1, 0.1, 1)

	f = np.frompyfunc(_density, 2, 1)
	density_grid = f(xgrid, ygrid).astype(float)

	if (mean_density := density_grid.mean()) > 0.7:
		print(f"{mean_density = } > 0.7, swap the foreground and background color")
		density_grid = 1 - density_grid
		color_fg, color_bg = color_bg, color_fg


	return density_grid, color_fg, color_bg


def parse_example_common(image, json_obj, nor2img, img2nor, xaxis, yaxis):
	xgrid, ygrid = np.meshgrid(xaxis, yaxis)
	label_names = set([name for obj in json_obj["shapes"] if (name := obj["label"]).startswith("region")])
	print(f"parse {label_names}")

	context_map: Mapping[str, Context] = dict()

	for i, label_name in enumerate(label_names):
		print(f"{label_name = }")
		dir_grid, normal_anno, inside_indicator_grid = _analyze_direction_and_indicator(
			json_obj, label_name, xgrid, ygrid, img2nor)

		if not np.allclose(normal_anno[0], normal_anno[1], atol=1e-6):
			normal_anno = np.row_stack((normal_anno, normal_anno[0]))  # close the curve


		indicator_interp = interpolate.RectBivariateSpline(xaxis, yaxis, inside_indicator_grid.transpose(), kx=1, ky=1)
		t_grid, color_fg, color_bg = color_gradient_extraction_with_pca(
			image, 
			lambda x, y: indicator_interp.ev(x, y).round().astype(bool), 
			nor2img,
			)

		# resample t_grid to the same resolution as other grids
		t_interpolant = interpolate.interp2d(
			np.arange(t_grid.shape[1]), 
			np.arange(t_grid.shape[0]), 
			t_grid)

		def _density(x, y):
			xk = np.array([x, y], dtype=float)
			img_xk = nor2img(xk)
			return math.map_range(
				1 - np.clip(t_interpolant(*img_xk), 0.001, 0.999),
				0, 1, 0.1, 1)

		f = np.frompyfunc(_density, 2, 1)
		density_grid = f(xgrid, ygrid).astype(float)

		if (mean_density := density_grid.mean()) > 0.7:
			print(f"{mean_density = } > 0.7, swap the foreground and background color")
			density_grid = 1 - density_grid
			color_fg, color_bg = color_bg, color_fg


		context_map[label_name] = Context(
			image=image, 
			boundary=normal_anno,
			annotation=_get_annotation(json_obj, label_name),
			holes=[],
			indicator_grid=inside_indicator_grid,
			direction_grid=dir_grid,
			density_grid=density_grid,
			t_grid=t_grid,
			t_interpolant=t_interpolant,
			color_fg=color_fg,
			color_bg=color_bg,
			line_fg=None,
			line_bg=None,
			)
	

	return context_map


def parse_example_by_name(name, image, json_obj):
	print(f"parsing example {name = }")
	xl, yl = 0, 0
	xr, yr = 1, 1
	nx, ny = 501, 500
	xaxis = np.linspace(xl, xr, nx)
	yaxis = np.linspace(yl, yr, ny)

	annotations = np.vstack(
		[np.array(shape["points"]) for shape in json_obj["shapes"] if shape["shape_type"] == "polygon"]
		)
	bbox_min, bbox_max = annotations.min(axis=0), annotations.max(axis=0)
	longest_edge_px = max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1])

	@numba.njit
	def _image_space_to_normal_space(xk):
		return (xk - bbox_min) / longest_edge_px

	@numba.njit
	def _normal_space_to_image_space(xk):
		return (xk * longest_edge_px) + bbox_min

	if name == "landscape":
		context_map = parse_example_common(
			image, json_obj, 
			_normal_space_to_image_space, _image_space_to_normal_space, 
			xaxis, yaxis)

		# revert the foreground and background of the cloud (region-2), mountain (region-3)
		for key in ("region-2", "region-3"):
			print(f"inverting the background and foreground of {key}.")
			label_ctx = context_map[key]
			label_ctx.color_bg, label_ctx.color_fg = label_ctx.color_fg, label_ctx.color_bg
			label_ctx.density_grid = 1 - label_ctx.density_grid
		
	elif name == "sunset":
		context_map = parse_example_common(
			image, json_obj, 
			_normal_space_to_image_space, _image_space_to_normal_space, 
			xaxis, yaxis)

		# dig the sun (region-5) as a hole in the sky (region-4)
		# context_map["region-4"].holes = [
		# 	np.flipud(context_map[rg].boundary)  # invert the order to make the orientation, and thus the normal direction right
		# 	for rg in ("region-5", )
		# ]

		# context_map["region-4"].indicator_grid = (
		# 	context_map["region-4"].indicator_grid - context_map["region-5"].indicator_grid
		# )

		# density_grid, cfg, cbg = _analyze_density_and_color(image, context_map["region-4"].indicator_grid,
		# 	_normal_space_to_image_space, xaxis, yaxis)

		# context_map["region-4"].density_grid = density_grid
		# context_map["region-4"].color_fg = cfg
		# context_map["region-4"].color_bg = cbg


	else:
		context_map = parse_example_common(image, json_obj, _normal_space_to_image_space, _image_space_to_normal_space, xaxis, yaxis)

	return context_map
