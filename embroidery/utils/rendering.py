import numpy as np
import cairo

from .generator import pairwise


def init_drawer(width, height, format="rgba", svg_filename=None):
	if format == "rgba":
		surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
	if format == "svg":
		surf = cairo.SVGSurface(svg_filename, width, height)

	drawer = cairo.Context(surf)
	drawer.save()
	drawer.set_source_rgb(1.0, 1.0, 1.0)
	drawer.set_line_width(0.3)
	drawer.paint()
	drawer.restore()

	# set the coordinate system: origin at bottom left corner, y-axis points up, x-axis points right
	drawer.set_matrix(cairo.Matrix(yy=-1, y0=height))

	return drawer


def do_stroke(drawer: cairo.Context, coordinates: np.ndarray):
	drawer_coords = coordinates
	drawer.move_to(*drawer_coords[0])
	for i, p in enumerate(drawer_coords):
		drawer.line_to(*p)

	drawer.stroke()


def to_ndarray(context):
	height, width = context.get_target().get_height(), context.get_target().get_width()
	buf = context.get_target().get_data()
	array = np.ndarray(shape=(width, height, 4), dtype=np.uint8, buffer=buf)
	return array


def _quad_from_nodes(node_a, node_b):
	vertices = (
		node_a.coordinate - node_a.halfwidth * node_a.cross_axis,
		node_a.coordinate + node_a.halfwidth * node_a.cross_axis,
		node_b.coordinate + node_b.halfwidth * node_b.cross_axis,
		node_b.coordinate - node_b.halfwidth * node_b.cross_axis,
		node_a.coordinate - node_a.halfwidth * node_a.cross_axis,
	)
	return vertices


def flows2svg(filename: str, flows, canvas_width, canvas_height):
	c_w, c_h = canvas_width, canvas_height
	with cairo.SVGSurface(filename, c_w, c_h) as surface:
		context = cairo.Context(surface)

		context.set_source_rgba(1, 1, 1, 1)
		for p in ((0, 0), (c_w, 0), (c_w, c_h), (0, c_h), (0, 0)):
			context.line_to(*p)
		context.fill()

		# context.rectangle(100, 100, 100, 100)
		# context.rectangle(500, 100, 100, 100)
		context.set_source_rgba(0.7, 0.8, 0.9, 0.4)
		# context.set_line_width(0.01)
		for i, flow in enumerate(flows):
			for n_a, n_b in pairwise(flow.node_list):
				vertices = _quad_from_nodes(n_a, n_b)
				for v in vertices:
					context.line_to(*v)
				context.fill()

		context.set_source_rgb(0, 0, 0)
		context.set_line_width(4)
		for _, flow in enumerate(flows):
			for i, (n_a, n_b) in enumerate(pairwise(flow.node_list)):
				if i % 2 == 0:
					context.set_source_rgb(0, 0, 0)
				else:
					context.set_source_rgb(0.2, 0.2, 0.2)
				# vertices = _quad_from_nodes(n_a, n_b)
				context.move_to(*n_a.coordinate)
				context.line_to(*n_b.coordinate)
				context.stroke()