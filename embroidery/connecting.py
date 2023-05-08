# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import networkx
from tqdm import tqdm
import numpy as np
import triangle

from embroidery.utils.generator import pairwise

def connect_lines(curves, dir_func, density_func, returns_new_edges=False):
	graph = networkx.Graph()
	vertices = np.row_stack(curves)
	graph.add_nodes_from(range(len(vertices)))

	node_index_offset = 0
	point_index_to_curve_index = {}
	for crv_ind, crv in tqdm(enumerate(curves)):
		point_index_to_curve_index.update({k: crv_ind for k in np.arange(len(crv)) + node_index_offset})
		graph.add_edges_from([(pi, qi) for pi, qi in pairwise(np.arange(len(crv)) + node_index_offset)], weight=-np.inf)
		node_index_offset += len(crv)

	print(f"initialization finished, {graph.number_of_nodes() = } {graph.number_of_edges() = }")

	assert len(point_index_to_curve_index) == sum([len(c) for c in curves])


	# find candidates by tessellating with delaunay
	tri_results = triangle.triangulate({
		"vertices": vertices,
		# "segments": np.array(segment_endpoints),
		}, "e")
	print("triangle.triangulate add new vertices?", len(tri_results["vertices"]), len(vertices))
	print(f"delaunay finished: #V = {len(tri_results['vertices'])}, #E = {len(tri_results['edges'])}, #F = {len(tri_results['triangles'])}")
	candidates = tri_results["edges"]

	for pi, qi in tqdm(candidates):
		assert pi < len(vertices) and qi < len(vertices)
		not_on_same_curve = point_index_to_curve_index[pi] != point_index_to_curve_index[qi] 
		if not_on_same_curve:
			pt_p, pt_q = vertices[[pi, qi]]
			pt_mid = (pt_p + pt_q) / 2
			edge_vec = pt_q - pt_p

			w = -np.abs(
				density_func(pt_mid[0], pt_mid[1]) / np.linalg.norm(edge_vec))

			graph.add_edge(pi, qi, weight=w)
		else:
			pass
	


	print(f"initialization finished, {graph.number_of_nodes() = } {graph.number_of_edges() = }")

	assert networkx.is_connected(graph)

	# minimum spanning tree for connecting the curves into a single path
	spanning_tree = networkx.minimum_spanning_tree(graph)
	print(f"min spanning tree finished, {spanning_tree.number_of_nodes() = } {spanning_tree.number_of_edges() = }")
	if returns_new_edges:
		new_edges = [e for e in spanning_tree.edges(data=True) if e[2]["weight"] > -np.inf]

	# find the eulerian path
	# double the edges of the spanning tree -> all nodes are of degree of multiple of 2
	double_graph = networkx.MultiGraph()
	double_graph.add_nodes_from(spanning_tree.nodes(data=True))
	double_graph.add_edges_from(spanning_tree.edges(data=True))
	double_graph.add_edges_from(spanning_tree.edges(data=True))

	assert networkx.is_eulerian(double_graph)
	eu_path = list(networkx.eulerian_circuit(double_graph))

	for e_a, e_b in pairwise(eu_path):
		assert e_a[1] == e_b[0]

	single_curve = []
	for _, i in eu_path:
		single_curve.append(vertices[i])
	single_curve = np.array(single_curve)

	if returns_new_edges:
		return single_curve, new_edges

	return single_curve