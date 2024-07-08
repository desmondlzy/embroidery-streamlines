# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

#%%
import sys; sys.path.append("..")
from IPython import get_ipython
if (ipython_instance := get_ipython()) is not None:
	ipython_instance.run_line_magic("load_ext", "autoreload")
	ipython_instance.run_line_magic("autoreload", "2")

from matplotlib import pyplot as plt
import numpy as np

from embroidery.pipeline import main_pipeline
from embroidery.utils import plotter, math

import common.input
import common.helpers
#%%
# generate the input: grids of density and direction
xl, yl = 0, 0
xr, yr = 1, 1
nx, ny = 501, 500
xaxis = np.linspace(xl, xr, nx)
yaxis = np.linspace(yl, yr, ny)
xgrid, ygrid = np.meshgrid(xaxis, yaxis)
eps = 0.03
boundary = np.array((
	(eps, eps), 
	(1 - eps, eps), 
	(1 - eps, 1 - eps), 
	(eps, 1 - eps), 
	(eps, eps)), dtype=float)

xx, yy = xgrid, ygrid

inside_indicator_grid = np.ones_like(xx)
inside_indicator_grid = np.where(
	(xx >= eps) & (xx <= 1 - eps) & (yy >= eps) & (yy <= 1 - eps), 1, 0)

def vertical_linear_density_constant_direction(xx, yy):
	density_grid = (1 - yy) * 0.56
	direction_grid = np.zeros_like(xx)   # this is the angle to the x-axis, so 0 means "to the right"
	return "vertical-linear-den-const-dir", density_grid, direction_grid

def horizontal_linear_density_constant_direction(xx, yy):
	density_grid = xx * 0.5 + 0.2
	direction_grid = np.zeros_like(xx)   # this is the angle to the x-axis, so 0 means "to the right"
	return "horizontal-linear-den-const-dir", density_grid, direction_grid


def constant_density_constant_direction(xx, yy):
	density_grid = np.full_like(yy, 0.4)   # spatially constant density
	direction_grid = np.zeros_like(xx)   # this is the angle to the x-axis, so 0 means "to the right"
	return "const-den-const-dir", density_grid, direction_grid

#%%
function = vertical_linear_density_constant_direction  # change the following line for different analytical fields shown above

relative_line_width = 0.010 # this is global density multiplier -- density = 1 / relative_line_width
task_name, density_grid, direction_grid = function(xx, yy)
print(f"{task_name = }")

from matplotlib import colors as mcolors
rgba_val = mcolors.to_rgba(common.helpers.primary_color)
density_image = np.zeros((*density_grid.shape, 4))
for i in range(4):
	density_image[:, :, i] = math.map_range(density_grid, 0, 1, 1, rgba_val[i])
density_image[:, :, 3] = 1

plt.imshow(density_image, cmap=common.helpers.primary_colormap())
plt.axis("scaled")
# plotter.savefig_tight(f"./output/analytical-{task_name}/density.svg")
plt.show()
plt.clf()

#%%
qxx, qyy = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
_, _, _dir = function(qxx, qyy)
plt.quiver(qxx, qyy, np.cos(_dir), np.sin(_dir), 
	pivot="middle", color=common.helpers.primary_color, scale=1 / 0.15, scale_units="height",
	width=0.02)
plt.axis("scaled")
plt.xlim(0, 1)
plt.ylim(0, 1)
# plotter.savefig_tight(f"./output/analytical-{task_name}/directionality.svg")
# plt.show()
plt.clf()


#%%
line = main_pipeline(
	xaxis=xaxis,
	yaxis=yaxis,
	boundary=boundary,
	holes=[],
	density_grid=density_grid,
	direction_grid=direction_grid,
	inside_indicator_grid=inside_indicator_grid,
	relative_line_width=relative_line_width,
	use_boundary_for_smoothing=True,
	streamline_step=5000,
	streamline_step_size=0.0008,
	# plot_save_folder=f"output/analytical-{task_name}",
	plot_save_folder="show",
)

