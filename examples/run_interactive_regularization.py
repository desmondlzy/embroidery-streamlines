# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

#%%
from math import radians
import sys, os

sys.path.append("..")

from matplotlib import pyplot as plt
from matplotlib import patches

import numpy as np

from embroidery.utils import math
from embroidery.smoothing import spring_energy_matrices, weigh_matrices_and_solve
from embroidery.pipeline import main_pipeline

import common.input
from common import helpers

from IPython import get_ipython
if (ipython_instance := get_ipython()) is not None:
	ipython_instance.run_line_magic("load_ext", "autoreload")
	ipython_instance.run_line_magic("autoreload", "2")

#%%
xl, yl = 0, 0
xr, yr = 1, 1
nx, ny = 501, 500
xaxis = np.linspace(xl, xr, nx)
yaxis = np.linspace(yl, yr, ny)
xgrid, ygrid = np.meshgrid(xaxis, yaxis)
eps = 0.01
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

def linear_density_constant_direction(xx, yy):
	density_grid = (1 - yy) * 0.56
	direction_grid = np.zeros_like(xx)
	return "linear-den-constant-dir", density_grid, direction_grid


def linear_horizontal_density_constant_direction(xx, yy):
	density_grid = (1 - xx) * 0.56
	direction_grid = np.zeros_like(xx)
	return "linear-horizontal-den-constant-dir", density_grid, direction_grid


def circular_density_constant_direction(xx, yy):
	xc = xx - 0.5
	yc = yy - 0.5
	dist = np.sqrt(xc ** 2 + yc ** 2)
	density_grid = math.map_range(
		np.clip(dist, 0, 0.5), 
		0.5, 0,
		0.02, 0.8) 
	direction_grid = np.zeros_like(xx)
	return "circular-den-constant-dir", density_grid, direction_grid

def circular_density_circular_direction(xx, yy):
	xc = xx 
	yc = yy - 0.5
	density_grid = np.sqrt(xc ** 2 + yc ** 2)
	density_grid = math.map_range(
		density_grid, 
		density_grid.max(), density_grid.min(),
		0.0, 1)
	direction_grid = np.arctan2(yc, xc) + np.pi / 2
	return "circular-den-circular-dir", density_grid, direction_grid

def constant_density_u_direction(xx, yy):
	xc = xx - 0.5
	yc = yy - 0.5
	density_grid = np.ones_like(xx) * 0.45
	direction_grid = np.where(
		xc > 0, 
		np.arctan2(yc, xc) + np.pi / 2,
		np.where(yc > 0, np.pi, 0))
	return "constant-den-u-dir", density_grid, direction_grid

def circular_density_hurricane_direction(xx, yy):
	xc = xx - 0.5
	yc = yy - 0.5
	density_grid = np.sqrt(xc ** 2 + yc ** 2)
	density_grid = math.map_range(
		density_grid, 
		density_grid.max(), density_grid.min(),
		0.0, 0.75)
	direction_grid = np.arctan2(yc, xc) + np.pi / 4
	return "circular-den-hurricane-dir", density_grid, direction_grid


def constant_density_hurricane_direction(xx, yy):
	xc = xx - 0.5
	yc = yy - 0.5
	density_grid = np.full_like(xx, 0.5)
	direction_grid = np.arctan2(yc, xc) + np.pi / 4
	return "constant-den-hurricane-dir", density_grid, direction_grid


def inversecircular_density_radial_direction(xx, yy):
	xc = xx 
	yc = yy 
	density_grid = np.sqrt(xc ** 2 + yc ** 2)
	density_grid = math.map_range(
		density_grid, 
		density_grid.max(), density_grid.min(),
		0.5, 0.1)
	direction_grid = np.arctan2(yc, xc) + np.pi
	return "inversecircular-den-radial-dir", density_grid, direction_grid

function = linear_horizontal_density_constant_direction
relative_line_width = 0.010
task_name, density_grid, direction_grid = function(xx, yy)
print(f"{task_name = }")

# %%
components = main_pipeline(
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
	plot_save_folder="figures/regweights",
	returns_regularization_matrices=True,
)
# %% interactive UI 
from matplotlib.widgets import Slider, Button
from embroidery import smoothing
# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
curves = weigh_matrices_and_solve(components)[:-1]
lines = []

for c in curves:
	l, = plt.plot(c[:, 0], c[:, 1], "b")
	lines.append(l)
plt.gca().set_xticks([])
plt.gca().set_yticks([])

box = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
plt.plot(box[:, 0], box[:, 1], "g--")

# adjust the main plot to make room for the sliders
plt.subplots_adjust(bottom=0.25)

plt.axis("scaled")
# plt.xlim(0, 1)
# plt.ylim(0, 1)

# Make a horizontal slider to control the frequency.
ax_align = plt.axes([0.25, 0.1, 0.65, 0.03])
align_slider = Slider(
	ax=ax_align,
	label="direction",
	valmin=-8,
	valmax=8,
	valinit=5,
	valstep=0.5,
)
ax_space = plt.axes([0.25, 0.15, 0.65, 0.03])
space_slider = Slider(
	ax=ax_space,
	label="density",
	valmin=-8,
	valmax=8,
	valinit=4,
	valstep=0.5,
)

# The function to be called anytime a slider's value changes
def update(val):
	updated_curves = weigh_matrices_and_solve(
		components, 
		w_align=10 ** align_slider.val,
		w_space=10 ** space_slider.val,
		w_dirreg=1e5,
		w_reg=1e-7,
		w_fix=1e5,
		)[:-1]
	for i, c in enumerate(updated_curves):
		lines[i].set_xdata(c[:, 0])
		lines[i].set_ydata(c[:, 1])
	fig.canvas.draw_idle()
	# plt.savefig(f"figures/regweights/align={align_slider.val},space={space_slider.val},dirreg={dirreg_slider.val},reg={reg_slider.val}.svg", transparent=True, pad_inches=0, bbox_inches="tight")
	# plt.savefig(f"figures/regweights/align={align_slider.val},space={space_slider.val},dirreg={dirreg_slider.val},reg={reg_slider.val}.png", transparent=True, pad_inches=0, bbox_inches="tight")

# register the update function with each slider
for slider in (align_slider, space_slider):
	slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
plt.show()
common.input.matplotlib_settings()
smoothed_lines_w_boundary = smoothing.weigh_matrices_and_solve(
	components,
	w_align=1e3, 
	w_space=1e1, 
	w_dirreg=1e5, 
	w_reg=1e-7, 
	w_fix=1e5
	)
