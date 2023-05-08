import os

from pyembroidery import (
	write_csv,
	write_dst,
	write_exp,
	write_png,
	convert
)
from pyembroidery.EmbPattern import EmbPattern

from embroidery.utils.stitch import stitch_over_path

def write_bundle(pattern_or_points, dirname, filename_prefix=None):
	os.makedirs(dirname, exist_ok=True)
	if filename_prefix is None:
		_, filename_prefix = os.path.split(dirname)

	if isinstance(pattern_or_points, EmbPattern):
		pattern = pattern_or_points
		write_csv(pattern, os.path.join(dirname, f"{filename_prefix}.csv"))
		write_dst(pattern, os.path.join(dirname, f"{filename_prefix}.dst"))
		write_exp(pattern, os.path.join(dirname, f"{filename_prefix}.exp"))
		write_png(pattern, os.path.join(dirname, f"{filename_prefix}.png"))
	else:
		pattern = EmbPattern()
		stitch_over_path(pattern, pattern_or_points)
		write_csv(pattern, os.path.join(dirname, f"{filename_prefix}.csv"))
		write_dst(pattern, os.path.join(dirname, f"{filename_prefix}.dst"))
		write_exp(pattern, os.path.join(dirname, f"{filename_prefix}.exp"))
		write_png(pattern, os.path.join(dirname, f"{filename_prefix}.png"))
	

def convert_exp_as_bundle(filename):
	name, ext = os.path.splitext(filename)
	convert(filename, f"{name}.csv")
	convert(filename, f"{name}.png")

