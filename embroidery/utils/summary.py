from collections import namedtuple
import numpy as np
from pyembroidery import EmbPattern
from pyembroidery.EmbConstant import SEQUENCE_BREAK, STITCH, TRIM

from embroidery.utils.stitch import stitch_over_path

PatternSummary = namedtuple("PatternSummary", "num_stitches, bbox, num_breaks, dimension, average_length, min_length")

def summarize_pattern(pattern_or_curve: EmbPattern):
	if isinstance(pattern_or_curve, EmbPattern):
		pattern = pattern_or_curve
	elif isinstance(pattern_or_curve, np.ndarray):
		pattern = EmbPattern()
		stitch_over_path(pattern, pattern_or_curve)
		
	stitches = pattern.stitches
	vertices = np.array([(x, y) for x, y, cmd in stitches if cmd == STITCH])

	bounding_box = np.min(vertices, axis=0), np.max(vertices, axis=0)

	stitch_lengths = np.linalg.norm(vertices[1:] - vertices[:-1], axis=1)

	width, height = bounding_box[1] - bounding_box[0]

	summary = PatternSummary(
		num_stitches=count_stitches(pattern), 
		num_breaks=count_trim_and_sequence_breaks(pattern),
		bbox=bounding_box,
		dimension=(width, height),
		average_length=np.mean(stitch_lengths),
		min_length=np.min(stitch_lengths))

	return summary


def count_stitches(pattern: EmbPattern):
	stitches = pattern.stitches
	return sum([1 for x, y, cmd in stitches if cmd == STITCH])


def count_trim_and_sequence_breaks(pattern: EmbPattern):
	stitches = pattern.stitches
	return sum([1 for x, y, cmd in stitches if cmd in (SEQUENCE_BREAK, TRIM)])
