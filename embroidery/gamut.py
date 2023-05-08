# Copyright (c) 2023 Zhenyuan Desmond Liu <desmondzyliu@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import csv

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# load input colors
def gamut_projection_minimum_lab_distance(input_colors, candidate_colors):
    """ Choose for each input color, a closest candidate color based on delta e cie 2000 distance

    input_colors: list of hex code of rgb colors
    candidate_colors: list of hex code of rgb colors

    return: list of (distance, name of the color, hexcode of the color)
    """
    icf = input_colors
    ic = []
    for i in range(len(icf)):
        rgb = sRGBColor.new_from_rgb_hex(icf[i])
        lab = convert_color(rgb, LabColor)
        ic.append(lab)

    # load color gamut
    cgf = candidate_colors
    cgn = []
    cg = []
    for i in range(len(cgf)):
        rgb = sRGBColor.new_from_rgb_hex(cgf[i][1])
        lab = convert_color(rgb, LabColor)
        cg.append(lab)
        cgn.append(cgf[i][1])

    # for each color find closest match inside our gamut
    nc = []
    for i in range(len(ic)):
        source = ic[i]
        min_distance = 10000 #infinity
        min_idx = 0
        for j in range(len(cg)):
            target = cg[j]
            dE = delta_e_cie2000(source, target)
            if dE < min_distance:
                min_distance = dE
                min_idx = j
        nc.append((min_distance, cgf[min_idx], cgf[min_idx][1]))

    return nc


def get_default_gamuts():
    from importlib import resources

    with resources.open_text("embroidery.assets", "color_dataset.csv") as fp:
        reader = csv.reader(fp)
        rows = [row for row in reader]
    
    return rows



def relative_luminance_srgb(rgb):
    if rgb.rgb_r <= 0.03928:
        R = rgb.rgb_r / 12.92
    else:
        R = ((rgb.rgb_r + 0.055) / 1.055) ** 2.4
    if rgb.rgb_g <= 0.03928:
        G = rgb.rgb_g / 12.92
    else:
        G = ((rgb.rgb_g + 0.055)/1.055) ** 2.4
    if rgb.rgb_b <= 0.03928:
        B = rgb.rgb_b / 12.92
    else:
        B = ((rgb.rgb_b+0.055)/1.055) ** 2.4
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


def gamut_projection_minimum_luminance_distance(input_colors, candidate_colors):
    # load input colors
    icf = input_colors
    ic = []
    icrgb = []

    for i in range(len(icf)):
        rgb = sRGBColor.new_from_rgb_hex(icf[i])
        lab = convert_color(rgb, LabColor)
        ic.append(lab)
        icrgb.append(rgb)

    # load color gamut
    cgf = candidate_colors
    cgn = []
    cg = []
    cgrgb = []
    for i in range(len(cgf)):
        rgb = sRGBColor.new_from_rgb_hex(cgf[i][1])
        lab = convert_color(rgb, LabColor)
        cg.append(lab)
        cgrgb.append(rgb)
        cgn.append(cgf[i][0])
    # for each color find closest match inside our gamut
    # nc = []
    # for i in range(len(ic)):
    #     source = ic[i]
    #     min_distance = 10000#infinity
    #     min_idx = 0
    #     for j in range(len(cg)):
    #         target = cg[j]
    #         dE = delta_e_cie2000(source, target)
    #         if dE < min_distance:
    #             min_distance = dE
    #             min_idx = j
    #     nc.append((min_distance,cgn[min_idx], cgf[min_idx][1], cgrgb[min_idx]))
    # print(nc)

    # for each color pair find closest match inside gamut that preserves the local contrast
    # fix the first color to white
    nc = []
    for i in range(np.floor(len(ic) / 2).astype(int)):
        color1rgb = icrgb[i*2+0]
        color2rgb = icrgb[i*2+1]
        # compute relative luminance of both colors
        L1 = relative_luminance_srgb(color1rgb)
        L2 = relative_luminance_srgb(color2rgb)
        # get relative contrast, range [1, 21]
        C = (max(L1,L2) + 0.05) / (min(L1,L2) + 0.05)
        # get LAB colors
        source1 = ic[i*2+0]
        source2 = ic[i*2+1]
        # for each pair of available colors
        min_loss = 10000#infinity
        min_idx = (0,0)
        for j in range(len(cg)):
            target1 = cg[j]
            target1rgb = cgrgb[j]
            L1 = relative_luminance_srgb(target1rgb)
            for k in range(len(cg)):
                target2 = cg[k]
                target2rgb = cgrgb[k]
                L2 = relative_luminance_srgb(target2rgb)
                Ctarget = (max(L1,L2) + 0.05) / (min(L1,L2) + 0.05)
                # calculate the relative differences
                dE1 = delta_e_cie2000(source1, target1)#[0,infinity] a typical good value is < 6
                dE2 = delta_e_cie2000(source2, target2)#[0,infinity] a typical good value is < 6
                dC = max(C - Ctarget, 0)#[1,21]
                # if target1.lab_b < 0 and target2.lab_b < 0:
                #     print(dE1, dE2, dC)
                loss = dE1 + dE2 + dC * 10.0
                if loss < min_loss:
                    min_loss = loss
                    min_idx = (j,k)
                    # print('\tnew min')
        nc.append((min_loss,cgn[min_idx[0]], cgf[min_idx[0]][1], cgrgb[min_idx[0]]))
        nc.append((min_loss,cgn[min_idx[1]], cgf[min_idx[1]][1], cgrgb[min_idx[1]]))
    return nc

def delta_e_cie2000(color1, color2, Kl=1, Kc=1, Kh=1):
    """
    the package 'colormath' is not maintained and incompatible with numpy 1.18+ 
    Calculates the Delta E (CIE2000) of two colors.
    """
    from colormath.color_diff import _get_lab_color1_vector, _get_lab_color2_matrix
    from colormath import color_diff_matrix
    color1_vector = _get_lab_color1_vector(color1)
    color2_matrix = _get_lab_color2_matrix(color2)
    delta_e = color_diff_matrix.delta_e_cie2000(
        color1_vector, color2_matrix, Kl=Kl, Kc=Kc, Kh=Kh)[0]

    # return numpy.asscalar(delta_e)  <--- asscalar is deprecated
    return delta_e.item()