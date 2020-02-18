# -*- coding: utf-8 -*-
"""
iran-matlab.ir

Kazem Gheysari

"""

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate universe variables
#   * Quality and service on subjective ranges [0, 10]
#   * Tip has a range of [0, 25] in units of percentage points
x_qual = np.arange(0, 11, .1)
x_serv = np.arange(0, 11, .1)
x_tip  = np.arange(0, 26, .1)

# Generate fuzzy membership functions
qual_lo = fuzz.trimf(x_qual, [0, 0, 5])
qual_md = fuzz.trimf(x_qual, [0, 5, 10])
qual_hi = fuzz.trimf(x_qual, [5, 10, 10])
serv_lo = fuzz.trimf(x_serv, [0, 0, 5])
serv_md = fuzz.trimf(x_serv, [0, 5, 10])
serv_hi = fuzz.trimf(x_serv, [5, 10, 10])
tip_lo = fuzz.trimf(x_tip, [0, 0, 13])
tip_md = fuzz.trimf(x_tip, [0, 13, 25])
tip_hi = fuzz.trimf(x_tip, [13, 25, 25])

upsampled = np.linspace(1, 10, 100)

x, y = np.meshgrid(upsampled, upsampled)
z = np.zeros_like(x)

for i in range(100):
    for j in range(100):
        qual_level_lo = fuzz.interp_membership(x_qual, qual_lo, x[i,j])
        qual_level_md = fuzz.interp_membership(x_qual, qual_md, x[i,j])
        qual_level_hi = fuzz.interp_membership(x_qual, qual_hi, x[i,j])

        serv_level_lo = fuzz.interp_membership(x_serv, serv_lo, y[i,j])
        serv_level_md = fuzz.interp_membership(x_serv, serv_md, y[i,j])
        serv_level_hi = fuzz.interp_membership(x_serv, serv_hi, y[i,j])

        active_rule1 = np.fmax(qual_level_lo, serv_level_lo)

        tip_activation_lo = np.fmin(active_rule1, tip_lo)  # removed entirely to 0

        tip_activation_md = np.fmin(serv_level_md, tip_md)
        
        active_rule3 = np.fmax(qual_level_hi, serv_level_hi)
        tip_activation_hi = np.fmin(active_rule3, tip_hi) 
        
        aggregated = np.fmax(tip_activation_lo,
                             np.fmax(tip_activation_md, tip_activation_hi))

        tip = fuzz.defuzz(x_tip, aggregated, 'centroid')
        tip_activation = fuzz.interp_membership(x_tip, aggregated, tip)  # for plot
        z[i, j] = tip


# Plot the result in pretty 3D with alpha blending
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                       linewidth=0.4, antialiased=True)


ax.view_init(30, 200)        