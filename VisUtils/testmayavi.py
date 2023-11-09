# import matplotlib
# matplotlib.use('Agg')
import os
#https://matplotlib.org/stable/api/backend_qt_api.html
#os.environ['QT_API'] = 'PyQt6'
import cv2
import sys
import argparse
import os
#import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = (25, 14)
#import mayavi.mlab as mlab
from mayavi import mlab

#test Mayavi
def testMayavi1():
    # Create the data.
    from numpy import pi, sin, cos, mgrid
    dphi, dtheta = pi/250.0, pi/250.0
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
    r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
    x = r*sin(phi)*cos(theta)
    y = r*cos(phi)
    z = r*sin(phi)*sin(theta)

    # View it.
    s = mlab.mesh(x, y, z)
    mlab.show()

import numpy as np
from mayavi.mlab import *
def test_plot3d():#https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#plot3d
    """Generates a pretty set of lines."""
    n_mer, n_long = 6, 11
    dphi = np.pi / 1000.0
    phi = np.arange(0.0, 2 * np.pi + 0.5 * dphi, dphi)
    mu = phi * n_mer
    x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    y = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    z = np.sin(n_long * mu / n_mer) * 0.5

    l = mlab.plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap='Spectral')
    mlab.show()
    #return l

def test_plot3d2():#https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#plot3d
    """Generates a axis."""
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    axes = np.array(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)#white
    mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),#red, X (0,0,0)->(2,0,0)
        tube_radius=None,
        figure=fig,
    )
    mlab.text3d(axes[0, 0], axes[0, 1], axes[0, 2], "X", scale=(0.1, 0.1, 0.1)) #(2,0,0) position

    mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),#green, Y ()
        tube_radius=None,
        figure=fig,
    )
    mlab.text3d(axes[1, 0], axes[1, 1], axes[1, 2], "Y", scale=(0.1, 0.1, 0.1)) #(0,2,0) position

    mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),#blue, z
        tube_radius=None,
        figure=fig,
    )
    mlab.text3d(axes[2, 0], axes[2, 1], axes[2, 2], "Z", scale=(0.1, 0.1, 0.1)) #(0,0,2) position
    mlab.show()

if __name__ == '__main__':
    testMayavi1()
    #test_plot3d2()