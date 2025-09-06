#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np

def frame_rotate(x,y,angle):
    '''
    Rotate Cartesian coordinates (x, y) by a specified angle in radians.

    Parameters
    ----------
    x : array_like or float
        Input x-coordinate(s).
    y : array_like or float
        Input y-coordinate(s).
    angle : float or array_like
        Rotation angle(s) in radians.

    Returns
    -------
    xp, yp : ndarray or float
        Rotated coordinates after applying the counter-clockwise rotation.

    Notes
    -----
    - Rotation follows a right-handed convention about the +z axis.
    - The transformation is:

          x' =  cos(angle) * x + sin(angle) * y
          y' = -sin(angle) * x + cos(angle) * y

    - Positive `angle` corresponds to a counter-clockwise rotation 
      in the (x, y) plane.
    '''
    xp = np.cos(angle)*x + np.sin(angle)*y
    yp = -np.sin(angle)*x + np.cos(angle)*y

    return xp, yp
