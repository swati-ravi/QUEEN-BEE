#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from pderiv import pderiv

def pderiv2(func, x, dx, *args):
    '''
    Compute the numerical Hessian matrix (matrix of second-order 
    partial derivatives) of a multivariable function using the 
    central difference method.

    Parameters
    ----------
    func : callable
        Function to be differentiated. Must accept arguments as 
        func(x0, x1, ..., *args).
    x : array_like
        Point at which to evaluate the second derivatives.
    dx : array_like
        Step sizes for each variable, same length as `x`.
    *args : tuple
        Additional arguments to pass to `func`.

    Returns
    -------
    numpy.ndarray
        Symmetric Hessian matrix of shape (n, n), where n = len(x).
        Entry (i, j) gives ∂²f / ∂x_i ∂x_j evaluated at `x`.

    Notes
    -----
    - Uses central differences on top of the `pderiv` function 
      to approximate second derivatives.
    - Symmetry of the Hessian is enforced by mirroring the 
      upper triangle into the lower triangle.
    '''
    npar = len(x)
    pder2 = np.zeros((npar, npar))

    for i in range(npar):
        for j in range(i, npar):
            x0 = x.copy()
            x1 = x.copy()
            x0[i] = x0[i] - 0.5 * dx[i]
            x1[i] = x1[i] + 0.5 * dx[i]
            pd0 = pderiv(func, x0, j, dx[j], *args)
            pd1 = pderiv(func, x1, j, dx[j], *args)
            pder2[i, j] = (pd1 - pd0) / dx[i]

    for i in range(1, npar):
        for j in range(i):
            pder2[i, j] = pder2[j, i]

    return pder2
