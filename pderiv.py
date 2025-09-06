#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np

def pderiv(func, x, i, dx, *args):
    '''
    Compute the numerical partial derivative of a function using 
    the central difference method.

    Parameters
    ----------
    func : callable
        Function of two variables (func(x0, x1, *args)) to be differentiated.
    x : array_like
        Point at which to evaluate the derivative (e.g., [x0, x1]).
    i : int
        Index of the variable with respect to which the derivative is taken.
        0 = first variable, 1 = second variable.
    dx : float
        Step size for the finite difference.
    *args : tuple
        Additional arguments to pass to `func`.

    Returns
    -------
    float
        Approximation of the partial derivative ∂f/∂x_i at the point `x`.

    Notes
    -----
    Uses the central difference formula:
        (f(x_i + dx/2) - f(x_i - dx/2)) / dx
    '''
    x0 = x.copy()
    x1 = x.copy()
    x0[i] = x0[i] - 0.5 * dx
    x1[i] = x1[i] + 0.5 * dx
    f0 = func(x0[0],x0[1],*args)
    f1 = func(x1[0],x1[1],*args)
    pder = (f1 - f0) / dx

    return pder

