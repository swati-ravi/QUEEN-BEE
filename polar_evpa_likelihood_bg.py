#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from polar_likelihood_bg import polar_likelihood_bg

def polar_evpa_likelihood_bg(params,ci,si,mu,area_ratio,count_bg):
    '''
    Compute the polarization likelihood in EVPA–PD parameter space 
    (polar form) by transforming to Stokes q–u space and evaluating 
    with `polar_likelihood_bg`.

    Parameters
    ----------
    params : list or tuple of floats
        [p, evpa]:
        - p : polarization degree (PD).
        - evpa : electric vector position angle (EVPA) in degrees.
    ci : array_like
        Cosine terms of photon azimuthal angles.
    si : array_like
        Sine terms of photon azimuthal angles.
    mu : float
        Instrument modulation factor.
    area_ratio : float
        Ratio of source to background extraction areas.
    count_bg : int
        Number of background counts.

    Returns
    -------
    float
        Background-corrected polarization likelihood evaluated in 
        EVPA–PD space.

    Notes
    -----
    - Transforms the parameters into normalized Stokes space:
          q = p cos(2 × EVPA)
          u = p sin(2 × EVPA)
    - Calls `polar_likelihood_bg(q, u, ...)` to compute the likelihood.
    - The EVPA is defined in degrees and converted internally to radians.
    '''
    # Transform the parameters to q-u space
    # Note that param[1] is the EVPA, not the Stokes phase
    param_pd, param_evpa = params
    param1 = np.zeros(2)
    param1[0] = param_pd * np.cos(2 * param_evpa * np.deg2rad(1))
    param1[1] = param_pd * np.sin(2 * param_evpa * np.deg2rad(1))

    # Form likelihood
    likelihood = polar_likelihood_bg(param1[0],param1[1],ci,si,mu,area_ratio,count_bg)

    return likelihood
