#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from tqdm import tqdm

def likelihood_bg_weights(n0,q,u,ci,si,mu,area_ratio,count_bg):
    '''
    Compute the background-weighted likelihood function used in solving 
    for the source normalization parameter n0 in IXPE-style polarization analysis.

    Parameters
    ----------
    n0 : float or array_like
        Trial values of the source normalization parameter.
    q : float
        Normalized Stokes parameter Q/I for the source.
    u : float
        Normalized Stokes parameter U/I for the source.
    ci : array_like
        Cosine terms of photon azimuthal angles.
    si : array_like
        Sine terms of photon azimuthal angles.
    mu : array_like
        Instrument modulation factor(s), reshaped internally for broadcasting.
    area_ratio : float
        Ratio of source to background extraction areas.
    count_bg : int
        Number of background counts.

    Returns
    -------
    float or ndarray
        Weighted sum function (Σ wᵢ - 1.0) used as the root-finding 
        target in Muller's method. A zero crossing corresponds to the 
        best-fit normalization n0.

    Notes
    -----
    - Implements the weight definition:

          wᵢ = [ n0 + n0 (q μ cᵢ + u μ sᵢ) + count_bg × area_ratio ]⁻¹

      and computes:

          Σ wᵢ - 1.0

    - This function is called iteratively by Muller's method within 
      likelihood evaluations (e.g., `polar_likelihood_bg`).
    - Handles both scalar and vector inputs for `n0`.
    '''
    nevt = len(ci) 
    sum_weights = np.zeros_like(n0)

    mu = mu.reshape(-1, 1)  # Reshaping mu to make its dimension compatible for broadcasting
    ci = ci.reshape(-1, 1)  
    si = si.reshape(-1, 1)  

    sum_weights = np.sum(1.0 / (n0 + n0*q*mu*ci + n0*u*mu*si + count_bg*area_ratio), axis=0)

    return sum_weights-1.0

def likelihood_polarbg_weights(n0,q,u,qb,ub,ci,si,mu,area_ratio,count_bg):
    '''
    Compute the background-weighted likelihood function including 
    polarized background contributions, used to solve for the 
    source normalization parameter n0 in IXPE-style polarization analysis.

    Parameters
    ----------
    n0 : float or array_like
        Trial values of the source normalization parameter.
    q : float
        Normalized Stokes parameter Q/I for the source.
    u : float
        Normalized Stokes parameter U/I for the source.
    qb : float
        Normalized Stokes parameter Q/I for the background.
    ub : float
        Normalized Stokes parameter U/I for the background.
    ci : array_like
        Cosine terms of photon azimuthal angles (source events).
    si : array_like
        Sine terms of photon azimuthal angles (source events).
    mu : array_like
        Instrument modulation factor(s), reshaped internally for broadcasting.
    area_ratio : float
        Ratio of source to background extraction areas.
    count_bg : int
        Number of background counts.

    Returns
    -------
    float or ndarray
        Weighted sum function (Σ wᵢ - 1.0) used as the root-finding 
        target in Muller's method. A zero crossing corresponds to the 
        best-fit normalization n0.

    Notes
    -----
    - Implements the weight definition:

          wᵢ = [ n0 + n0(q μ cᵢ + u μ sᵢ) 
                  + count_bg × area_ratio (1 + q_b μ cᵢ + u_b μ sᵢ) ]⁻¹

      and computes:

          Σ wᵢ - 1.0

    - Extends `likelihood_bg_weights` by including background 
      polarization terms `(q_b, u_b)`.
    - Handles both scalar and vector inputs for `n0`.
    '''
    nevt = len(ci) 
    sum_weights = np.zeros_like(n0)

    mu = mu.reshape(-1, 1)  # Reshaping mu to make its dimension compatible for broadcasting
    ci = ci.reshape(-1, 1)  
    si = si.reshape(-1, 1)  

    sum_weights = np.sum(1.0 / (n0 + n0*q*mu*ci + n0*u*mu*si + count_bg*area_ratio*(1 + qb*mu*ci + ub*mu*si) ), axis=0)

    return sum_weights-1.0














