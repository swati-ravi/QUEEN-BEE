#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np  
from utils_2 import mullers_method
from tqdm import tqdm
from likelihood_bg_weights import likelihood_bg_weights, likelihood_polarbg_weights

def polar_likelihood_bg(q,u,ci,si,mu,area_ratio,count_bg):
    '''
    Compute the background-corrected polarization likelihood for 
    IXPE-style Stokes parameters using Muller's method to solve 
    for the normalization factor n0.

    Parameters
    ----------
    q : float
        Normalized Stokes parameter Q/I.
    u : float
        Normalized Stokes parameter U/I.
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
        Value of the background-corrected likelihood. If Muller's 
        method fails to converge, returns infinity.

    Notes
    -----
    - Estimates the source normalization `n0` by solving the 
      background-weighted likelihood equation with Muller's method.
    - The likelihood is given by:
          L = -2 * Σ log(n0 + n0*q*μ*ci + n0*u*μ*si + count_bg*area_ratio) 
              + 2*n0
    - Handles cases where the result is an ndarray, list, float, 
      or complex value. Complex results are reduced to their real part.
    '''
    # Estimate n0
    n0_est_ratio = np.array([0.7, 1.0, 1.3])
    n0_est = len(ci) - area_ratio * count_bg
    n0_input = n0_est * n0_est_ratio

    args = (q,u,ci,si,mu,area_ratio,count_bg)
    kwargs = {'q': q, 'u': u, 'ci': ci, 'si': si, 'mu': mu, 'area_ratio': area_ratio, 'count_bg': count_bg}
    muller_result = mullers_method(func=likelihood_bg_weights, xi=n0_input, double=True, itmax=1000000, stop=0, tol=0.01/n0_est, **kwargs) #original tolerance was 0.01/n0_est
    n0_solution = muller_result.root if muller_result.converged else None

    if muller_result.converged:
        n0_solution = muller_result.root
        likelihood = -2 * np.sum(np.log(n0_solution + n0_solution*q*mu*ci + n0_solution*u*mu*si + count_bg*area_ratio)) + 2*n0_solution
    else:
        likelihood = float('inf')
    
    if isinstance(likelihood, (np.ndarray, list)):
        return float(likelihood[0])
    elif isinstance(likelihood, float):
        return likelihood 
    elif isinstance(likelihood, complex):
        return np.real(likelihood)
    else: 
        raise TypeError("likelihood must be an np.ndarray, list, or float... but it is {0}".format(type(likelihood)))

def polar_likelihood_polarbg(q,u,qb,ub,ci,si,cj,sj,mu,mu_bg,area_ratio,count_bg):
    '''
    Compute the polarization likelihood for IXPE-style Stokes parameters 
    with explicit background polarization modeling.

    Parameters
    ----------
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
    cj : array_like
        Cosine terms of photon azimuthal angles (background events).
    sj : array_like
        Sine terms of photon azimuthal angles (background events).
    mu : float
        Instrument modulation factor for the source.
    mu_bg : float
        Instrument modulation factor for the background.
    area_ratio : float
        Ratio of the source to background extraction areas.
    count_bg : int
        Number of background counts.

    Returns
    -------
    float
        Value of the polarization likelihood with background included. 
        If Muller's method fails to converge, returns infinity.

    Notes
    -----
    - Estimates the source normalization `n0` via Muller's method, using 
      trial values scaled from an initial estimate.
    - Likelihood includes both source and background polarization terms:
      
      L = -2 * Σ log( n0 + n0(q μ ci + u μ si) 
                      + count_bg * area_ratio (1 + qb μ ci + ub μ si) )
          + 2n0 
          - 2 * Σ log(1 + qb μ_bg cj + ub μ_bg sj)

    - Handles ndarray, list, float, and complex results, reducing complex 
      values to their real component.
    '''

    # Estimate n0
    n0_est_ratio = np.array([0.7, 1.0, 1.3])
    n0_est = len(ci) - area_ratio * count_bg
    n0_input = n0_est * n0_est_ratio

    args = (q,u,qb,ub,ci,si,mu,area_ratio,count_bg)
    kwargs = {'q': q, 'u': u, 'qb': qb, 'ub': ub, 'ci': ci, 'si': si, 'mu': mu, 'area_ratio': area_ratio, 'count_bg': count_bg}
    muller_result = mullers_method(func=likelihood_polarbg_weights, xi=n0_input, double=True, itmax=500, stop=0, tol=0.01/n0_est, **kwargs) #original tolerance was 0.01/n0_est
    n0_solution = muller_result.root if muller_result.converged else None

    if muller_result.converged:
        n0_solution = muller_result.root
        likelihood = -2 * np.sum(np.log(n0_solution + n0_solution*q*mu*ci + n0_solution*u*mu*si + count_bg*area_ratio*(1+qb*mu*ci+ub*mu*si))) + 2*n0_solution - 2*np.sum(np.log(1+qb*mu_bg*cj+ub*mu_bg*sj))
    else:
        likelihood = float('inf')
    
    if isinstance(likelihood, (np.ndarray, list)):
        return float(likelihood[0])
    elif isinstance(likelihood, float):
        return likelihood 
    elif isinstance(likelihood, complex):
        return np.real(likelihood)
    else: 
        raise TypeError("likelihood must be an np.ndarray, list, or float... but it is {0}".format(type(likelihood)))









