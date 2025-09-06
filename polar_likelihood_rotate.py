#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from frame_rotate import frame_rotate
from likelihood_bg_weights import likelihood_bg_weights, likelihood_polarbg_weights
import utils_2

def polar_likelihood_rotate(param,ci,si,mu,t,tref,area_ratio,count_bg):
    '''
    Compute the polarization likelihood for a rotating EVPA (polarization angle) 
    model in IXPE-style Stokes parameter space.

    Parameters
    ----------
    param : tuple of floats
        Model parameters (q, u, r):
        - q : normalized Stokes parameter Q/I for the source.
        - u : normalized Stokes parameter U/I for the source.
        - r : EVPA rotation rate in degrees per unit time.
    ci : array_like
        Cosine terms of photon azimuthal angles (source events).
    si : array_like
        Sine terms of photon azimuthal angles (source events).
    mu : float
        Instrument modulation factor.
    t : array_like
        Photon arrival times.
    tref : float
        Reference time for applying EVPA rotation.
    area_ratio : float
        Ratio of the source to background extraction areas.
    count_bg : int
        Number of background counts.

    Returns
    -------
    float
        Value of the polarization likelihood under a rotating EVPA model.
        If Muller's method fails to converge, returns infinity.

    Notes
    -----
    - Introduces a rotation in the polarization frame:
          θ = 2 r (t - tref) in radians
    - Rotates the Stokes event frame by θ using `frame_rotate`.
    - Estimates the source normalization `n0` via Muller's method, with 
      trial values scaled from an initial estimate.
    - Likelihood is given by:

      L = -2 * Σ log( n0 + n0(q μ ci_rot + u μ si_rot) + count_bg * area_ratio )
          + 2 n0

    - Handles ndarray, list, float, and complex results, reducing complex 
      values to their real part.
    '''

    q, u, r = param 

    theta = 2*r*(t-tref)*np.deg2rad(1)

    ### rotate according to theta
    ci_rot,si_rot = frame_rotate(ci,si,theta)

    ### estimating n0
    n0_est_ratio = np.array([0.7, 1.0, 1.3])
    n0_est = len(ci) - area_ratio * count_bg
    n0_input = n0_est * n0_est_ratio
    kwargs = {'q': q, 'u': u, 'ci': ci_rot, 'si': si_rot, 'mu': mu, 'area_ratio': area_ratio, 'count_bg': count_bg}
    muller_result = utils.mullers_method(func=likelihood_bg_weights, xi=n0_input, double=True, itmax=1000, stop=0, tol=0.01/n0_est, **kwargs)
    n0_solution = muller_result.root if muller_result.converged else None 

    ### form the likelihood
    if muller_result.converged:
        likelihood = -2 * np.sum( np.log( n0_solution + n0_solution*q*mu*ci_rot + n0_solution*u*mu*si_rot + count_bg*area_ratio ) ) + 2*n0_solution
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

def polar_likelihood_rotate_polarbg(param,ci,si,cj,sj,mu,mu_bg,t,tref,area_ratio,count_bg):
    '''
    Compute the polarization likelihood for a rotating EVPA model 
    including explicit background polarization contributions.

    Parameters
    ----------
    param : tuple of floats
        Model parameters (q, u, qb, ub, r):
        - q : normalized Stokes parameter Q/I for the source.
        - u : normalized Stokes parameter U/I for the source.
        - qb : normalized Stokes parameter Q/I for the background.
        - ub : normalized Stokes parameter U/I for the background.
        - r : EVPA rotation rate in degrees per unit time.
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
    t : array_like
        Photon arrival times.
    tref : float
        Reference time for applying EVPA rotation.
    area_ratio : float
        Ratio of the source to background extraction areas.
    count_bg : int
        Number of background counts.

    Returns
    -------
    float
        Value of the polarization likelihood for a rotating EVPA model 
        with background polarization included. If Muller's method fails 
        to converge, returns infinity.

    Notes
    -----
    - Introduces a rotation in the polarization frame:
          θ = 2 r (t - tref) in radians
    - Rotates the Stokes event frame by θ using `frame_rotate`.
    - Estimates the source normalization `n0` via Muller's method, 
      starting from scaled trial values.
    - Likelihood is given by:

      L = -2 * Σ log( n0 + n0(q μ ci_rot + u μ si_rot) 
                      + count_bg * area_ratio (1 + qb μ ci_rot + ub μ si_rot) )
          + 2 n0
          - 2 * Σ log(1 + qb μ_bg cj + ub μ_bg sj)

    - Handles ndarray, list, float, and complex results, reducing complex 
      values to their real part.

    '''
    q, u, qb, ub, r = param 

    theta = 2*r*(t-tref)*np.deg2rad(1)

    ### rotate according to theta
    ci_rot,si_rot = frame_rotate(ci,si,theta)

    ### estimating n0
    n0_est_ratio = np.array([0.7, 1.0, 1.3])
    n0_est = len(ci) - area_ratio * count_bg
    n0_input = n0_est * n0_est_ratio
    kwargs = {'q': q, 'u': u, 'qb': qb, 'ub':ub, 'ci': ci_rot, 'si': si_rot, 'mu': mu, 'area_ratio': area_ratio, 'count_bg': count_bg}
    muller_result = utils.mullers_method(func=likelihood_polarbg_weights, xi=n0_input, double=True, itmax=1000, stop=0, tol=0.01/n0_est, **kwargs)
    n0_solution = muller_result.root if muller_result.converged else None 

    ### form the likelihood
    if muller_result.converged:
        likelihood = -2 * np.sum( np.log( n0_solution + n0_solution*q*mu*ci_rot + n0_solution*u*mu*si_rot + count_bg*area_ratio*(1+qb*mu*ci_rot+ub*mu*si_rot) ) ) + 2*n0_solution - 2*np.sum(np.log(1+qb*mu_bg*cj+ub*mu_bg*sj))
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

def polar_likelihood_pvar(params, ci, si, mu, t, tref, area_ratio, count_bg):
    '''
    Compute the polarization likelihood for a model where the 
    polarization degree (PD) varies linearly with time.

    Parameters
    ----------
    params : list or tuple of floats
        Model parameters [q, u, p_rate]:
        - q : normalized Stokes parameter Q/I at reference time.
        - u : normalized Stokes parameter U/I at reference time.
        - p_rate : linear rate of change of polarization degree with time.
    ci : array_like
        Cosine terms of photon azimuthal angles.
    si : array_like
        Sine terms of photon azimuthal angles.
    mu : float
        Instrument modulation factor.
    t : array_like
        Photon arrival times.
    tref : float
        Reference time for applying polarization evolution.
    area_ratio : float
        Ratio of source to background extraction areas.
    count_bg : int or float
        Number of background counts.

    Returns
    -------
    float
        Log-likelihood value for the time-varying polarization model.

    Notes
    -----
    - Defines a time-dependent polarization degree:
          p(t) = sqrt(q² + u²) + p_rate * (t - tref)
    - Clips `p(t)` to remain within the physical range [0, 1].
    - Current implementation uses a placeholder Gaussian likelihood:
          L = -0.5 Σ [ (p(t) - mean(p(t)))² / var(p(t)) ]
      which should be replaced with a full polarization likelihood 
      consistent with IXPE event distributions.
    '''
    q, u, p_rate = params
    delta_t = t - tref
    p_t = np.sqrt(q**2 + u**2) + p_rate * delta_t  # Evolving polarization fraction
    p_t = np.clip(p_t, 0, 1)  # Ensure valid range

    # Compute likelihood (replace with your actual calculation)
    likelihood = -0.5 * np.sum((p_t - np.mean(p_t))**2 / np.var(p_t))

    return likelihood


