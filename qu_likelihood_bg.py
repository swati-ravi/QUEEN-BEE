#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from scipy.optimize import minimize 
from numpy.linalg import inv
import frame_rotate as frame_rotate
from pderiv2 import pderiv2
from polar_likelihood_bg import polar_likelihood_bg
from polar_evpa_likelihood_bg import polar_evpa_likelihood_bg

def wrapper_for_minimize(param, ci, si, mu, area_ratio, count_bg):
    '''
    Wrapper function for optimization routines (e.g., scipy.optimize.minimize) 
    that evaluates the background-corrected polarization likelihood in q–u space.

    Parameters
    ----------
    param : array_like
        Model parameters [q, u]:
        - q : normalized Stokes parameter Q/I.
        - u : normalized Stokes parameter U/I.
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
        Value of the background-corrected polarization likelihood 
        from `polar_likelihood_bg`.
    '''
    q, u = param  # Unpack q and u from param array
    return polar_likelihood_bg(q, u, ci, si, mu, area_ratio, count_bg)

def wrapper_polar_evpa_likelihood_bg(param_pd, param_evpa, *args):
    '''
    Wrapper function to evaluate the background-corrected polarization 
    likelihood in EVPA–PD parameter space.

    Parameters
    ----------
    param_pd : float
        Polarization degree (PD).
    param_evpa : float
        Electric vector position angle (EVPA), in degrees.
    *args : tuple
        Additional arguments passed directly to `polar_evpa_likelihood_bg`.

    Returns
    -------
    float
        Value of the background-corrected polarization likelihood 
        from `polar_evpa_likelihood_bg`.
    '''
    params = [param_pd, param_evpa]
    return polar_evpa_likelihood_bg(params, *args)

def qu_likelihood_bg(qi,ui,wi,mu_i,mu0_i,area_ratio,count_bg,count_bg_wt):
    '''
    Estimate Stokes parameters q and u, polarization degree (PD), 
    and electric vector position angle (EVPA) using a likelihood 
    method with background subtraction.

    Parameters
    ----------
    qi : array_like
        Event Stokes Q contributions (cosine terms).
    ui : array_like
        Event Stokes U contributions (sine terms).
    wi : array_like
        Event weights (e.g., exposure or quality factors).
    mu_i : array_like
        Weighted modulation factors for each event.
    mu0_i : array_like
        Unweighted modulation factors for each event.
    area_ratio : float
        Ratio of source to background extraction areas.
    count_bg : int
        Total number of background counts.
    count_bg_wt : array_like
        Weights associated with background events.

    Returns
    -------
    tuple
        Polarization estimates in both Stokes and PD/EVPA space:
        - qu_simple : ndarray, shape (2,)
            Simple (unweighted) Q and U estimates.
        - qu_simple_err : ndarray, shape (2,)
            Errors on simple Q and U estimates.
        - qu : ndarray, shape (2,)
            Maximum-likelihood Q and U estimates.
        - qu_err : ndarray, shape (2,)
            Errors on Q and U estimates from the inverse Hessian.
        - poln : float
            Maximum-likelihood polarization degree.
        - poln_err : float
            Error on polarization degree.
        - evpa : float
            Maximum-likelihood electric vector position angle (degrees).
        - evpa_err : float
            Error on EVPA (degrees).
        - qu_wt : ndarray, shape (2,)
            Weighted Q and U estimates (Di Marco method).
        - qu_wt_err : ndarray, shape (2,)
            Errors on weighted Q and U estimates.
        - poln_wt : float
            Weighted polarization degree.
        - poln_wt_err : float
            Error on weighted polarization degree.

    Notes
    -----
    - Implements both weighted (Di Marco method) and likelihood-based 
      estimates of q, u, PD, and EVPA.
    - Likelihood optimization is performed via Nelder–Mead minimization 
      on both q–u and PD–EVPA parameterizations.
    - Errors are derived from the inverse Hessian matrix evaluated at 
      the likelihood minimum.
    - Includes background subtraction through `area_ratio`, `count_bg`, 
      and `count_bg_wt`.
    - Returns parallel results for:
        (a) simple unweighted estimates,
        (b) weighted estimates,
        (c) maximum-likelihood estimates.
    '''
    #global qudata, ci, si, mu, bg_data, count_bg, area_ratio, count_bg_wt

    ci = qi #evtq from the events
    si = ui #evtu from the events
    mu = mu0_i #mu0 from the events

    ##### Computing the weighted q,u using the Di Marco method
    var_i = np.sum(wi)
    w2 = np.sum(wi**2)
    bg_wt = area_ratio * count_bg_wt
    n_eff = var_i**2/w2 
    net_var_i = var_i - bg_wt 

    ##### Computing weighted q, u, and p 
    mubar = np.mean(mu_i)
    qu_wt = np.array([ np.sum(2*wi*ci)/var_i/mubar, np.sum(2*wi*si)/var_i/mubar ])
    poln_wt = np.sqrt( np.sum(qu_wt**2) )

    qu_wt_err = np.array([ np.sqrt(w2 * ( 2/mubar**2 - qu_wt[0]**2 ) ) / net_var_i, np.sqrt( w2 * ( 2/mubar**2 - qu_wt[1]**2 ) )/ net_var_i ])
    poln_wt_err = np.sqrt(w2 * ( 2/mubar**2 - poln_wt**2 ) ) / (net_var_i - 1)

    ##### Unweighted modulation factor 
    evtq = ci * mu0_i 
    evtu = si * mu0_i 

    ##### Get a nominal value of the likelihood for the entire sample 
    qu_simple_err = np.array([ 1.0/np.sqrt( np.sum(evtq**2) ), 1.0/np.sqrt( np.sum(evtu**2) )  ])
    qu_simple = np.array([ np.sum(evtq) * qu_simple_err[0]**2, np.sum(evtu) * qu_simple_err[1]**2 ])

    parm_est = qu_simple 
    err_est = qu_simple_err 

    like0 = polar_likelihood_bg(parm_est[0],parm_est[1],ci,si,mu,area_ratio,count_bg)
    tolerance = abs(0.001/like0)
    param = parm_est

    args = (ci,si,mu,area_ratio,count_bg)
    result = minimize(wrapper_for_minimize, param, method='Nelder-Mead', tol=tolerance, args=args)
    qu = result.x 

    ### Now for the uncertainties, estimated via the error matrix
    hessian = 0.5 * pderiv2(polar_likelihood_bg, qu, err_est, ci, si, mu, area_ratio, count_bg)

    try:
        err_matrix = inv(hessian)
    except np.linalg.LinAlgError:
        print("Inversion failed or small pivot -- accuracy questioned")

    qu_err = np.array([ np.sqrt(abs(err_matrix[0, 0])), np.sqrt(abs(err_matrix[1, 1])) ])

    # Examine the cross-correlation coefficient
    cross_corr = err_matrix[0, 1] / (qu_err[0] * qu_err[1])

    ## Compute the significance for this observation
    #delta_like = polar_likelihood_bg(0.0,0.0,ci,si,mu,area_ratio,count_bg) - np.min(likelihood)

    # Now use the polarization/EVPA version
    poln_est = np.sqrt(np.sum(qu**2))
    poln_err_est = np.sqrt(np.sum((qu * qu_err / poln_est)**2))
    evpa_est = 0.5 * np.arctan2(qu[1], qu[0]) * np.rad2deg(1)
    evpa_err_est = 0.5 * np.sqrt((qu[1]*qu_err[0])**2 + (qu[0]*qu_err[1])**2) / poln_est**2 * np.rad2deg(1)

    parm_est = [poln_est, evpa_est]
    err_est = [poln_err_est, evpa_err_est]
    like0 = polar_evpa_likelihood_bg(parm_est,ci,si,mu,area_ratio,count_bg)

    # Optimization with Nelder-Mead (similar to amoeba in IDL)
    result = minimize(polar_evpa_likelihood_bg, parm_est, method='Nelder-Mead', tol=tolerance, args=args)
    poln, evpa = result.x

    # Error estimation for polarization and EVPA
    hessian = 0.5 * pderiv2(wrapper_polar_evpa_likelihood_bg, [poln, evpa], err_est, ci, si, mu, area_ratio, count_bg)

    try:
        err_matrix = inv(hessian)
    except np.linalg.LinAlgError:
        print("Inversion failed or small pivot -- accuracy questioned")

    poln_err = np.sqrt(abs(err_matrix[0, 0]))
    evpa_err = np.sqrt(abs(err_matrix[1, 1]))

    return qu_simple, qu_simple_err, qu, qu_err, poln, poln_err, evpa, evpa_err, qu_wt, qu_wt_err, poln_wt, poln_wt_err



















