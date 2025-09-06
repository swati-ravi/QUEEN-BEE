#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from qu_likelihood_bg import qu_likelihood_bg
import logging
logger = logging.getLogger("queen-bee")

def polarization_bg_radii(r_max, r, q, u, weight, mu, mu0, nbg, nbg_wt, area_ratios):
    '''
    Compute polarization quantities as a function of radial extraction 
    region, including background subtraction and weighting.

    Parameters
    ----------
    r_max : array_like
        Maximum extraction radii to evaluate.
    r : array_like
        Radial distances of individual events.
    q : array_like
        Event Stokes Q/I values.
    u : array_like
        Event Stokes U/I values.
    weight : array_like
        Event weights (e.g., exposure or quality factors).
    mu : array_like
        Event-by-event modulation factors (weighted).
    mu0 : array_like
        Event-by-event modulation factors (unweighted).
    nbg : int or float
        Total number of background counts.
    nbg_wt : array_like
        Weights associated with background events.
    area_ratios : array_like
        Ratios of source-to-background extraction areas, one per radius.

    Returns
    -------
    tuple
        A collection of polarization results for each radius:
        - nsrc : ndarray
            Number of source counts within each extraction radius.
        - qu_simple : ndarray, shape (2, N)
            Simple (unweighted) Q and U estimates.
        - qu_simple_err : ndarray, shape (2, N)
            Errors on simple Q and U estimates.
        - qu : ndarray, shape (2, N)
            Background-subtracted Q and U estimates.
        - qu_err : ndarray, shape (2, N)
            Errors on Q and U estimates.
        - poln : ndarray
            Polarization degree estimates.
        - poln_err : ndarray
            Errors on polarization degree.
        - evpa : ndarray
            Electric vector position angle (EVPA) estimates, in degrees.
        - evpa_err : ndarray
            Errors on EVPA, in degrees.
        - qu_wt : ndarray, shape (2, N)
            Weighted Q and U estimates.
        - qu_wt_err : ndarray, shape (2, N)
            Errors on weighted Q and U estimates.
        - poln_wt : ndarray
            Weighted polarization degree.
        - poln_wt_err : ndarray
            Errors on weighted polarization degree.

    Notes
    -----
    - Loops over each extraction radius in `r_max` and selects events 
      with r ≤ r_max[i].
    - Calls `qu_likelihood_bg` internally to compute background-corrected 
      Q/U, polarization degree, and EVPA for each subset.
    - Results include both simple (unweighted) and weighted estimates.
    - Useful for optimizing source extraction radii and evaluating 
      systematic effects of background subtraction.
    '''
    nradii = len(r_max)

    # Initialize output variables
    nsrc = np.zeros(nradii)
    qu_simple = np.zeros((2, nradii))
    qu_simple_err = np.zeros((2, nradii))
    qu = np.zeros((2, nradii))
    qu_err = np.zeros((2, nradii))
    poln = np.zeros(nradii)
    poln_err = np.zeros(nradii)
    evpa = np.zeros(nradii)
    evpa_err = np.zeros(nradii)
    #delta_like = np.zeros(nradii)
    qu_wt = np.zeros((2, nradii))
    qu_wt_err = np.zeros((2, nradii))
    poln_wt = np.zeros(nradii)
    poln_wt_err = np.zeros(nradii)

    # Set common block variables
    count_bg = float(nbg)
    count_bg_wt = nbg_wt

    # Run likelihood calculation for each radius
    for i in range(nradii):
        src = np.where(r <= r_max[i])
        nsrc[i] = len(src[0])
        evtq = q[src]
        evtu = u[src]
        evt_wt = weight[src]
        evt_mu = mu[src]    # Weighted mu
        evt_mu0 = mu0[src]  # Unweighted mu
        area_ratio = area_ratios[i]
        logger.debug(f"variables are len(evtq)={len(evtq)} len(evtu)={len(evtu)} count_bg={count_bg}")

        qu_simple1, qu_simple_err1, qu1, qu_err1, poln1, poln_err1, evpa1, evpa_err1, qu_wt1, qu_wt_err1, poln_wt1, poln_wt_err1 = qu_likelihood_bg(evtq, evtu, evt_wt, evt_mu, evt_mu0, area_ratio, count_bg, count_bg_wt)

        qu_simple[:, i] = qu_simple1
        qu_simple_err[:, i] = qu_simple_err1
        qu[:, i] = qu1
        qu_err[:, i] = qu_err1
        poln[i] = poln1
        poln_err[i] = poln_err1
        evpa[i] = evpa1
        evpa_err[i] = evpa_err1
        #delta_like[i] = dlike1
        qu_wt[:, i] = qu_wt1
        qu_wt_err[:, i] = qu_wt_err1
        poln_wt[i] = poln_wt1
        poln_wt_err[i] = poln_wt_err1

        #print(r_max[i], poln[i], poln_err[i], poln_wt[i], poln_wt_err[i])

    return nsrc, qu_simple, qu_simple_err, qu, qu_err, poln, poln_err, evpa, evpa_err, qu_wt, qu_wt_err, poln_wt, poln_wt_err
