#!/usr/bin/env python
# coding: utf-8
from __future__ import division, print_function
import numpy as np  
from scipy.interpolate import interp1d #,LinearNDInterpolator
from polarization_bg_radii import polarization_bg_radii
import bilby
from bilby.core.prior import (
    Prior, PriorDict, ConditionalPriorDict,
    Uniform, ConditionalUniform, Constraint, 
    DeltaFunction, Gaussian
)
from astropy.io import fits
import mpmath
import copy
import dynesty
import corner
import matplotlib.pyplot as plt
from utils_2 import PolarLikelihoodBG, PolarLikelihoodRotate, PolarLikelihood_PolarBG, PolarLikelihoodRotate_PolarBG
import argparse

__version__ = "3.2"  

WELCOME_BANNER = f"""
============================================================
  Welcome to QUEEN-BEE
  Version: {__version__}
  Authors: Swati Ravi, Mason Ng, and Herman Marshall
------------------------------------------------------------
  Copyright (C) 2025, the QUEEN-BEE team.

  License: This is free research software provided as-is for scientific use.
  You are welcome to redistribute it under certain conditions.
  Licensed under the MIT License (see LICENSE file for details).

  Disclaimer: QUEEN-BEE comes with ABSOLUTELY NO WARRANTY.

  Contact: swatir@mit.edu
============================================================
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("queen-bee")

modfact = 'ixpe_d1_obssim20230702_mfact_v013.fits'
eventdir = '/Users/swatiravi/Downloads/02006801/event_l2/ixpe02006801_det{0}_evt2_v01.fits'
emin = 2.0
emax = 8.0
x_center = 306.12195
y_center = 302.08328
reg = 60
r_innerbg = 150
r_outerbg = 300
source = 'gx13+1'
modfact_directory = '/Users/swatiravi/anaconda3/lib/python3.11/site-packages/ixpeobssim/caldb/ixpe/gpd/cpf/modfact/'


mpmath.mp.dps = 50 # decimal digits of precision

def convert_to_radial(parameters):
    '''
    Convert Cartesian Stokes parameters (q, u) into a radial 
    polarization parameterization.

    Parameters
    ----------
    parameters : dict
        Dictionary of parameters containing at least:
        - 'q' : float, normalized Stokes parameter Q/I.
        - 'u' : float, normalized Stokes parameter U/I.

    Returns
    -------
    dict
        Copy of the input dictionary with an additional entry:
        - 'r' : float, radial polarization parameter defined as 
                r = q² + u².

    Notes
    -----
    - This function is typically used for defining conditional priors 
      in `bilby` analyses (see: 
      https://lscsoft.docs.ligo.org/bilby/conditional_priors.html).
    - High-precision arithmetic is enabled by setting 
      `mpmath.mp.dps = 50` for numerical stability in downstream 
      Bayesian inference.
    '''
    p = parameters.copy()
    p['r'] = p['q']**2 + p['u']**2
    return p


def setup(tmin=None,tmax=None):
    '''
    Load and preprocess IXPE event data with optional time filtering

        Parameters
    ----------
    tmin : float, optional
        Minimum time (relative to TSTART) for event selection. 
        If None, no lower time bound is applied.
    tmax : float, optional
        Maximum time (relative to TSTART) for event selection. 
        If None, no upper time bound is applied.

    Returns
    -------
    tuple
        A collection of event- and region-specific quantities:
        - r_max : ndarray
            Maximum source extraction radius (arcsec).
        - r : ndarray
            Radial distances of events from the source center (arcsec).
        - evtq : ndarray
            Event Stokes Q contributions.
        - evtu : ndarray
            Event Stokes U contributions.
        - evtq_bg : ndarray
            Background Stokes Q contributions.
        - evtu_bg : ndarray
            Background Stokes U contributions.
        - wt1 : ndarray
            Event weights.
        - mu2 : ndarray
            Event-by-event modulation factors with corrections applied.
        - mu_bg : ndarray
            Modulation factors for background events.
        - mu1 : ndarray
            Energy-dependent modulation factors (linear interpolation).
        - mu0 : ndarray
            Unweighted modulation factors from calibration model.
        - nbg : int
            Number of background events.
        - nbg_wt : float
            Weighted number of background events.
        - area_ratio : float
            Ratio of source to background extraction areas.
        - times : ndarray
            Event times relative to TSTART (filtered if `tmin`/`tmax` given).

    Notes
    -----
    - Loads IXPE event FITS files defined by `eventdir`.
    - Applies energy cuts (`emin`, `emax`) and quality cuts 
      derived from `bg_flag.pro`.
    - Computes:
        * Event energies and PIs.
        * Source modulation factors (`mu1`, `mu2`, `mu0`).
        * Background region events and weights.
    - Time filtering is applied if both `tmin` and `tmax` are set.
    - Returns variables suitable for downstream polarization and 
      likelihood calculations.
    '''
    ixpe_modfact = modfact_directory
    modf_w = fits.open(ixpe_modfact + modfact)['SPECRESP'].data
    energy_w = 0.5 * (modf_w['ENERG_LO'] + modf_w['ENERG_HI'])
    mu_w = modf_w['SPECRESP']

    modf_uw = fits.open(ixpe_modfact + modfact)['SPECRESP'].data
    energy_uw = 0.5 * (modf_uw['ENERG_LO'] + modf_uw['ENERG_HI'])
    mu_uw = modf_uw['SPECRESP']

    eventfiles = np.array([eventdir.format(i) for i in range(1,4)])

    header = fits.open(eventfiles[0])[1].header
    events = np.concatenate([fits.open(eventfiles[i])[1].data for i in range(len(eventfiles))])

    scale = header['TCDLT7'] if header['TCDLT7'] > 0 else header['TCDLT8']
    t0 = header['TSTART']
    arcsec_per_pix = scale * 3600

    events_pi = events['PI']
    events_energy = 0.04 * events_pi 
    alpha = events['W_MOM']**(4/3)

    ### From bg_flag.pro 
    a1 = 0.35 
    a2 = 0.7 
    nrg0 = 2.0 
    b1 = (1-a1)/(5.5-nrg0)
    b2 = (0.95-a2)/(8.0-nrg0)

    bad1 = (alpha > a1 + b1*(events_energy-nrg0))
    bad2 = (alpha > a2 + b2*(events_energy-nrg0))
    bad = bad1 | bad2 
    good = ~bad 

    conditions = (events_energy >= emin) & (events_energy <= emax) & good & (np.isfinite(events['X'])) & (np.isfinite(events['Y']))
    
    times = events['TIME'][conditions] - t0

    # Apply time filtering if `tmin` and `tmax` are provided
    if tmin is not None and tmax is not None:
        time_mask = (times >= tmin) & (times <= tmax)
        times = times[time_mask]
    else:
        time_mask = np.ones_like(times, dtype=bool)  # Select all times if no filtering

    nevt = len(times)
    x = events['X'][conditions][time_mask]
    y = events['Y'][conditions][time_mask]
    energy = events_energy[conditions][time_mask]
    pi = events_pi[conditions][time_mask]
    evtq = 0.5 * events['Q'][conditions][time_mask]
    evtu = 0.5 * events['U'][conditions][time_mask]
    wt1 = events['W_MOM'][conditions][time_mask]

    alpha = wt1**(4/3)

    aa = -0.28 
    bb = 0.2 
    cc = 0.21 
    dd = 1.0/18.5 
    mu0 = (1.0 / ((-aa - bb * energy)**(-4) + (-cc - dd * energy)**(-4)))**0.25

    mu1 = interp1d(energy_w, mu_w, kind='linear', fill_value='extrapolate')(energy)
    mu2 = 0.05 + 0.80 * alpha

    center_extra = 0.476 + 0.0011 * pi 
    norm_extra = np.zeros(nevt)
    
    low_pi = np.where(pi < 66)
    norm_extra[low_pi] = -0.643 + 0.0124*pi[low_pi]

    mid_pi = np.where((pi>=66)&(pi<150))
    norm_extra[mid_pi] = 0.309 - 0.0021 * pi[mid_pi]

    sigma2_extra = np.ones(nevt)
    low_pi = np.where((pi>=55)&(pi<85))
    sigma2_extra[low_pi] = 0.02

    mid_pi = np.where((pi>=85)&(pi<150))
    sigma2_extra[mid_pi] = 0.03 

    delta_mu = norm_extra * np.exp( -0.5*(np.sqrt(alpha) - center_extra )**2 / sigma2_extra  )
    mu2 += delta_mu

    size = reg/arcsec_per_pix #60'' is the default source region size; can change 

    x0_best, y0_best = np.array([x_center, y_center])  # GX 13+1, Apr 2024

    dx = x - x0_best 
    dy = y - y0_best    
    r = np.sqrt(dx**2 + dy**2) * arcsec_per_pix
    pa = np.degrees(np.arctan2(-dx, dy))

    rmax_bg = r_outerbg #arcsec 
    rmin_bg = r_innerbg #arcsec 
    conditions_bg = (r > rmin_bg) & (r <= rmax_bg) 
    bg = r[conditions_bg]
    area_bg = np.pi * (rmax_bg**2 - rmin_bg**2)
    nbg = len(bg)
    nbg_wt = np.sum(wt1[conditions_bg])
    nbg_wt2 = np.sum(wt1[conditions_bg]**2)

    evtq_bg = 0.5 * events['Q'][conditions][time_mask][conditions_bg]
    evtu_bg = 0.5 * events['U'][conditions][time_mask][conditions_bg]
    mu_bg = mu2[conditions_bg]

    count_bg = nbg 
    count_bg_wt = nbg_wt 

    r_max = np.array([reg]) #arcsec; default source region size
    area_source = np.pi * r_max**2

    return r_max, r, evtq, evtu, evtq_bg, evtu_bg, wt1, mu2, mu_bg, mu1, mu0, nbg, nbg_wt, area_source/area_bg, times

def scout_unpolar():
    '''
    Compute the Bayesian log-evidence for an unpolarized source 
    (q = 0, u = 0) using Bilby nested sampling.

    This function fixes the normalized Stokes parameters q and u 
    to zero (within narrow Gaussian priors) and evaluates the 
    background-corrected polarization likelihood with Bilby.

    Returns
    -------
    None
        Prints the log-evidence and its uncertainty to stdout. 
        Results are also written to an output directory.

    Notes
    -----
    - Defines priors:
        * q ~ N(0, 1e-6)
        * u ~ N(0, 1e-6)
      These act as delta-function priors fixing q and u to 0.
    - Initializes `PolarLikelihoodBG` with fixed q and u.
    - Runs Bilby with the `dynesty` nested sampler:
        * nlive = 100 live points
        * Output directory = "{source}_scout_unpolar"
        * Label = "unpolarized"
    - Prints the recovered log-evidence and its uncertainty.
    - Useful as a baseline (null hypothesis) for model comparison 
      against polarized models via Bayes factors.
    '''
    fixed_q = 0.0
    fixed_u = 0.0

    # Define priors using a very narrow Gaussian instead of DeltaFunction
    priors = PriorDict()
    priors["q"] = Gaussian(fixed_q, 1e-6)  # Narrow Gaussian to approximate a fixed value
    priors["u"] = Gaussian(fixed_u, 1e-6)

    # Initialize likelihood
    likelihood = PolarLikelihoodBG(evtqq, evtuu, mu1, area_ratios[0], nbg, fixed_q=fixed_q, fixed_u=fixed_u)

    # Run Bilby nested sampling to compute log-evidence
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=100,
        outdir= source + "_scout_unpolar",
        label="unpolarized"
    )

    # Extract log-evidence and uncertainty
    log_evidence = result.log_evidence
    log_evidence_err = result.log_evidence_err

    print(f'Fixed q = {fixed_q}, Fixed u = {fixed_u}')
    print(f'Log-Evidence: {log_evidence:.3f} ± {log_evidence_err:.3f}')

def scout_const():
    '''
    Perform Bayesian inference of the normalized Stokes parameters 
    (q, u) for a constant polarization model using Bilby.

    This function runs nested sampling with uniform priors on (q, u) 
    constrained to the unit circle, computes credible intervals, 
    and visualizes the posterior distributions.

    Returns
    -------
    None
        Prints median and 90% credible intervals for q and u, and 
        generates a corner plot of the posterior samples.

    Notes
    -----
    - Priors:
        * q ~ Uniform(-1, 1)
        * u ~ Uniform(-1, 1)
        * r = q² + u² constrained to [0, 1]
      with a conversion applied by `convert_to_radial`.
    - Likelihood:
        Uses `PolarLikelihoodBG` with background correction.
    - Sampler:
        Runs Bilby with the `dynesty` nested sampler:
            * nlive = 100 live points
            * Output directory = "{source}_scout_const"
    - Posterior summaries:
        * Prints q and u median values with 90% credible intervals.
        * Generates and displays a corner plot of posterior distributions.
    '''
    # Define prior for parameters q and u on the unit circle
    uniform_circular_prior = PriorDict(dictionary=dict(q=Uniform(-1, 1),u=Uniform(-1, 1),r=Constraint(0, 1)),conversion_function=convert_to_radial)
    likelihood = PolarLikelihoodBG(evtqq,evtuu,mu1,area_ratios[0],nbg)
    result = bilby.run_sampler(likelihood,uniform_circular_prior,sampler="dynesty",nlive=100,outdir=source + "_scout_const")

    median_q = np.percentile(result.samples[:,0],50)
    lower_q = np.percentile(result.samples[:,0],5)
    upper_q = np.percentile(result.samples[:,0],95)

    median_u = np.percentile(result.samples[:,1],50)
    lower_u = np.percentile(result.samples[:,1],5)
    upper_u = np.percentile(result.samples[:,1],95)

    print('q = {0} +{1} -{2}'.format(median_q,upper_q-median_q,median_q-lower_q))
    print('u = {0} +{1} -{2}'.format(median_u,upper_u-median_u,median_u-lower_u))

    result.plot_corner()
    plt.show()

def scout_rot():
    '''
    Perform Bayesian inference of normalized Stokes parameters (q, u) and 
    the EVPA rotation rate using Bilby for a rotating polarization model.

    This function runs nested sampling with uniform priors on q, u, and 
    rotation rate, computes credible intervals, and visualizes the posterior.

    Returns
    -------
    None
        Prints median values with 90% credible intervals for q, u, and 
        the EVPA rotation rate, and generates a corner plot of the posteriors.

    Notes
    -----
    - Priors:
        * q ~ Uniform(-1, 1)
        * u ~ Uniform(-1, 1)
        * r = q² + u² constrained to [0, 1]
        * rotrate ~ Uniform(-200, 200) [deg/day]
    - Reference time `tref` is chosen as the midpoint of the observed times.
    - Likelihood:
        Uses `PolarLikelihoodRotate` with background correction and rotation.
    - Sampler:
        Runs Bilby with the `dynesty` nested sampler:
            * nlive = 100 live points
            * Output directory = "{source}_scout_rot"
    - Posterior summaries:
        * Prints q, u, and rotrate with 5th–95th percentile intervals (90% CI).
        * Generates and displays a corner plot of the posterior samples.
    '''
    tref = 0.5 * (np.min(ttimes) + np.max(ttimes))
    # Define prior for parameters q and u on the unit circle
    uniform_circular_prior = PriorDict(dictionary=dict(q=Uniform(-1, 1),u=Uniform(-1, 1),r=Constraint(0, 1),rotrate=Uniform(-200,200)),conversion_function=convert_to_radial)
    #uniform_circular_prior = PriorDict(dictionary=dict(q=Uniform(0.05, 0.10),u=Uniform(0.1, 0.2),r=Constraint(0, 1)),conversion_function=convert_to_radial)
    likelihood = PolarLikelihoodRotate(evtqq,evtuu,mu1,ttimes,tref,area_ratios[0],nbg)
    result = bilby.run_sampler(likelihood,uniform_circular_prior,sampler="dynesty",nlive=100,outdir=source + "scout_rot")

    median_q = np.percentile(result.samples[:,0],50)
    lower_q = np.percentile(result.samples[:,0],5)
    upper_q = np.percentile(result.samples[:,0],95)

    median_u = np.percentile(result.samples[:,1],50)
    lower_u = np.percentile(result.samples[:,1],5)
    upper_u = np.percentile(result.samples[:,1],95)

    median_r = np.percentile(result.samples[:,2],50)
    lower_r = np.percentile(result.samples[:,2],5)
    upper_r = np.percentile(result.samples[:,2],95)

    print('q = {0} +{1} -{2}'.format(median_q,upper_q-median_q,median_q-lower_q))
    print('u = {0} +{1} -{2}'.format(median_u,upper_u-median_u,median_u-lower_u))
    print('rotrate = {0} +{1} -{2}'.format(median_r,upper_r-median_r,median_r-lower_r))

    result.plot_corner(parameters=['q','u','rotrate'])
    plt.show()

MODEL_FUNCS = {
    "unpolarized": scout_unpolar,
    "constant":    scout_const,
    "rotating":    scout_rot,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="queen-bee",
        description="QUEEN-BEE — Bayesian EVPA Evolution Framework"
    )
    parser.add_argument(
        "model",
        choices=MODEL_FUNCS.keys(),
        help="which model to run: %(choices)s"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="increase output verbosity"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"QUEEN-BEE {__version__} \n"
                "Contact: swatir@mit.edu"
    )
    
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    
    # print banner at startup
    print(WELCOME_BANNER)

    ##### Getting the outputs from polarization_bg_radii.py
    r_max, r, evtq, evtu, evtq_bg, evtu_bg, weight, mu2, mu_bg, mu, mu0, nbg, nbg_wt, area_ratios, times = setup()
    logger.debug(f"area_ratios={area_ratios}, nbg={nbg}")

    nsrc, qu_simple, qu_simple_err, qu, qu_err, poln, poln_err, evpa, evpa_err, qu_wt, qu_wt_err, poln_wt, poln_wt_err = polarization_bg_radii(
        r_max, r, evtq, evtu, weight, mu, mu0, nbg, nbg_wt, area_ratios
    )
    logger.debug(f"qu_simple={qu_simple}, qu_simple_err={qu_simple_err}")
    logger.debug(f"qu={qu}, qu_err={qu_err}")
    logger.debug(f"qu_wt={qu_wt}, qu_wt_err={qu_wt_err}")
    logger.debug(f"poln={poln}, poln_err={poln_err}")
    logger.debug(f"evpa={evpa}, evpa_err={evpa_err}")

    # Derive arrays used by model functions
    q0 = qu[0, 0]
    u0 = qu[1, 0]
    qu_delta = qu_err[0, 0] / 2
    qu_halfrange = qu_err[0, 0] * 5.0

    qmin = q0 - qu_halfrange
    qmax = q0 + qu_halfrange
    nq = int(np.floor((qmax - qmin) / qu_delta + 1 + 0.5))
    qval = np.arange(nq) * qu_delta + qmin

    umin = u0 - qu_halfrange
    umax = u0 + qu_halfrange
    nu = int(np.floor((umax - umin) / qu_delta + 1 + 0.5))
    uval = np.arange(nu) * qu_delta + umin

    src = np.where(r <= r_max[0])  # r_max is a single element
    evtqq = evtq[src]
    evtuu = evtu[src]
    mu1 = mu2[src]
    ttimes = times[src]

    logger.info(f'Running model: "{args.model}" …')
    # call the right function
    MODEL_FUNCS[args.model]()
    logger.info(f'Finished model: "{args.model}".')





