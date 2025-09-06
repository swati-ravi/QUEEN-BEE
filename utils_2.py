#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
For Muller's method: https://github.com/apauley/numerical-analysis/blob/master/Chapter1/Python/muller.py

'''
from __future__ import division, print_function
import numpy as np  
from bilby.core.prior import (
    Prior, PriorDict, ConditionalPriorDict,
    Uniform, ConditionalUniform, Constraint)
import bilby
import polar_likelihood_bg
import polar_likelihood_rotate
from scipy.stats import semicircular

class MullersResult:
    def __init__(self, root, converged):
        self.root = root
        self.converged = converged

def mullers_method(func, xi, double=True, itmax=100, stop=0, tol=0.002, **kwargs):
    '''
    Find a root of a nonlinear function using Muller's method.

    Parameters
    ----------
    func : callable
        Function for which the root is sought. Must accept a scalar 
        argument and return a scalar value. Additional arguments can 
        be passed via `**kwargs`.
    xi : array_like, shape (3,)
        Initial guess values for the root. Must contain exactly three 
        starting points for the iteration.
    double : bool, optional
        Placeholder argument (not directly used in the current implementation). 
        Retained for compatibility with other code. Default is True.
    itmax : int, optional
        Maximum number of iterations. Default is 100.
    stop : int, optional
        Stopping criterion:
        - 0 : stop when successive root estimates converge within `tol`.
        - nonzero : stop when |f(root)| <= `tol`.
        Default is 0.
    tol : float, optional
        Convergence tolerance for the chosen stopping criterion. 
        Default is 0.002.
    **kwargs : dict
        Additional keyword arguments passed to `func`.

    Returns
    -------
    MullersResult
        A named tuple-like result containing:
        - root : float
            Estimated root of the function.
        - converged : bool
            True if the algorithm converged within the iteration limit, 
            False otherwise.

    Raises
    ------
    ValueError
        If `xi` does not contain exactly 3 initial guesses.
    RuntimeError
        If the algorithm fails to converge within `itmax` iterations.

    Notes
    -----
    - Muller's method uses quadratic interpolation through three points 
      to estimate the next root candidate.
    - The algorithm supports complex discriminants and returns real or 
      complex roots depending on the function.
    - Stopping can be controlled either by step size (root difference) 
      or by function value at the estimated root.
    '''
    x = np.array(xi)

    if len(x) != 3:
        raise ValueError('x must be a 3-element initial guess vector.')
    
    it = 0
    cond = False
    
    while it < itmax and not cond:
        q = (x[2] - x[1]) / (x[1] - x[0])
        pls = 1 + q
        f = np.array([func(x[0],**kwargs), func(x[1],**kwargs), func(x[2],**kwargs)])
        a = q * f[2] - q * pls * f[1] + q**2 * f[0]
        b = (2*q + 1) * f[2] - pls**2 * f[1] + q**2 * f[0]
        c = pls * f[2]
        disc = b**2 - 4*a*c
        
        if np.iscomplexobj(disc) or disc < 0:
            disc = complex(disc) if disc < 0 else disc
            r0 = b + np.sqrt(disc)
            r1 = b - np.sqrt(disc)
            div = r0 if abs(r0) > abs(r1) else r1
        else:
            rR0 = b + np.sqrt(disc)
            rR1 = b - np.sqrt(disc)
            div = rR0 if abs(rR0) >= abs(rR1) else rR1
        
        root = x[2] - (x[2] - x[1]) * (2 * c / div)

        #print('it is',it)
        #print('root-x[2] is',abs(root[0]-x[2]),tol,it)
        #print('tol is',tol)
        if stop == 0 and abs(root[0] - x[2]) <= tol:
            cond = True
            return MullersResult(root[0],True)
        else:
            evalFunc = func(root[0],**kwargs)
            if stop != 0 and abs(evalFunc) <= tol:
                cond = True
            elif evalFunc == 0:
                cond = True
        
        x = np.array([x[1], x[2], root[0]])
        it += 1

    if it >= itmax and not cond:
        raise RuntimeError('Algorithm failed to converge within given parameters.')

        return MullersResult(None,False)

    return MullersResult(root,True)

class PolarLikelihoodBG(bilby.core.likelihood.Likelihood):
    def __init__(self, ci, si, mu, area_ratio, count_bg, fixed_q=None, fixed_u=None):
        '''
        Initialize a background-corrected polarimetric likelihood object.
        Parameters
        ----------
        ci : array_like
            Cosine terms of photon azimuthal angles (cos(2ψ)).
        si : array_like
            Sine terms of photon azimuthal angles (sin(2ψ)).
        mu : float or array_like
            Instrument modulation factor(s).
        area_ratio : float
            Ratio of source to background extraction areas.
        count_bg : int or float
            Number of background counts.
        fixed_q : float, optional
            Fixed value of the normalized Stokes parameter Q/I. If provided, 
            this overrides `q` as a free parameter (e.g., for unpolarized models).
        fixed_u : float, optional
            Fixed value of the normalized Stokes parameter U/I. If provided, 
            this overrides `u` as a free parameter (e.g., for unpolarized models).

        Notes
        -----
        - Inherits from `PolarLikelihoodBG`.
        - By default, `q` and `u` are treated as free parameters in the likelihood.
        - If `fixed_q` or `fixed_u` are provided, those parameters are locked 
          to the specified values and removed from the free parameter set.
        - This class is intended for constructing polarization likelihoods 
          that include explicit background correction.
        '''
        super(PolarLikelihoodBG, self).__init__(parameters={'q': None, 'u': None})
        self.ci = ci
        self.si = si
        self.mu = mu
        self.area_ratio = area_ratio
        self.count_bg = count_bg
        self.fixed_q = fixed_q
        self.fixed_u = fixed_u
        
        # If fixed_q and fixed_u are provided, override parameters
        if self.fixed_q is not None:
            self.parameters['q'] = self.fixed_q
        if self.fixed_u is not None:
            self.parameters['u'] = self.fixed_u

    def log_likelihood(self):
        '''
        Compute the log-likelihood for the background-corrected 
        polarimetric model.

        Returns
        -------
        float
            Log-likelihood value evaluated at the current parameter 
            values (`q`, `u`).

        Notes
        -----
        - Retrieves `q` and `u` from `self.parameters`. If unset, they 
          default to 0.0 (unpolarized).
        - Calls `polar_likelihood_bg` to compute the likelihood.
        - Returns the negative of that likelihood so the result can be 
          used directly in statistical frameworks (e.g., nested sampling, 
          MCMC) where higher values correspond to better fits.
        '''
        q = self.parameters['q'] if self.parameters['q'] is not None else 0.0
        u = self.parameters['u'] if self.parameters['u'] is not None else 0.0
        
        likelihood = polar_likelihood_bg.polar_likelihood_bg(q, u,
                                         self.ci, self.si, self.mu,
                                         self.area_ratio, self.count_bg)
        return -likelihood

class PolarLikelihood_PolarBG(bilby.core.likelihood.Likelihood):
    def __init__(self, ci, si, cj, sj, mu, mu_bg, area_ratio, count_bg):
        '''
        Initialize a background-corrected polarimetric likelihood object 
        with explicit modeling of polarized background.

        Parameters
        ----------
        ci : array_like
            Cosine terms of photon azimuthal angles (source events).
        si : array_like
            Sine terms of photon azimuthal angles (source events).
        cj : array_like
            Cosine terms of photon azimuthal angles (background events).
        sj : array_like
            Sine terms of photon azimuthal angles (background events).
        mu : float or array_like
            Modulation factor(s) for source events.
        mu_bg : float or array_like
            Modulation factor(s) for background events.
        area_ratio : float
            Ratio of source to background extraction areas.
        count_bg : int or float
            Number of background counts.

        Notes
        -----
        - Inherits from `PolarLikelihood_PolarBG`.
        - By default, the free parameters are:
            - `q`, `u`  : source Stokes parameters (Q/I, U/I).
            - `qb`, `ub`: background Stokes parameters (Q/I, U/I).
        - Designed for likelihood evaluation in cases where the 
          background is itself polarized.
        '''
        super(PolarLikelihood_PolarBG, self).__init__(parameters={'q': None, 'u': None, 'qb': None, 'ub': None})
        self.ci = ci
        self.si = si
        self.cj = cj 
        self.sj = sj
        self.mu = mu
        self.mu_bg = mu_bg
        self.area_ratio = area_ratio
        self.count_bg = count_bg

    def log_likelihood(self):
        '''
        Compute the log-likelihood for the polarimetric model with 
        polarized background.
    
        Returns
        -------
        float
            Log-likelihood value evaluated at the current parameter 
            values (`q`, `u`, `qb`, `ub`).
    
        Notes
        -----
        - Calls `polar_likelihood_polarbg` with the stored source and 
          background Stokes parameters, event terms, and instrument 
          configuration.
        - Returns the negative of the likelihood value so that higher 
          values correspond to better fits, consistent with statistical 
          frameworks (e.g., nested sampling, MCMC).
        - Free parameters include both source polarization (`q`, `u`) 
          and background polarization (`qb`, `ub`).
        '''
        likelihood = polar_likelihood_bg.polar_likelihood_polarbg(self.parameters['q'], self.parameters['u'],
                                        self.parameters['qb'], self.parameters['ub'],
                                         self.ci, self.si, self.cj, self.sj, self.mu, self.mu_bg,
                                         self.area_ratio, self.count_bg)
        return -likelihood 

class PolarLikelihoodRotate(bilby.core.likelihood.Likelihood):
    def __init__(self, ci, si, mu, t, tref, area_ratio, count_bg):
        '''
        Initialize a likelihood object for modeling polarization with 
        a rotating EVPA (polarization angle).

        Parameters
       ----------
        ci : array_like
            Cosine terms of photon azimuthal angles (cos(2ψ)).
        si : array_like
            Sine terms of photon azimuthal angles (sin(2ψ)).
        mu : float or array_like
            Instrument modulation factor(s).
        t : array_like
            Photon arrival times.
       tref : float
            Reference time about which the EVPA rotation is defined.
        area_ratio : float
            Ratio of source to background extraction areas.
        count_bg : int or float
            Number of background counts.

        Notes
        -----
        - Inherits from `PolarLikelihoodRotate`.
        - Defines free parameters:
            - `q`, `u` : source Stokes parameters (Q/I, U/I).
            - `r`     : EVPA rotation rate in degrees per unit time.
        - Designed for likelihood evaluation in time-dependent EVPA 
          rotation models (e.g., detecting smooth angle swings).
        '''
        super(PolarLikelihoodRotate, self).__init__(parameters={'q': None, 'u': None, 'r': None})
        self.ci = ci
        self.si = si
        self.mu = mu
        self.t = t 
        self.tref = tref
        self.area_ratio = area_ratio
        self.count_bg = count_bg

    def log_likelihood(self):
        '''
    Compute the log-likelihood for a rotating EVPA (polarization angle) model.
    
        Returns
       -------
        float
            Log-likelihood value evaluated at the current parameter values
            (`q`, `u`, `rotrate`).
    
        Notes
        -----
        - Retrieves parameters from `self.parameters`:
            - `q`, `u`     : source Stokes parameters (Q/I, U/I).
            - `rotrate`    : EVPA rotation rate in degrees per day, 
                         converted internally to degrees per second.
        - Calls `polar_likelihood_rotate` with the current parameters,
          photon arrival times, and instrument configuration.
        - Returns the negative likelihood so that higher values indicate
          better fits, consistent with statistical samplers 
          (e.g., MCMC, nested sampling).
        '''
        q = self.parameters['q']
        u = self.parameters['u']
        r = self.parameters['rotrate']
        
        likelihood = polar_likelihood_rotate.polar_likelihood_rotate([q, u, r/86400], self.ci, self.si, self.mu, self.t, self.tref, self.area_ratio, self.count_bg)
        
        return -likelihood

class PolarLikelihoodRotate_PolarBG(bilby.core.likelihood.Likelihood):
    def __init__(self, ci, si, cj, sj, mu, mu_bg, t, tref, area_ratio, count_bg):
        '''
        Initialize a likelihood object for modeling polarization with 
        a rotating EVPA (polarization angle) and explicitly polarized 
        background.

        Parameters
        ----------
        ci : array_like
            Cosine terms of photon azimuthal angles (source events).
        si : array_like
            Sine terms of photon azimuthal angles (source events).
        cj : array_like
            Cosine terms of photon azimuthal angles (background events).
        sj : array_like
            Sine terms of photon azimuthal angles (background events).
        mu : float or array_like
            Modulation factor(s) for source events.
        mu_bg : float or array_like
            Modulation factor(s) for background events.
        t : array_like
            Photon arrival times.
        tref : float
            Reference time about which EVPA rotation is defined.
        area_ratio : float
            Ratio of source to background extraction areas.
        count_bg : int or float
            Number of background counts.

        Notes
        -----
        - Inherits from `bilby.core.likelihood.Likelihood`.
        - Defines free parameters:
            - `q`, `u`   : source Stokes parameters (Q/I, U/I).
            - `qb`, `ub` : background Stokes parameters (Q/I, U/I).
            - `r`        : EVPA rotation rate in degrees per unit time.
        - Designed for likelihood evaluation in scenarios where both 
          time-dependent EVPA rotation and background polarization 
          must be modeled simultaneously.
        '''
        super(PolarLikelihoodRotate_PolarBG, self).__init__(parameters={'q': None, 'u': None, 'qb': None, 'ub': None, 'r': None})
        self.ci = ci
        self.si = si
        self.cj = cj 
        self.sj = sj 
        self.mu = mu
        self.mu_bg = mu_bg
        self.t = t 
        self.tref = tref
        self.area_ratio = area_ratio
        self.count_bg = count_bg

    def log_likelihood(self):
        '''
        Compute the log-likelihood for a rotating EVPA polarization model 
        with explicitly polarized background.

        Returns
        -------
        float
            Log-likelihood value evaluated at the current parameter values
            (`q`, `u`, `qb`, `ub`, `rotrate`).

        Notes
        -----
        - Retrieves parameters from `self.parameters`:
            - `q`, `u`   : source Stokes parameters (Q/I, U/I).
            - `qb`, `ub` : background Stokes parameters (Q/I, U/I).
            - `rotrate`  : EVPA rotation rate in degrees per day, converted 
                       internally to degrees per second.
        - Calls `polar_likelihood_rotate_polarbg` with the current 
          parameters, photon event information, and instrument configuration.
        - Returns the negative likelihood so that higher values correspond 
          to better fits, consistent with statistical samplers 
          (e.g., MCMC, nested sampling).
        '''
        q = self.parameters['q']
        u = self.parameters['u']
        qb = self.parameters['qb']
        ub = self.parameters['ub']
        r = self.parameters['rotrate']
        
        likelihood = polar_likelihood_rotate.polar_likelihood_rotate_polarbg([q, u, qb, ub, r/86400], self.ci, self.si, self.cj, self.sj, self.mu, self.mu_bg, self.t, self.tref, self.area_ratio, self.count_bg)
        
        return -likelihood


class PolarLikelihoodPVar(bilby.core.likelihood.Likelihood):
    def __init__(self, ci, si, mu, t, tref, area_ratio, count_bg):
        '''
        Initialize a likelihood object for modeling polarization with a 
        linearly time-varying polarization degree (PD).

        Parameters
        ----------
        ci : array_like
            Cosine terms of photon azimuthal angles (cos(2ψ)), assumed constant.
        si : array_like
            Sine terms of photon azimuthal angles (sin(2ψ)), assumed constant.
        mu : float or array_like
            Instrument modulation factor(s).
        t : array_like
            Photon arrival times.
        tref : float
            Reference time about which PD variation is defined.
        area_ratio : float
            Ratio of source to background extraction areas.
        count_bg : int or float
            Number of background counts.

        Notes
        -----
        - Inherits from `PolarLikelihoodPVar`.
        - Defines free parameters:
           - `q`, `u`      : reference Stokes parameters (Q/I, U/I).
            - `p_rate`      : linear rate of change of polarization degree.
        - Intended for scenarios where the polarization degree evolves 
          smoothly with time while EVPA remains fixed.
        '''
        super(PolarLikelihoodPVar, self).__init__(parameters={'q': None, 'u': None, 'p_rate': None})
        self.ci = ci
        self.si = si
        self.mu = mu
        self.t = t
        self.tref = tref
        self.area_ratio = area_ratio
        self.count_bg = count_bg

    def log_likelihood(self):
        '''
         Compute the log-likelihood for a model with linearly time-varying 
        polarization degree (PD).

        Returns
        -------
        float
            Log-likelihood value evaluated at the current parameter values
            (`q`, `u`, `p_rate`).

        Notes
        -----
        - Retrieves parameters from `self.parameters`:
            - `q`, `u`      : reference Stokes parameters (Q/I, U/I).
            - `p_rate`      : linear PD variation rate in units of 
                          degrees per day, converted internally 
                          to per-second.
        - Calls `polar_likelihood_pvar` with the current parameters, 
          photon arrival times, and instrument configuration.
       - Returns the negative likelihood so that higher values correspond 
          to better fits, consistent with statistical samplers 
          (e.g., MCMC, nested sampling).
        '''
        q = self.parameters['q']
        u = self.parameters['u']
        p_rate = self.parameters['p_rate']

        likelihood = polar_likelihood_rotate.polar_likelihood_pvar(
            [q, u, p_rate / 86400],  # Convert rate to per-day units
            self.ci, self.si, self.mu, self.t, self.tref, self.area_ratio, self.count_bg
        )

        return -likelihood
