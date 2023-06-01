import numpy as np
from scipy.optimize import curve_fit
from utils.CustomObjects import DottableDefaultDict
import uncertainties as unc
from uncertainties import unumpy as unp


def GenLogFunc(x, M, D, Q, B, nu):
    """
    Generalized logistic function
    https://en.wikipedia.org/wiki/Generalised_logistic_function
    Assumptions:
        K = 0
        A = 1
    Interpetations: #! These interpretations are not correct!
        A: left asymptote
        K: right asymptote
        C: maximum value
        M: midpoint
        D: scale
        Q: steepness
        B: asymmetry
        nu: shape
    """
    V = 1 - 1 / (1 + Q * np.exp(-B * (np.asarray(x) / D - M))) ** (1 / abs(nu))
    return V[:, 0]

def GenLogFunc_inflection(M, D, Q, B, nu):
    """ Where the double-derivative wrt x of the above is 0. Found this by putting into Wolfram Alpha """
    R = D * (M * B + unp.log(Q / nu)) / B
    return R

def central_uncertainty_squish(x):
    """ The uncertainty near the ends of the [0, 1] domain is low, while in between it is large. Pass values through a gaussian to achieve this when fitting. """
    return np.exp(-((x - 0.5) ** 2 / 0.4))

def generalized_logistic(x, y, verbose=False):
    """
    Fit a generalized logistic function to the data
    """

    #* Initial guess
    # p0 = {"M": -0.25, "D": 0.18, "Q": 1.35, "B": 30, "nu": 0.000001}
    p0 = {"M": -0.26, "D": 0.24, "Q": 1.38, "B": 30, "nu": 7e-5}
    try:
        popt, pconv = curve_fit(GenLogFunc, x, y, p0=list(p0.values()), bounds=([-np.inf,]*(len(p0)-1) + [0,], np.inf))
        # popt, pconv = curve_fit(GenLogFunc, x, y, p0=list(p0.values()), method="dogbox")
    except RuntimeError:
        print("Error - curve_fit failed")
        return None
    
    dpopt = {k: v for k, v in zip(p0.keys(), popt)}

    fit = GenLogFunc(x, *popt)  #* the fit evaluated at the x values
    upopt = unc.correlated_values(popt, pconv)
    R = GenLogFunc_inflection(*upopt)
    
    MSE = np.mean((fit - y) ** 2)
    R2 = 1 - MSE / np.var(y)

    result = DottableDefaultDict()
    result.eval = fit
    result.crit_eps = unc.nominal_value(R)
    result.err = unc.std_dev(R)
    result.R = R
    result.popt = dpopt
    # result.pconv = pconv
    result.MSE = MSE
    result.R2 = R2

    return result
