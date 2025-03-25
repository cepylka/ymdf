import numpy
from .flare_finder import flareModelMendoza2022, flareModelDavenport2014, flareEqn

def flareEqnFenstein(t, tpeak, fwhm, ampl, rises="Gaussian"):
    # growth = numpy.where(t <= tpeak)[0]
    # decay  = numpy.where(t >  tpeak)[0]
    def studentTRise(t, tpeak, fwhm, ampl, df=2.1):
    # Convert FWHM to scale parameter for t-distribution
        scale = fwhm / (2 * np.sqrt(2 * (df/(df-2)) * (2**(2/df) - 1)))

        flux1 = np.zeros_like(t)
        rise = t <= tpeak
        x = (t[rise] - tpeak) / scale
        flux1[rise] = ampl * (1 + x**2/df)**(-0.5*(df+1))
        return flux1[rise]
    def gaussianRise(t, tpeak, fwhm, ampl):
        # Convert FWHM to sigma
        sigma = fwhm / (2 * numpy.sqrt(2 * numpy.log(2)))/2

        # Calculate the flux
        flux1 = numpy.zeros_like(t)

        # Gaussian rise (t <= tpeak)
        rise = t <= tpeak
        flux1[rise] = ampl * numpy.exp(-(t[rise] - tpeak)**2 / (2 * sigma**2))

        # # Constant peak value (t > tpeak)
        # flux[~rise] = ampl

        return flux1[rise]

    def exponentialDecay(t, tpeak, ampl, decay_rate=0.009):
        # Initialize flux array
        flux2 = np.zeros_like(t)

        # Exponential decay (t >= tpeak)
        decay = t > tpeak
        flux2[decay] = ampl * np.exp(-decay_rate * (t[decay] - tpeak))

        # # Constant value before peak (t < tpeak)
        # flux[~decay] = ampl
        return flux2[decay]
    if rises=="Student":
        growth_model = studentTRise(t, tpeak, fwhm, ampl)
    elif rises=="Gaussian":
        growth_model = gaussianRise(t, tpeak, fwhm, ampl)
    decay_model  = exponentialDecay(t, tpeak, ampl, decay_rate=0.005)

    flare = numpy.append(growth_model, decay_model)
    return flare

def flareModelFeinstenModified(t,tpeak,fwhm, ampl):
    """
    Generates a flare model with a Gaussian rise and an
    exponential decay. Feinstein et al. 2020 with the options to use Student distribution instead if Gaussian

    Parameters
    ----------
    time : numpy.ndarray
         A time array.
    amp : float
         The amplitude of the flare.
    t0 : int
         The index in the time array where the flare will occur.
    rise : float
         The Gaussian rise of the flare.
    fall : float
         The exponential decay of the flare.
    y : numpy.ndarray, optional
         Underlying stellar activity. Default if None.

    Returns
    ----------
    flare_model : numpy.ndarray
         A light curve of zeros with an injected flare of given parameters

    """

    flare = flareEqnFenstein(t, tpeak, fwhm, ampl)
    return flare

def betaFromEps(eps, alpha, deltaT, mined):
    """Obtain the FFD beta parameters from
    the occurrence probability epsilon of a flare
    above mined within time period deltaT.

    Parameters:
    ------------
    eps : float
        flaring probability within time period deltaT
        and above energy mined
    alpha : float
        power law exponent
    deltaT : float
        time period for eps
    mined : float
        minimum energy for eps

    Return:
    --------
    float - FFD beta parameter
    """
    # See Wheatland 2004: flaring rate vs. flaring probability eps
    rate = -numpy.log(1 - eps) / deltaT

    # Use cumulative flare frequency formula
    return rate * (alpha - 1.) * numpy.power(mined, alpha - 1.)


def epsFromBeta(beta, alpha, deltaT, mined):
    """
    Obtain epsilon from beta. Reverse of `beta_from_epsilon`.

    Parameters:

    - `beta`: FFD beta parameter;
    - `alpha`: power law exponent;
    - `deltaT`: time period for eps;
    - `mined`: minimum energy for eps.

    Returns:

    - flaring probability within time period deltaT and above energy mined.
    """
    exponent = beta * numpy.power(mined, 1. - alpha) * deltaT / (alpha - 1.)

    return 1. - numpy.exp(-exponent)

def generateSinglePowerLawDistribution(x_min, x_max, alpha_prior, size=1, seed=None):
    """Power-law generator for pdf(x)\propto x^{g-1}
    for a<=x<= and single power law
    The formalism of sampling:

    x_min - lower bound, alpha - the scaling parameter of the distribution.
    The PDF (Probability density function p(x) of a continuous power-law distribution) can be given as:
    p(x) = ((alpha - 1) / x_min) * (x / x_min) ** -alpha
    The cumulative density function, CDF can be given as:
    P(x) = (x / x_min) ** (-alpha + 1)
    The inverse of CDF:
    Pinv(x) = x_min * x ** (-1 / (alpha - 1))
    We got the distribution from the inverse CDF using x = numpy.random.uniform(0, 1, n_samples)

    """
    g=-alpha_prior + 1.
    if seed is not None:
        numpy.random.seed(seed)
    r = numpy.random.random(size=size)
    ag, bg = x_min**g, x_max**g
    return (ag + (bg - ag) * r)**(1. / g)


def integratePowerLaw(x0, x1, x2, a1, a2):
    """
    This implementation of Broken Power Law sampling was based
    on https://github.com/grburgess/brokenpl_sample/blob/master/sample_broken_power_law.ipynb

    x0: lower bound
    x1: break point
    x2: upper bound
    a1: lower power law index
    a2: upper power low index

    """

    # compute the integral of each piece analytically
    int_1 = (numpy.power(x1, a1+1.) - numpy.power(x0, a1+1.))/(a1+1)
    int_2 = numpy.power(x1,a1-a2)*(numpy.power(x2, a2+1.) - numpy.power(x1, a2+1.))/(a2+1)

    # compute the total integral
    total = int_1 + int_2

    # compute the weights of each piece of the function
    w1 = int_1/total
    w2 = int_2/total

    return w1, w2, total

def sampleBrokenPowerLaw(u, x0, x1, x2, a1, a2):
    """
    This implementation of Broken Power Law sampling was based
    on https://github.com/grburgess/brokenpl_sample/blob/master/sample_broken_power_law.ipynb

    u: uniform random number between on {0,1}
    x0: lower bound
    x1: break point
    x2: upper bound
    a1: lower power law index
    a2: upper power low index
    """

    # compute the weights with our integral function
    w1, w2, _ = integratePowerLaw(x0, x1, x2, a1, a2)

    # create a holder array for our output
    out = numpy.empty_like(u)

    # compute the bernoulli trials for lower piece of the function
    # *if we wanted to do the upper part... we just reverse our index*
    # We also compute these to bools for numpy array selection
    idx = stats.bernoulli.rvs(w1, size=len(u)).astype(bool)

    # inverse transform sample the lower part for the "successes"
    out[idx] = numpy.power(
        u[idx] * (numpy.power(x1, a1 + 1.0) - numpy.power(x0, a1 + 1.0))
        + numpy.power(x0, a1 + 1.0),
        1.0 / (1 + a1),
    )

    # inverse transform sample the upper part for the "failures"
    out[~idx] = numpy.power(
        u[~idx] * (numpy.power(x2, a2 + 1.0) - numpy.power(x1, a2 + 1.0))
        + numpy.power(x1, a2 + 1.0),
        1.0 / (1 + a2),
    )

    return out

def brokenPowerLaw(x,x0, x1, x2, a1, a2):
    """
    This implementation of Broken Power Law sampling was based
    on https://github.com/grburgess/brokenpl_sample/blob/master/sample_broken_power_law.ipynb

    x: the domain of the function lin or logspace usually \in[x_min,x_max]
    x0: lower bound
    x1: break point
    x2: upper bound
    a1: lower power law index, should be provided in form: a1=-a1 where a\in[1.01,2.99]
    a2: upper power low index, should be provided in form: a2=-a2 where a\in[1.01,2.99]

    """

    # creatre a holder for the values
    out = numpy.empty_like(x)

    # get the total integral to compute the normalization
    _,_,C = integrate_pl(x0, x1, x2, a1, a2)
    norm = 1./C

    # create an index to select each piece of the function
    idx = x<x1

    # compute the lower power law
    out[idx] = numpy.power(x[idx],a1)

    # compute the upper power law
    out[~idx] = numpy.power(x[~idx],a2) * numpy.power(x1,a1-a2)

    return out* norm


def generateFakeData(Tprime, cadence, beta_prior,
                       alpha_prior, mined=None,
                       deltaT=None, seed=None,
                       Emin=1., Emax=1e7,
                       T0=3000):
    """Generate a fake flare distribution from given
    alpha and beta. Also produces the list of flaring times
    using Tprime and cadence, which is not yet used.

    Parameters:
    -----------
    Tprime : int
        total observation time,
        must be int for now
    cadence : int
        observations per time
    beta_prior : float
        value for the power law intercept
        to start MCMC chain with.
        NOT a prior in the Bayesian sense.
    alpha_prior : float
        value for the power law exponent
        alpha to start MCMC chain with,
        NOT a prior in the Bayesian sense.
    mined : float
        minimum energy we want to predict a rate for
        (same as S2 in Wheatland 2004 paper). Default will
        be set to 10x the maximum generated energy.
    deltaT : float
        The time period corresponding to mined. Defualt will
        be set to Tprime
    seed : int
        Default None. If set, the generated events
        will be fixed.
    Emin : float
        minimum energy of events
    Emax : float
        maximum energy of events
    T0 : int
        start time of observations

    Return:
    -------
    dict : dictionary of kwargs (pass to
        BayesianFlaringAnalysis)
    """
    # Flares per time given prior on alpha and beta
    flarespertime = (beta_prior *
                     numpy.power(Emin, alpha_prior - 1.) /
                     (alpha_prior - 1.))

    # observing time stamps
    obstimes = numpy.linspace(T0, T0 + Tprime, Tprime * cadence)

    # observing time stamps with flares from Poisson distribution
    times = obstimes[numpy.where(numpy.random.poisson(lam=flarespertime / cadence,
                                                size=Tprime * cadence))[0]]

    Mprime = len(times) # number of events

    if Mprime == Tprime * cadence:
        raise ValueError("Every time stamp is a flare. "
                         "Choose a lower beta_prior, or a higher "
                         "alpha_prior.")

    # Energy related stuff:

    # Finally, Generate power law distributed data:
    events = generateSinglePowerLawDistribution(Emin, Emax,
                                                    -alpha_prior + 1.,
                                                    size=Mprime, seed=seed)

    # Energy and time period for predition:
    if mined is None:
        mined = numpy.max(events) * 10.
    if deltaT is None:
        deltaT = Tprime

    # Time related stuff:

    # starting occurrence probability for MCMC
    eps_prior = epsFromBeta(beta_prior, alpha_prior, deltaT, mined)


    return {"events":events, "mined":mined, "deltaT":deltaT,
            "alpha_prior": alpha_prior, "eps_prior": eps_prior,
            "threshed": Emin, "Mprime" : Mprime, "Tprime":Tprime,
            "beta_prior": beta_prior}

