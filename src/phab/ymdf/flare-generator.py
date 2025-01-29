import numpy


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


def generateRandomPowerLawDistribution(
    a,
    b,
    alpha1,
    alpha2,
    beta_ini,
    x_break,
    size=1,
    seed=None
):
    """
    Power-law generator for pdf(x) propto x^{g-1} for a<=x<=b.

    Parameters:

    - `a`: min ED of the flare;
    - `b`: max ED of the flare;
    - `alpha1`: alpha 1 coefficient;
    - `alpha2`: alpha 2 coefficient;
    - `beta_ini`: initial estimate beta;
    - `x_break`: x break value in ED dimensions.
    """

    if seed is not None:
        numpy.random.seed(seed)

    # calculate the fraction of samples before the break
    c1 = a**(1-alpha1) - xbreak**(1-alpha1)
    c2 = xbreak**(1-alpha2)
    frac = c1 / (c1 + c2 * (1-alpha1) / (1-alpha2))

    r = numpy.random.random(size=size)
    samples = numpy.zeros(size)
    mask = r < frac
    # Samples before the break
    samples[mask] = (xmin**(1 - alpha1) - u[mask] * c1)**(1 / (1 - alpha1))

    # Samples after the break
    samples[~mask] = (
        (
            xbreak**(1 - alpha2)
            -
            (u[~mask] - frac)
            *
            (1 - alpha2)
            *
            c2
            /
            (1 - frac)
        )**(1 / alpha2)
    )

    ag1, bg1 = alpha1**a, b**g
    ag2, bg2 = alpha2**a, b**g
    (ag + (bg - ag) * r)**(1. / g)

    return samples


def generate_random_broken_power_law_distribution(a, b, g, size=1, seed=None):
    """Power-law generator for pdf(x)\propto x^{g-1}
    for a<=x<=b
    """
    if seed is not None:
        numpy.random.seed(seed)
    r = numpy.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag) * r)**(1. / g)


def generate_fake_data(Tprime, cadence, beta_prior,
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
    events = generateRandomPowerLawDistribution(Emin, Emax,
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

