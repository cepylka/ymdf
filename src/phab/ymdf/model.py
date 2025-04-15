import numpy
import pandas
from scipy import special

def flareEqn(t, tpeak, fwhm, ampl):
    """
    The equation that defines the shape for the Continuous Flare Model
    """

    # values were fit and calculated using MCMC 256 walkers and 30000 steps
    A, B, C, D1, D2, f1 = [
        0.9687734504375167,
        -0.251299705922117,
        0.22675974948468916,
        0.15551880775110513,
        1.2150539528490194,
        0.12695865022878844
    ]

    # we include the corresponding errors for each parameter
    # from the MCMC analysis
    A_err, B_err, C_err, D1_err, D2_err, f1_err = [
        0.007941622683556804,
        0.0004073709715788909,
        0.0006863488251125649,
        0.0013498012884345656,
        0.00453458098656645,
        0.001053149344530907
    ]

    f2 = 1 - f1

    eqn = (
        (
            (1 / 2)
            *
            numpy.sqrt(numpy.pi)
            *
            A
            *
            C
            *
            f1
            *
            numpy.exp(-D1 * t + ((B / C) + (D1 * C / 2)) ** 2)
            *
            special.erfc(((B - t) / C) + (C * D1 / 2))
        )
        +
        (
            (1 / 2)
            *
            numpy.sqrt(numpy.pi)
            *
            A
            *
            C
            *
            f2
            *
            numpy.exp(-D2 * t+ ((B / C) + (D2 * C / 2)) ** 2)
            *
            special.erfc(((B - t) / C) + (C * D2 / 2))
        )
    )

    return eqn * ampl

def flareModelMendosa(t,tpeak,fwhm, ampl):
    t_new = (t - tpeak) / fwhm

    flare = flareEqn(t_new, tpeak, fwhm, ampl)
    return flare

def flareModelDavenport(t,tpeak,fwhm, ampl):
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]
    flare = numpy.piecewise(
            t,
            [(t <= tpeak) * (t-tpeak)/fwhm > -1.0, (t > tpeak)],
            [
                lambda x: (
                    _fr[0]  # 0th order
                    +
                    _fr[1] * ((x - tpeak) / fwhm)  # 1st order
                    +
                    _fr[2] * ((x - tpeak) / fwhm)**2.0  # 2nd order
                    +
                    _fr[3] * ((x - tpeak) / fwhm)**3.0  # 3rd order
                    +
                    _fr[4] * ((x - tpeak) / fwhm)**4.0  # 4th order
                ),
                lambda x: (
                    _fd[0] * numpy.exp(((x - tpeak) / fwhm) * _fd[1])
                    +
                    _fd[2] * numpy.exp(((x - tpeak) / fwhm) * _fd[3])
                )
            ]
        ) * numpy.abs(ampl)
    return flare

def flareEqnFenstein(t, tpeak, fwhm, ampl, rises="Gaussian"):

    def studentTRise(t, tpeak, fwhm, ampl, df=2.1):
    # Convert FWHM to scale parameter for t-distribution
        scale = fwhm / (2 * numpy.sqrt(2 * (df/(df-2)) * (2**(2/df) - 1)))

        flux1 = numpy.zeros_like(t)
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


        return flux1[rise]

    def exponentialDecay(t, tpeak, ampl, decay_rate=0.009):
        # Initialize flux array
        flux2 = numpy.zeros_like(t)

        # Exponential decay (t >= tpeak)
        decay = t > tpeak
        flux2[decay] = ampl * numpy.exp(-decay_rate * (t[decay] - tpeak))

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


def model_M(ED, tstart, interval=20, model="Mendosa", seed=None, duration=None):
    print("model_M seed", seed)
    if seed is not None:
        numpy.random.seed(seed)
    else:
        numpy.random.seed()  # do not remove, otherwise the seed is fixed
    if duration is None:
        # duration = numpy.random.uniform(1e-2, 1.0, 1) * ED * 1.0e3 # in sec, min duration 10 sec * ED
        duration = numpy.random.uniform(3e-2, 3.0, 1) * (ED ** 0.3) * 60 * 10 # Proportional to ED^0.3, in seconds, producing duration in [10.21,3280]seconds

        # Check if the initial duration is less than 10 minutes (600 seconds)
        if duration > 600.:
            # Randomly decide whether to choose another duration from an interval
            if numpy.random.rand() < 0.5:  # Controlled chance for small or large durations
                # Randomly pick a small value (e.g., between 10 and 600 seconds)
                duration = numpy.random.uniform(10, 600) # to account for Zhang, L., et al.: A&A, 689, A103 (2024) 2-branch effect in Duration/Energy relation
    time = numpy.arange(tstart, tstart + duration + interval, step=interval)
    print(time)
    h=numpy.random.randint(1, int(len(time)/2), size=1)
    tpeak = time[h]
    fwhm=(time[-1]-time[0])/2.0

    if model=="Mendosa":
        ampl=1.05249
        flareModel = flareModelMendosa(time, tpeak,fwhm, ampl)
    elif model=="Davenport":
        ampl=1.0
        flareModel = flareModelDavenport(time, tpeak,fwhm, ampl)
    elif model=="Feinsten":
        ampl=1.0
        flareModel = flareModelFeinstenModified(time, tpeak,fwhm, ampl)
    flare = pandas.DataFrame(flareModel,columns=["flux"], index=time)
    return flare



def calculateCoefficients(ED, tstart, interval, c, F_quiescent, F1, F2, model="Mendosa", seed=None, duration = None):
    # factor = c * (F1 - F_quiescent) + F2 # - F_quiescent+F_quiescent - simplification of the formula
    factor =  (c * (F1 - F_quiescent) + (F2 - F_quiescent)) / F_quiescent
    flareModel = model_M(ED, tstart, interval, model="Mendosa", seed=seed, duration = duration)
    # integrand = ((factor * flareModel["flux"]) / F_quiescent -1.)
    # C = numpy.trapz(integrand, flareModel.index)
    S = numpy.trapz(flareModel["flux"], flareModel.index.values)  # Trapezoidal integration

    coeff = ED / (factor * S)

    # Solve for the coefficient for Model I = coeff * c * F1(t), then the coefficient for Model II will be coeff * F2(t)
    # coeff = ED / C

    return coeff, flareModel

def variableSpectraOneFlare(c, ED, tstart, modelAtmo, interval=20, model="Mendosa", seed=None, duration = None):

    """
    c is ratio of the coefficients for F1 and F2 models
    x  - coefficient (c*x for F1 model, x for F2 model)
    models is pandas DataFrame with columns: "out1" for F1, "out2" for F2, "flux_zero_model" for F_quiescent


    """
    print("variableSpectraOneFlare seed", seed)
    F1 = numpy.trapz(modelAtmo["out1"],modelAtmo["wave"])
    F2 = numpy.trapz(modelAtmo["out2"],modelAtmo["wave"])
    F_quiescent = numpy.trapz(modelAtmo["flux_zero_model"],modelAtmo["wave"])

    coeff, flareModel = calculateCoefficients(ED, tstart, interval, c, F_quiescent, F1, F2, model="Mendosa", seed=seed, duration=duration)
    print("coeff",coeff, "flareModel", flareModel)
    time = flareModel.index
    fin_convolition = (modelAtmo["out1"].values- modelAtmo["flux_zero_model"].values) * coeff *c + (modelAtmo["out2"].values - modelAtmo["flux_zero_model"].values) * coeff + modelAtmo["flux_zero_model"]
    oneFlare = pandas.DataFrame(columns=modelAtmo.index, index=time)
    for ix, r in flareModel.iterrows():
        oneFlare.loc[ix] = fin_convolition*flareModel.at[ix, "flux"] #+ modelAtmo["flux_zero_model"].values
    a = time[0]-interval
    b = time[-1]+interval
    # row_from_column = pandas.DataFrame(modelAtmo["flux_zero_model"]).T

    # Optionally, set the index or column names for clarity
    # row_from_column.columns = []
    # row_from_column.index = ["new_row"]
    first_row = modelAtmo[["flux_zero_model"]].T
    first_row.rename(index={"flux_zero_model": float(a)}, inplace=True)
    last_row = modelAtmo[["flux_zero_model"]].T
    last_row.rename(index={"flux_zero_model": float(b)}, inplace=True)
    oneFlare = pandas.concat([first_row, oneFlare, last_row], ignore_index=False)

    return oneFlare

def variableSpectraOneFlareObs(c, ED, tstart, modelAtmo, interval=20, model="Mendosa", seed=None, duration = None, denom=1.):

    """
    c is ratio of the coefficients for F1 and F2 models
    x  - coefficient (c*x for F1 model, x for F2 model)
    models is pandas DataFrame with columns: "out1" for F1, "out2" for F2, "flux_zero_model" for F_quiescent


    """
    print("variableSpectraOneFlare seed", seed)
    F1 = numpy.trapz(modelAtmo["out1"],modelAtmo["wave"])
    F2 = numpy.trapz(modelAtmo["out2"],modelAtmo["wave"])
    F_quiescent = numpy.trapz(modelAtmo["flux_zero_model"],modelAtmo["wave"])

    coeff, flareModel = calculateCoefficients(ED, tstart, interval, c, F_quiescent, F1, F2, model="Mendosa", seed=seed, duration=duration)
    print("coeff",coeff)
    time = flareModel.index
    fin_convolition = (modelAtmo["out1"].values- modelAtmo["flux_zero_model"].values) * coeff *c + (modelAtmo["out2"].values - modelAtmo["flux_zero_model"].values) * coeff #+ modelAtmo["flux_zero_model"]
    oneFlare = pandas.DataFrame(columns=modelAtmo.index, index=time)
    print("fin_convolition/denom", max(fin_convolition/denom))
    for ix, r in flareModel.iterrows():
        oneFlare.loc[ix] = fin_convolition/denom*flareModel.at[ix, "flux"]+(modelAtmo["flux_zero_model"]/denom)# + modelAtmo["flux_zero_model"].values
    a = time[0]-interval
    b = time[-1]+interval
    # row_from_column = pandas.DataFrame(modelAtmo["flux_zero_model"]).T

    # Optionally, set the index or column names for clarity
    # row_from_column.columns = []
    # row_from_column.index = ["new_row"]
    first_row = modelAtmo[["flux_zero_model"]].T
    first_row.rename(index={"flux_zero_model": float(a)}, inplace=True)
    last_row = modelAtmo[["flux_zero_model"]].T
    last_row.rename(index={"flux_zero_model": float(b)}, inplace=True)
    oneFlare = pandas.concat([first_row, oneFlare, last_row], ignore_index=False)

    return oneFlare

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

def sampleSinglePowerLaw(x_min, x_max, alpha_prior, size=1, seed=None):
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
    else:
        numpy.random.seed()  # do not remove, otherwise the seed is fixed
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

def sampleBrokenPowerLaw(x0, x1, x2, a1, a2, n_samples, seed=None):
    """
    This implementation of Broken Power Law sampling was based
    on https://github.com/grburgess/brokenpl_sample/blob/master/sample_broken_power_law.ipynb

    u: an array of uniform random numbers between on {0,1},  size: n_samples
    x0: lower bound
    x1: break point
    x2: upper bound
    a1: lower power law index
    a2: upper power low index
    """
    if seed is not None:
        numpy.random.seed(seed)
    # create an array of random values between (0,1) in size of n_samples
    u = numpy.random.uniform(0,1, n_samples)

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


def generateSimulatedData(Tprime, interval, beta_prior,
                       alpha1_prior, alpha2_prior=None, xbreak=None,  seed=None,
                       EDmin=1., EDmax=1e7,
                       T0=0.0, law="piecewise"):
    """Generate a fake flare distribution from given law with coefficients:
    : broken (alpha1, alpha2, beta) or single (alpha1, alpha2=None, beta)
    alpha and beta. Also produces the list of flaring times
    using Tprime, interval and flare rate.

    Parameters:
    -----------
    Tprime : float
        total observation time in sec
    interval :
        interval for observations (20 sec to imitate TESS fast cadence)
    beta_prior : float
        value for the power law intercept
    alpha1_prior : float
        value for the power law exponent 1
    alpha2_prior : float
        value for the power law exponent 2, optional
    xbreak : float
        equivalent duration in sec at break for broken power law
    seed : int
        Default None. If set, the generated events
        will be fixed.
    EDmin : float
        minimum equivalent duration of events
    EDmax : float
        maximum equivalent duration of events
    T0 : usually 0.0
        start time of observations
    law : "piecewise" or "single" for broken power law with alpha1_prior
        and alpha2_prior and single power law with alpha1_prior, respectively

    Return:
    -------
    pandas DataFrane : tstart of flares with corresponding ED
    """
    # Flares per time given prior on alpha1 and beta
    flaresperday = (beta_prior *
                         numpy.power(Emin, alpha1_prior - 1.) /
                         (alpha1_prior - 1.))
    # observing time stamps
    obstimes = numpy.arange(T0, T0 + Tprime + interval, interval)
    num_intervals = int(obs_time // interval)
    # observing time stamps with flares from Poisson distribution
    flare_rate_per_interval = flaresperday / (24 * 60 * 60 / interval)
    poisson_values = numpy.random.poisson(lam=flare_rate_per_interval, size=num_intervals)
    flare_indices = numpy.where(poisson_values > 0)[0]
    tstarts = obstimes[flare_indices]

     # number of events

    if len(tstarts) == len(obstimes):
        raise ValueError("Every time stamp is a flare.")

    # Energy related stuff:

    # Finally, Generate power law distributed data:
    if law=="piecewise":
        events = sampleBrokenPowerLaw(EDmin, xbreak, EDmax, alpha1_prior, alpha2_prior, size=len(tstarts), seed=seed)
    elif law=="single":
        events = sampleSinglePowerLaw(EDmin, EDmax, alpha1_prior, size=len(tstarts), seed=seed)


    flaring_events = pandas.DataFrame(
                {"tstart":tstarts,
                "ED":events,
                }
                )


    return flaring_events

