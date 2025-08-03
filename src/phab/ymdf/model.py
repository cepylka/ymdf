import numpy
import pandas
from scipy import special
from astropy import units, constants
from scipy import stats

fluxline = None
baseline=None
response_curve = {"TESS" : "TESS.txt",
                  "Kepler" : "kepler_lowres.txt"}

for key, val in response_curve.items():
    df = pandas.read_csv(f"/uio/hypatia/geofag-felles/ceed/exoplanets/Elena/COS-HST//data/{val}",
                     delimiter="\s+", skiprows=8)
    df = df.sort_values(by="nm", ascending=True)
    rwav = (df.nm * 10).values * units.angstrom  # convert to angstroms
    rres = (df.response).values
    response_curve[key] = (rwav,rres)

def black_body_spectrum(wav, t):
    """Takes an array of wavelengths and
    a temperature and produces an array
    of fluxes from a thermal spectrum
    Parameters:
    -----------
    wav : Astropy array
        wavenlength array
    t : float
        effective temperature in Kelvin
    """
    t = t * units.K # set unit to Kelvin

    return (( (2 * numpy.pi * constants.h * constants.c**2) / (wav**5) / (numpy.exp( (constants.h * constants.c) / (wav * constants.k_B * t) ) - 1))
            .to("erg*s**(-1)*cm**(-3)"))

def calculate_flare_flux_at_mission(wave, flux, mission):
    """ Get specific Kepler/TESS flux
    Parameters:
    -----------
    mission : string
        TESS or Kepler
    flaret : float
        black body temperature
    Return:
    -------
    specific Kepler/TESS flux in units erg*cm**(-2)*s**(-1)
    """

    try:
        # Read in response curve:
        rwav, rres = response_curve[mission]
        rwav = rwav.value

    except KeyError:
        raise KeyError("Mission can be either Kepler or TESS.")

    w = wave
    thermf = flux

    # Interpolate response from rwav to w:
    rress = numpy.interp(w,rwav,rres)#, left=0, right=1)

    # Integrating the flux of the thermal
    # spectrum times the response curve over wavelength:
    return numpy.trapz(thermf * rress, x=w)

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

def flareModelMendoza(t,tpeak,fwhm, ampl):
    """
    The Continuous Flare Model evaluated for single-peak (classical) flare
    events. Use this function for fitting classical flares
    with most curve_fit tools.

    References:

    - Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6
    - Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    - Jackman et al. (2018) https://arxiv.org/abs/1804.03377

    Parameters:

    - `t`: the time array to evaluate the flare over;
    - `tpeak`: the center time of the flare peak;
    - `fwhm`: the Full Width at Half Maximum, timescale of the flare;
    - `ampl`: The amplitude of the flare.

    Returns:

    - `flare`: the flux of the flare model evaluated at each time. A continuous
    flare template whose shape is defined by the convolution of a Gaussian
    and double exponential and can be parameterized by three parameters:
    center time (tpeak), FWHM, and ampitude.
    """
    t_new = (t - tpeak) / fwhm

    flare = flareEqn(t_new, tpeak, fwhm, ampl)
    return flare

def flareModelDavenport(t,tpeak,fwhm, ampl):
    """
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference: Davenport et al. (2014) http://arxiv.org/abs/1411.3723

    Use this function for fitting classical flares with most curve_fit
    tools.

    Note, this model assumes the flux before the flare is zero centered.

    Parameters:

    - `t`: the time array to evaluate the flare over;
    - `tpeak`: the time of the flare peak;
    - `dur`: the duration of the flare;
    - `ampl`: the amplitude of the flare;

    Returns:

    - `flare`: the flux of the flare model evaluated at each time.
    """

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

def flareEqnFenstein(t, tpeak, fwhm, ampl, rises="Student"):
    """
    The simplified Gaussian rise–exponential decay temporal flare model (Feinstein et al. 2022).
    The modifications: you can choose the Student’s t-distribution with a non-integer fractional
    degrees of freedom. This modification accounts for sharper peaking and better captures the
    observed flare rise morphology. In the exponential decay, we suggest alpha = −0.09,
    which corresponds to a slower decay.

    Use this function for fitting FUV flares.

    Parameters:

    - `t`: the time array to evaluate the flare over;
    - `tpeak`: the time of the flare peak;
    - `dur`: the duration of the flare;
    - `ampl`: the amplitude of the flare;

    Returns:

    - `flare`: the flux of the flare model evaluated at each time.
    """

    def studentTRise(t, tpeak, fwhm, ampl, df=2.05):
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

    def exponentialDecay(t, tpeak, ampl, decay_rate=0.09):
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
    decay_model  = exponentialDecay(t, tpeak, ampl, decay_rate=0.09)

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

def fit_to_flare(time, a, c, tpeak,fwhm, out2_integ, out1_integ, flux_a_integ, fluxline, baseline=None, model="Mendoza"):

    if (baseline is None
        and
        fluxline is not None
        ):
        baseline = fluxline
    else:
        raise ValueError('Both baseline and fluxline are not set. Provide baseline!')

    # denom = 3.300127774365737e+17
    # convolition = numpy.zeros((len(t)), dtype=float)
    # print("corfficients",a, b, c)

    # convolition = ((out1_integ -flux_a_integ) * a + (out2_integ-flux_a_integ) * c)/denom
    # convolition = ((out1_integ -flux_a_integ) * a + (out2_integ-flux_a_integ) * c)/denom
    # convolition = ((out1_integ -flux_a_integ) * a + (out2_integ-flux_a_integ) * c) + flux_a_integ
    convolition = (((out1_integ -flux_a_integ) * a + (out2_integ-flux_a_integ) * c)+flux_a_integ)/ flux_a_integ

    # print("t",t, tpeak,fwhm, convolition)
    ampl = 1.
    # convolition = ((out1_integ -flux_a_integ) * a + (out2_integ-flux_a_integ) * c)/denom

    # # print("t",t, tpeak,fwhm, convolition)
    # ampl = convolition/baseline - 1.
    #
    if model=="Mendoza":
        flare = flareModelMendoza(time, tpeak,fwhm, ampl)
    elif model=="Davenport":
        flare = flareModelDavenport(time, tpeak,fwhm, ampl)
    elif model=="Feinsten":
        flare = flareModelFeinstenModified(time, tpeak,fwhm, ampl)

    return flare*convolition/baseline + baseline

def calculateEDfrommodelLoyd(time, a, c, tpeak,fwhm, modelLoyd, model="Mendoza", baseline=None):
    fin_convolition = (modelAtmo["out1"].values- modelAtmo["flux_zero_model"].values) * a + (modelAtmo["out2"].values- modelAtmo["flux_zero_model"].values) * c
    fin_convolition_zero = modelAtmo["flux_zero_model"]
    flux_a_integ = numpy.trapz(fin_convolition_zero, modelAtmo.index)
    flux_f_integ = numpy.trapz(fin_convolition, modelAtmo.index)
    ampl = 1.

    if model=="Mendoza":
        flare = flareModelMendoza(time, tpeak,fwhm, ampl)
    elif model=="Davenport":
        flare = flareModelDavenport(time, tpeak,fwhm, ampl)
    elif model=="Feinsten":
        flare = flareModelFeinstenModified(time, tpeak,fwhm, ampl)
    ED = numpy.trapz((flare*flux_f_integ + flux_a_integ)/flux_a_integ - 1., time)
    return ED

def calculateEDfromTESS(time, a, c, tpeak,fwhm, modelAtmo, model=None, ampl_coeff=1., baseline=None):
    if model == None:
        model = "Mendoza"
    fin_convolition = (modelAtmo["out1"].values- modelAtmo["flux_zero_model"].values) * a + (modelAtmo["out2"].values- modelAtmo["flux_zero_model"].values) * c
    fin_convolition_zero = modelAtmo["flux_zero_model"]
    flux_a_integ = numpy.trapz(fin_convolition_zero, modelAtmo.index)
    flux_f_integ = numpy.trapz(fin_convolition, modelAtmo.index)
    ampl = 1. * ampl_coeff

    if model=="Mendoza":
        # ampl = 1.7
        flare = flareModelMendoza(time, tpeak,fwhm, ampl)
        # print("Mendoza", flare)
    elif model=="Davenport":
        flare = flareModelDavenport(time, tpeak,fwhm, ampl)
        # print("Davenport", flare)
    elif model=="Feinsten":
        flare = flareModelFeinstenModified(time, tpeak,fwhm, ampl)
        # print("Feinsten", flare)

    ED = numpy.trapz((flare*flux_f_integ + flux_a_integ)/flux_a_integ - 1., time)
    return ED
    # models1 = modelAtmo[modelAtmo.index <= 10000.]
    # models1 = models1[models1.index >= 6000.]
    # out2_integ = calculate_flare_flux_at_mission(models1.index,models1["out2"], mission="TESS")
    # out1_integ= calculate_flare_flux_at_mission(models1.index,models1["out1"], mission="TESS")
    # flux_a_integ = calculate_flare_flux_at_mission(models1.index,models1["flux_zero_model"], mission="TESS")
    # fluxline = flux_a_integ
    # final_flux = fit_to_flare(time, a, c, tpeak,fwhm, out2_integ, out1_integ, flux_a_integ, fluxline)
    # # ed_model = numpy.trapz((final_flux/fluxline - 1.),time)
    # residual = final_flux/fluxline - 1.
    # # ed_model = numpy.trapz((final_flux/fluxline - 1.),time)
    # ed_model =numpy.sum(numpy.diff(time) * residual[:-1])
    # return ed_model

def calculateEDFUVtoTESS(time, a, c, tpeak,fwhm, modelAtmo, model=None, ampl_coeff=1., baseline=None):
    # models1 = modelAtmo[modelAtmo.index <= 1340.]
    # models1 = models1[models1.index >= 1060.]
    models1 = modelAtmo[modelAtmo.index <= 10000.]
    models1 = models1[models1.index >= 6000.]
    # out2_integ = numpy.trapz(models1["out2"],models1["wave"])
    # out1_integ = numpy.trapz(models1["out1"],models1["wave"])
    # flux_a_integ = numpy.trapz(models1["flux_zero_model"],models1["wave"])
    # fluxline = flux_a_integ
    # final_flux = fit_to_flare(time, a, c, tpeak,fwhm, out2_integ, out1_integ, flux_a_integ, fluxline)
    # residual = final_flux/fluxline - 1.
    # # ed_model = numpy.trapz((final_flux/fluxline - 1.),time)
    # ed_model =numpy.sum(numpy.diff(time) * residual[:-1])
    # return ed_model
    out2_integ = calculate_flare_flux_at_mission(models1.index,models1["out2"], mission="TESS")
    out1_integ= calculate_flare_flux_at_mission(models1.index,models1["out1"], mission="TESS")
    flux_a_integ = calculate_flare_flux_at_mission(models1.index,models1["flux_zero_model"], mission="TESS")
    if model == None:
        model = "Mendoza"
    fin_convolition = (models1["out1"].values- models1["flux_zero_model"].values) * a + (models1["out2"].values- models1["flux_zero_model"].values) * c
    fin_convolition_zero = models1["flux_zero_model"]
    flux_a_integ = numpy.trapz(fin_convolition_zero, models1.index)
    flux_f_integ = numpy.trapz(fin_convolition, models1.index)
    ampl = 1. * ampl_coeff

    if model=="Mendoza":
        # ampl = 1.7
        flare = flareModelMendoza(time, tpeak,fwhm, ampl)
        # print("Mendoza", flare)
    elif model=="Davenport":
        flare = flareModelDavenport(time, tpeak,fwhm, ampl)
        # print("Davenport", flare)
    elif model=="Feinsten":
        flare = flareModelFeinstenModified(time, tpeak,fwhm, ampl)
        # print("Feinsten", flare)

    ED = numpy.trapz((flare*flux_f_integ + flux_a_integ)/flux_a_integ - 1., time)
    return ED

def model_M(ED, tstart, interval=20, time=None, model=None, seed=None, duration=None, tpeak_i=None):
    """
    Generate a temporal flare model flux evolution based on a specified flare model.

    This function simulates the time-dependent flux of a flare event using parametric models
    such as "Mendoza", "Davenport", or "Feinsten". The flare duration is either given or
    sampled probabilistically as a function of the equivalent duration (ED). The flare peak time
    is selected within the temporal domain, and the flare shape is characterized by a full width at half maximum (FWHM).

    Parameters:
        ED (float): Equivalent Duration;

        tstart (float): Start time of the flare event;

        interval (float, optional): Time step interval for flare model evaluation. Default is 20;

        time (numpy.ndarray or None, optional): Array of time points for the flare model evaluation.
        If None, time is managed internally. Default is None;

        seed (int or None, optional): Random seed for reproducibility. Default is None;

        duration (float or None, optional): Total duration of the flare in seconds.
        If None, duration is sampled based on ED. Default is None;

        tpeak_i (int or None, optional): Index of the flare peak time within the time array. If None, selected randomly
        within the first half of the time range. Default is None.

    Returns:
        pandas.DataFrame: DataFrame indexed by time with a single column "flux" representing the normalized flare flux evolution over time.
    """
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
                duration = numpy.random.uniform(10, 600) # to account for Zhang, L., et al.: A&A, 689, A103 (2024) 2-branch effect in Duration/Energy relation Fig. 5.
    if time is None:
        time = numpy.arange(tstart, tstart + duration + interval, step=interval)
        print("time is none")#, time)
    else:
        time = time
        print("time is not none")#, time)
    if tpeak_i == len(time): # safety net for gap boundaries
        tpeak_i= tpeak_i - 1
    elif tpeak_i == 0 :
        tpeak_i= tpeak_i + 1
    if tpeak_i is None:
        # h=numpy.random.randint(1, int(len(time)/2), size=1)
        # tpeak = time[h]
        upper = int(len(time) / 2)
        if upper > 1:
            h = numpy.random.randint(1, upper, size=1)
        else:
            # Handle the case where the time array is too short
            # For example:
            h = 1
        tpeak = time[h]
    else:
        tpeak = time[tpeak_i]

    fwhm=(time[-1]-time[0])/2.0
    print("fwhm", fwhm)

    if model == None:
        model = "Mendoza"
    # ampl = ED / (1.827 * fwhm) # /24./60./60.
    # print("ampl", ampl)
    if model=="Mendoza":
        ampl=1.0#5249
        flareModel = flareModelMendoza(time, tpeak,fwhm, ampl)
    elif model=="Davenport":
        ampl=1.0
        flareModel = flareModelDavenport(time, tpeak,fwhm, ampl)
    elif model=="Feinsten":
        ampl=1.0
        flareModel = flareModelFeinstenModified(time, tpeak,fwhm, ampl)
    # print("flareModel",flareModel)
    flare = pandas.DataFrame(flareModel,columns=["flux"], index=time)
    return flare



def calculateCoefficients(ED, tstart, interval, c, F_quiescent, F1, F2, time=None, model=None, seed=None, duration = None, tpeak_i=None):
    """
    Calculate the scaling coefficient for combining two stellar atmospheric models.

    This function computes a normalization coefficient to scale two flare spectral components (F1 and F2 models) weighted by
    a coefficient ratio `c`, so that their combined flux matches the equivalent duration (ED) of the flare event.

    Parameters:
        ED (float): Equivalent Duration representing the total energy input of the flare;

        tstart (float): Start time of the flare observation or simulation in sec.;

        interval (float): Time step interval in sec.;

        c (float): Ratio of coefficients scaling the F1 to F2 flare model components;

        F_quiescent (float): Quiescent flux from the stellar atmosphere models;

        F1 (float): Integrated flux of the F1 model component over the spectral range;

        F2 (float): Integrated flux of the F2 model component over the spectral range;

        time (numpy.ndarray or None, optional): Array of time points for the flare model evaluation. If None, time is managed internally. Default is None.

        model (str or None, optional): Temporal flare model to use (e.g., "Mendoza"). If None, defaults to "Mendoza";

        seed (int or None, optional): Random seed for reproducibility of any stochastic elements in the model. Default is None;

        duration (float or None, optional): Total duration of the flare event for the temporal model;

        tpeak_i (float or None, optional): Time of flare peak within the model time base.

    Returns:
        tuple:
            - coeff (float): Scaling coefficient to apply to the flare spectral models, ensuring the combined flux matches the flare energy.
            - flareModel (pandas.DataFrame): DataFrame containing the temporal flare model flux evolution indexed by time.

    """


    # factor = c * (F1 - F_quiescent) + F2 # - F_quiescent+F_quiescent - simplification of the formula
    print("c",c, "F1",F1,"F_quiescent",F_quiescent,"F2",F2,)
    factor =  (c * (F1 - F_quiescent) + (F2 - F_quiescent)) / F_quiescent

    if model == None:
        model = "Mendoza"
    flareModel = model_M(ED, tstart, interval, time=time, model=model, seed=seed, duration = duration,  tpeak_i=tpeak_i)
    # print("flareModel with coeff", flareModel)
    # integrand = ((factor * flareModel["flux"]) / F_quiescent -1.)
    # C = numpy.trapz(integrand, flareModel.index)
    S = numpy.trapz(flareModel["flux"], flareModel.index.values)  # Trapezoidal integration
    print("S", S)
    coeff = ED / (factor * S)
    print("ED", ED, "factor", factor,"coeff", coeff)
    # Solve for the coefficient for Model I = coeff * c * F1(t), then the coefficient for Model II will be coeff * F2(t)
    # coeff = ED / C

    return coeff, flareModel

def variableSpectraOneFlare(c, ED, tstart, modelAtmo, time=None, waverange=[None, None], interval=20, model=None, seed=None, duration = None, tpeak_i=None):

    """
    Simulates the time-variable spectral signal of a single flare.

    This function calculates a composite flare spectrum as a linear combination of two components (F1 and F2 models),
    weighted by a coefficient ratio `c`. The flare evolution is modelled over time using provided coefficients
    derived from input parameters and an atmospheric spectral model (`modelAtmo`) on the spectral range specified by `wave`.
    The final output is a pandas DataFrame representing the time-dependent flare flux at each wavelength that can be added to light curve.

    Parameters:
        c (float): Ratio of coefficients between the F1 and F2 flare atmospheric models. The F1 model is scaled by `c * x`,
            and the F2 model by `x`, where `x` is the base coefficient;

        ED (float): Equivalent Duration parameter specifying the flare energy input for coefficient calculation;

        tstart (float): Start time of the flare observation in sec;

        modelAtmo (pandas.DataFrame): Stellar atmosphere spectral model containing columns "out1" (F1 model),
        "out2" (F2 model),  and "flux_zero_model" (quiescent flux). Index corresponds to wavelength or spectral coordinate;

        time (numpy.ndarray or None, optional): Array of time points (timestamps) representing the flare event when fitting actual flare observations.
        For simulated data generation, this should be set to `None`.

        waverange (list or tuple of two floats or None, optional):   Specifies the wavelength interval [lower_bound, upper_bound] to be used for spectral integration and flux calculations.
        If set to [None, None] (default), the full wavelength range covered by the atmospheric model (`modelAtmo`) is used without restriction.
        Otherwise, only wavelengths within the specified bounds are considered.

        interval (int, optional): Time interval (in sec.) for flare modelling steps. Default is 20;

        model (str, optional): Flare temporal model to use for coefficient calculation. Default is "Mendoza";

        seed (int or None, optional): Random seed for reproducibility in coefficient calculations. Default is None;

        duration (float or None, optional): Total duration of the flare event. If None, inferred from flare model. Default is None;

        tpeak_i (float or None, optional): Time of flare peak within model time base. Default is None;

        denom (float, optional): Normalization denominator applied to the flux. Default is 1.0.

        wave (str, optional): Must be one of "photo", "uv", or "custom". Determines
        wavelength range used in flux integration. Default is "photo".

    Returns:
        pandas.DataFrame: Time-indexed DataFrame where each row represents a time step of the flare evolution and each column corresponds
        to a wavelength from `modelAtmo`.

    """
    if waverange == [None, None]:
        F1 = numpy.trapz(modelAtmo["out1"],modelAtmo["wave"])
        F2 = numpy.trapz(modelAtmo["out2"],modelAtmo["wave"])
        F_quiescent = numpy.trapz(modelAtmo["flux_zero_model"],modelAtmo["wave"])
    elif waverange == [600., 1000.]:
        print("wave:", waverange[0], waverange[1], modelAtmo.index.min(), modelAtmo.index.max())
        modelst = modelAtmo[modelAtmo.index <= waverange[1]]
        modelst = modelst[modelst.index >= waverange[0]]
        print("variableSpectraOneFlare seed", seed)
        F1 = calculate_flare_flux_at_mission(modelst.index, modelst["out1"], mission="TESS")
        F2 = calculate_flare_flux_at_mission(modelst.index, modelst["out2"], mission="TESS")
        F_quiescent = calculate_flare_flux_at_mission(modelst.index, modelst["flux_zero_model"], mission="TESS")
    else:
        print("wave:", waverange[0], waverange[1], modelAtmo.index.max(), modelAtmo.index.max())
        modelst = modelAtmo[modelAtmo.index <= waverange[1]]
        modelst = modelst[modelst.index >= waverange[0]]
        print("variableSpectraOneFlare seed", seed)
        F1 = numpy.trapz(modelst["out1"],modelst["wave"])
        F2 = numpy.trapz(modelst["out2"],modelst["wave"])
        F_quiescent = numpy.trapz(modelst["flux_zero_model"],modelst["wave"])
    if model == None:
        model = "Mendoza"
    print("inside the flux prod", F_quiescent, F1, F2)
    coeff, flareModel = calculateCoefficients(ED, tstart, interval, c, F_quiescent, F1, F2, time=time, model=model, seed=seed, duration=duration, tpeak_i=tpeak_i)
    print("first coeff",coeff*c, "second coeff",coeff)#, "flareModel",flareModel)
    time = flareModel.index
    fin_convolition = (modelAtmo["out1"].values- modelAtmo["flux_zero_model"].values) * coeff *c + (modelAtmo["out2"].values - modelAtmo["flux_zero_model"].values) * coeff# + modelAtmo["flux_zero_model"]
    oneFlare = pandas.DataFrame(columns=modelAtmo.index, index=time)
    for ix, r in flareModel.iterrows():
        oneFlare.loc[ix] = fin_convolition*flareModel.at[ix, "flux"] + modelAtmo["flux_zero_model"].values
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

def variableSpectraOneFlareObs(c, ED, tstart, modelAtmo, interval=20, model="Mendoza", seed=None, duration = None,  tpeak_i=None, denom=1., wave="photo"):

    """
    Simulates the time-variable spectral signal of a single flare in the certain wavelength interval.

    This function calculates a composite flare spectrum as a linear combination of two components (F1 and F2 models),
    weighted by a coefficient ratio `c`. The flare evolution is modelled over time using provided coefficients
    derived from input parameters and an atmospheric spectral model (`modelAtmo`) on the spectral range specified by `wave`.
    The final output is a pandas DataFrame representing the time-dependent flare flux at each wavelength that can be added to light curve.

    Parameters:
        c (float): Ratio of coefficients between the F1 and F2 flare atmospheric models. The F1 model is scaled by `c * x`,
            and the F2 model by `x`, where `x` is the base coefficient;

        ED (float): Equivalent Duration parameter specifying the flare energy input for coefficient calculation;

        tstart (float): Start time of the flare observation in sec;

        modelAtmo (pandas.DataFrame): Stellar atmosphere spectral model containing columns "out1" (F1 model),
        "out2" (F2 model),  and "flux_zero_model" (quiescent flux). Index corresponds to wavelength or spectral coordinate;

        interval (int, optional): Time interval (in sec.) for flare modelling steps. Default is 20;

        model (str, optional): Flare temporal model to use for coefficient calculation. Default is "Mendoza";

        seed (int or None, optional): Random seed for reproducibility in coefficient calculations. Default is None;

        duration (float or None, optional): Total duration of the flare event. If None, inferred from flare model. Default is None;

        tpeak_i (float or None, optional): Time of flare peak within model time base. Default is None;

        denom (float, optional): Normalization denominator applied to the flux. Default is 1.0.

        wave (str, optional): Must be one of "photo", "uv", or "custom". Determines
        wavelength range used in flux integration. Default is "photo".

    Returns:
        pandas.DataFrame: Time-indexed DataFrame where each row represents a time step of the flare evolution and each column corresponds
        to a wavelength from `modelAtmo`.

    """
    print("variableSpectraOneFlare seed", seed)
    if wave=="photo":
        #for TESS range
        modelAtmo_range = modelAtmo[modelAtmo.index <= 10000.]
        modelAtmo_range = modelAtmo_range[modelAtmo_range.index >= 6000.]
    elif wave=="uv":
        modelAtmo_range = modelAtmo[modelAtmo.index <= 1355.49]
        modelAtmo_range = modelAtmo_range[modelAtmo_range.index >= 1058.15]
    F1 = numpy.trapz(modelAtmo_range["out1"],modelAtmo_range["wave"])
    F2 = numpy.trapz(modelAtmo_range["out2"],modelAtmo_range["wave"])
    F_quiescent = numpy.trapz(modelAtmo_range["flux_zero_model"],modelAtmo_range["wave"])

    coeff, flareModel = calculateCoefficients(ED, tstart, interval, c, F_quiescent, F1, F2, model="Mendoza", seed=seed, duration=duration, tpeak_i=tpeak_i)
    print("first coeff",coeff*c, "second coeff",coeff, )
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
    """
    Power-law generator for pdf(x)\propto x^{g-1}
    for a<=x<= and single power law
    The formalism of sampling:

    x_min - lower bound, alpha - the scaling parameter of the distribution.
    The PDF (Probability density function p(x) of a continuous power-law distribution) can be given as:
    p(x) = ((alpha - 1) / x_min) * (x / x_min) ** -alpha
    The cumulative density function, CDF can be given as:
    P(x) = (x / x_min) ** (-alpha + 1)
    The inverse of CDF:
    Pinv(x) = x_min * x ** (-1 / (alpha - 1))
    The distribution is obtained from the inverse CDF using x = numpy.random.uniform(0, 1, n_samples)


    Parameters:
        x_min (float): Lower bound of the sampling range.
        x_max (float): Upper bound of the sampling range.
        alpha_prior (float): Scaling exponent alpha (> 1) of the power-law distribution.
        size (int, optional): Number of samples to generate. Default is 1.
        seed (int or None, optional): Seed for random number generator to ensure reproducibility.

    Returns:
        numpy.ndarray: Array of sampled values drawn from the specified power-law distribution.
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

    Parameters:
        x0: lower bound
        x1: break point
        x2: upper bound
        a1: lower power law index
        a2: upper power low index
    Returns:
        the weights w1, w2 of each piece of the function and the total integral
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

    Parameters:
        u: an array of uniform random numbers between on {0,1},  size: n_samples
        x0: lower bound
        x1: break point
        x2: upper bound
        a1: lower power law index
        a2: upper power low index
    Returns:
        numpy.ndarray: Array of sampled values drawn from the specified broken power-law distribution.
    """
    if seed is not None:
        numpy.random.seed(seed)
    # create an array of random values between (0,1) in size of n_samples
    u = numpy.random.uniform(0,1, n_samples)

    a1=-a1
    a2=-a2
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
    # print(out)
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


def generateSimulatedData(flaresperday, Tprime, interval, beta_prior,
                       alpha1_prior, alpha2_prior=None, xbreak=None,  seed=None,
                       EDmin=None, EDmax=None,
                       T0=None, law="piecewise"):
    """
    Generate a fake flare distribution from given law with coefficients:
    : broken (alpha1, alpha2, beta) or single (alpha1, alpha2=None, beta)
    alpha and beta. Also produces the list of flaring times
    using Tprime, interval and flare rate.

    Parameters:
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
        pandas DataFrane : tstart of flares with corresponding ED
    """
    # Flares per time given prior on alpha1 and beta
    flaresperday = flaresperday#(beta_prior *
                         # numpy.power(EDmin, alpha1_prior - 1.) /
                         # (alpha1_prior - 1.))
    # observing time stamps
    obstimes = numpy.arange(T0, T0 + Tprime + interval, interval)
    num_intervals = len(obstimes) #// interval
    # print("num_intervals", num_intervals)
    # observing time stamps with flares from Poisson distribution
    flare_rate_per_interval = flaresperday / (24 * 60 * 60 / interval)
    poisson_values = numpy.random.poisson(lam=flare_rate_per_interval, size=num_intervals)
    flare_indices = numpy.where(poisson_values > 0)[0]
    tstarts = obstimes[flare_indices]
    # print("tstarts",tstarts)
    #  # number of events
    # print("flaresperday:", flaresperday)
    # print("interval (s):", interval)
    # print("flare_rate_per_interval:", flare_rate_per_interval)
    # print("num_intervals:", num_intervals)
    # print("poisson_values:", poisson_values)
    if len(tstarts) == len(obstimes):
        raise ValueError("Every time stamp is a flare.")

    # Energy related stuff:
    # u = numpy.random.uniform(0,1, len(tstarts))
    # Finally, Generate power law distributed data:
    if law=="piecewise":
        events = sampleBrokenPowerLaw(EDmin, xbreak, EDmax, alpha1_prior, alpha2_prior, len(tstarts), seed=seed)
    elif law=="single":
        events = sampleSinglePowerLaw(EDmin, EDmax, alpha1_prior, len(tstarts), seed=seed)

    # print("events",events)
    flaring_events = pandas.DataFrame(
                {"tstart":tstarts,
                "ED":events,
                }
                )


    return flaring_events

