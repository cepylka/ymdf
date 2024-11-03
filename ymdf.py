import numpy
import pandas
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic
import progressbar

import pandera
from typing import Tuple, List, Optional, Any


def findGaps(
    lightCurve: pandas.DataFrame,
    maximumGap: float = 30,
    minimumObservationPeriod: int = 10
) -> List[Tuple[int, int]]:
    """
    Find gaps in light curve.

    Parameters:

    - `maximumGap`: should be larger than step in time series
    - `minimumObservationPeriod`:
    """

    dt = numpy.diff(lightCurve["time"].values)

    gap = numpy.where(numpy.append(0, dt) >= maximumGap)[0]

    # add start/end of lightCurve to loop over easily
    gapOut = numpy.append(0, numpy.append(gap, len(lightCurve.index)))

    # start
    left = gapOut[:-1]
    # end of data
    right = gapOut[1:]-1

    # drop too short observation periods
    tooShort = numpy.where(numpy.diff(gapOut) < minimumObservationPeriod)
    left = numpy.delete(left, tooShort)
    right = numpy.delete(right, tooShort)

    return list(zip(left, right))


def medSig(a):
    """
    Returns median and outlier-robust estimate of standard deviation
    (1.48 x median of absolute deviations).

    Adapted from K2SC (Aigrain et al. 2016).
    """
    lst = numpy.isfinite(a)
    nfinite = lst.sum()
    if nfinite == 0:
        return numpy.nan, numpy.nan
    if nfinite == 1:
        return a[lst], numpy.nan
    med = numpy.median(a[lst])
    sig = 1.48 * numpy.median(numpy.abs(a[lst] - med))

    return med, sig


def expandMask(a, longdecay=1):
    """
    Expand the mask if multiple outliers occur in a row. Add
    `sqrt(outliers in a row)` masked points before and after
    the outlier sequence.

    Parameters:

    - `a`: mask;
    - `longdecay`: optional parameter to expand the mask more by
        this factor after the series of outliers.

    Returns:

    - `array`: expanded mask
    """
    i, j, k = 0, 0, 0

    while i < len(a):
        v = a[i]

        if v == 0 and j == 0:
            k += 1
            j = 1
            i += 1
        elif v == 0 and j == 1:
            k += 1
            i += 1
        elif v == 1 and j == 0:
            i += 1
        elif v == 1 and j == 1:
            if k >= 2:
                addto = int(numpy.rint(numpy.sqrt(k)))
                a[i - k - addto:i - k] = 0
                a[i:i + longdecay * addto] = 0
                i += longdecay * addto
            else:
                i += 1
            j = 0
            k = 0

    return a


def sigmaClip(
    a,
    max_iter=10,
    max_sigma=3.,
    separate_masks=False,
    mexc=None
):
    """
    Iterative sigma-clipping routine that separates not finite points
    and down and upwards outliers.

    Adapted from (Aigrain et al. 2016).

    1: good data point
    0: masked outlier

    Parameters:

    - `a`: flux array;
    - `max_iter`: how often do we want to recalculate sigma
    to get ever smaller outliers;
    - `max_sigma` where do we clip the outliers;
    - `separate_masks`: if true, will give to boolean arrays for positive
    and negative outliers;
    - `mexc`: custom mask to additionally account for.

    Returns:

    - boolean array (all) or two boolean arrays (positive/negative)
    with the final outliers as zeros.
    """

    # perform sigma-clipping on finite points only,
    # or custom indices given by mexc
    mexc = numpy.isfinite(a) if mexc is None else numpy.isfinite(a) & mexc
    # initialize different masks for up and downward outliers
    mhigh = numpy.ones_like(mexc)
    mlow = numpy.ones_like(mexc)
    mask = numpy.ones_like(mexc)

    # iteratively (with i) clip outliers above(below) (-)max_sigma *sig
    i, nm = 0, None

    while (nm != mask.sum()) & (i < max_iter):
        # okay values are finite and not outliers
        mask = mexc & mhigh & mlow
        # safety check if the mask looks fine
        nm = mask.sum()
        if nm > 1:
            # calculate median and MAD adjusted standard deviation
            med, sig = medSig(a[mask])
            # indices of okay values above median
            mhigh[mexc] = a[mexc] - med < max_sigma * sig
            # indices of okay values below median
            mlow[mexc] = a[mexc] - med > -max_sigma * sig

            # okay values are finite and not outliers
            mask = mexc & mhigh & mlow

            # expand the mask left and right
            mhigh = expandMask(mhigh)

            i += 1

    if separate_masks:
        return mlow, mhigh
    else:
        return mlow & mhigh


def detrendSavGolUltraViolet(
    lightCurve: pandas.DataFrame,
    gaps: List[Tuple[int, int]],
    windowLength: int
) -> pandas.DataFrame:
    """
    Construct a light curve model. Based on original Appaloosa (Davenport 2016)
    with Savitzky-Golay filtering from `scipy` and iterative `sigmaClip`ping
    adopted from K2SC (Aigrain et al. 2016).

    Parameters:

    - `lightCurve`: light curve;
    - `gaps`: found gaps in series;
    - `windowLength`: number of datapoints for Savitzky-Golay filter, either
    one value for entire light curve of piecewise for gaps.
    """

    maximumWindowLength = 5
    if windowLength > maximumWindowLength:
        raise ValueError(
            f"Windows length cannot be larger than {maximumWindowLength}"
        )

    if gaps is None:
        raise ValueError(
            " ".join((
                "Gaps cannot be None, so if your series has no gaps,",
                "then pass an empty list"
            ))
        )

    lightCurve["fluxDetrended"] = numpy.array(
        [numpy.nan]*len(lightCurve.index), dtype=float
    )
    lightCurve["fluxModel"] = numpy.array(
        [numpy.nan]*len(lightCurve.index), dtype=float
    )

    for (le, ri) in gaps:
        # iterative sigma clipping
        correctValues = numpy.where(
            sigmaClip(lightCurve.iloc[le:ri]["flux"])
        )[0] + le
        # incorrect values (inverse of correct ones)
        outliers = list(set(list(range(le, ri))) - set(correctValues))

        betweenGaps = pandas.DataFrame(columns=lightCurve.columns)
        for index, row in lightCurve.iterrows():
            if index in correctValues:
                prwt = pandas.DataFrame([row], index=[index])
                betweenGaps = pandas.concat([betweenGaps, prwt])
            elif index in outliers:
                lightCurve.at[index, "fluxDetrended"] = lightCurve.at[
                    index,
                    "flux"
                ]
                lightCurve.at[index, "fluxModel"] = numpy.nanmean(
                    lightCurve["flux"]
                )

        if not betweenGaps.empty:
            betweenGaps["fluxDetrended"] = savgol_filter(
                betweenGaps["flux"],
                windowLength,
                3,
                mode="nearest"
            )
            betweenGaps["fluxModel"] = (
                betweenGaps["flux"]
                -
                betweenGaps["fluxDetrended"]
                +
                numpy.nanmean(betweenGaps["fluxDetrended"])
            )

        for index, row in lightCurve.iterrows():
            if index in betweenGaps.index:
                lightCurve.at[index, "fluxDetrended"] = betweenGaps.at[
                    index,
                    "fluxDetrended"
                ]
                lightCurve.at[index, "fluxModel"] = betweenGaps.at[
                    index,
                    "fluxModel"
                ]

    # with pandas.option_context("mode.use_inf_as_null", True):
    lightCurve = lightCurve.replace([numpy.inf, -numpy.inf], numpy.nan)
    # lightCurve = lightCurve.dropna()
    return lightCurve


def findIterativeMedian(
    lightCurve: pandas.DataFrame,
    gaps: List[Tuple[int, int]],
    n: int = 30
) -> pandas.DataFrame:
    """
    Find the iterative median value for a continuous observation period
    using flare finding to identify outliers.

    Parameters:

    - `lightCurve`: light curve;
    - `gaps`: found gaps in series;
    - `n`: maximum number of iterations.
    """

    lightCurve["iterativeMedian"] = numpy.array(numpy.NaN, dtype=float)

    for (le, ri) in gaps:
        flux = lightCurve.iloc[le:ri]["fluxDetrended"].values
        # find a median that is not skewed by outliers
        good = sigmaClip(flux)
        goodflux = flux[good]
        for index, row in lightCurve.iterrows():
            if index in range(le, ri + 1):
                lightCurve.at[
                    index,
                    "iterativeMedian"
                ] = numpy.nanmedian(goodflux)

    return lightCurve


def findFlaresInContObsPeriod(
    flux,
    median,
    error,
    sigma=None,
    n1=3,
    n2=2,
    n3=3,
    addtail=False,
    tailthreshdiff=1.0,
    fake=False
) -> list[bool]:
    """
    The algorithm for local changes due to flares defined
    by S. W. Chang et al. (2015), Eqn. 3a-d,
    https://ui.adsabs.harvard.edu/abs/2015ApJ...814...35C/abstract

    These equations were originally in magnitude units, i.e. smaller
    values are increases in brightness. The signs have been changed, but
    coefficients have not been adjusted to change from log(flux) to flux.

    Parameters:

    - `flux`: data to search over;
    - `median`: median value of quiescent stellar flux;
    - `error`: errors corresponding to data;
    - `sigma`: local scatter of the flux. Array should be the same length
    as the detrended flux array. If sigma is None, error is used instead;
    - `n1`: how many times above `sigma` is required;
    - `n2`: how many times above `sigma` and `error` is required
    - `n3`: the number of consecutive points required to flag as a flare;
    - `addtail`: optionally, add data points to the flares
    with a lower threshold;
    - `tailthreshdiff`: relaxes the detection threshold for datapoints
    that are added to the decay tails of flare candidates. The `tailthreshdiff`
    value is subtracted from `n1` and `n2` and should not be larger than
    either of the two.
    """
    isFlare = numpy.zeros_like(flux, dtype="bool")

    # If no local scatter characteristics are given, use formal error as sigma
    if sigma is None:
        sigma = error
    T0 = flux - median  # excursion should be positive # "n0"
    T1 = numpy.abs(flux - median) / sigma  # n1
    T2 = numpy.abs(flux - median - error) / sigma  # n2

    # print(
    #     "\n".join((
    #         f"- factor above standard deviation. n1 = {n1}",
    #         f"- factor above standard deviation + uncertainty. n2 = {n2}",
    #         f"- minimum number of consecutive data points for candidate, n3 = {n3}"
    #     ))
    # )

    # apply thresholds n0-n2:
    pass_thresholds = numpy.where((T0 > 0) & (T1 > n1) & (T2 > n2))

    # array of indices where thresholds are exceeded:
    is_pass_thresholds = numpy.zeros_like(flux)
    is_pass_thresholds[pass_thresholds] = 1

    # need to find cumulative number of points that pass_thresholds,
    # counted in reverse,
    # examples:
    # reverse_counts = [0 0 0 3 2 1 0 0 1 0 4 3 2 1 0 0 0 1 0 2 1 0]
    #        isFlare = [0 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0]

    reverse_counts = numpy.zeros_like(flux, dtype="int")
    for k in range(2, len(flux)):
        reverse_counts[-k] = (
            is_pass_thresholds[-k]
            *
            (
                reverse_counts[-(k-1)]
                +
                is_pass_thresholds[-k]
            )
        )

    # find flare start where values in reverse_counts switch from 0 to >=n3
    istart_i = numpy.where(
        (reverse_counts[1:] >= n3)
        &
        (reverse_counts[:-1] - reverse_counts[1:] < 0)
    )[0] + 1

    # use the value of reverse_counts to determine how many points away stop is
    istop_i = istart_i + (reverse_counts[istart_i])

    # add decay phase data point with a lower detection threshold
    if addtail is True:
        # check for bad values of tailthreshdiff
        if ((tailthreshdiff > n1) | (tailthreshdiff > n2)):
            raise ValueError(
                " ".join((
                    "The threshold on the decay tail should be > 0.",
                    "Note that n1tail = n1-tailthreshdiff and the same for n2."
                ))
            )

        # calculate new n1 and n2 for the tails
        n1tail = n1 - tailthreshdiff
        n2tail = n2 - tailthreshdiff

        # add data points from tail until threshold no longer satisfied
        newstops = []
        for s in istop_i:
            while ((T0[s] > 0) & (T1[s] > n1tail) & (T2[s] > n2tail)):
                s += 1
            newstops.append(s)

        # overwrite old flare stop indices
        istop_i = newstops

    # Create boolean flare mask
    isFlare = numpy.zeros_like(flux, dtype="bool")

    for (l, r) in list(zip(istart_i, istop_i)):
        isFlare[l:r+1] = True

    return isFlare


def chiSquare(residual, error):
    """
    Compute the normalized chi square statistic:
    chisq = 1 / N * SUM(i) ((data(i) - model(i)) / error(i))^2
    """
    return (
        numpy.sum((residual / error)**2.0)
        /
        numpy.size(error)
    )


def equivalentDuration(
    lightCurve: pandas.DataFrame,
    start: int,
    stop: int,
    error=False
) -> float | Tuple[float, float]:
    """
    Returns the equivalent duration of a flare event found within
    indices [start, stop], calculated as the area under
    the residual (flux-flux_median).

    Use only on detrended light curves.

    Returns also the uncertainty on ED following Davenport (2016)

    Parameters:

    - `lightCurve`: flare light curve;
    - `start`: start time index of a flare event;
    - `stop`: end time index of a flare event;
    - `error`: if true, then will compute uncertainty on ED.

    Returns:

    - `ed`: equivalent duration in seconds;
    - `edError`: uncertainty in seconds.
    """

    start = int(start)
    stop = int(stop)+1

    lct = lightCurve.loc[start:stop]
    residual = (
        lct["fluxDetrended"].values
        /
        numpy.nanmedian(lct["iterativeMedian"].values)
        -
        1.0
    )
    x = lct["time"].values  # in seconds, add `* 60.0 * 60.0 * 24.0` for days
    ed = numpy.sum(numpy.diff(x) * residual[:-1])

    if error is True:
        flare_chisq = chiSquare(
            residual[:-1],
            (
                lct["fluxDetrended"].values[:-1]
                /
                numpy.nanmedian(lct["iterativeMedian"].values)
            )
        )
        edError = numpy.sqrt(ed**2 / (stop-1-start) / flare_chisq)
        return ed, edError
    else:
        return ed


def findFlares(
    timeSeries,
    fluxSeries,
    fluxErrorSeries,
    n1: int,
    n2: int,
    n3: int,
    minSep: int = 3,
    sigma: Optional[list[float]] = None,
    doPeriodicityRemoving: bool = False
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Obtaining and processing a light curve.

    Parameters:

    - `timeSeries`: a list of time values;
    - `fluxSeries`: a list of flux values;
    - `fluxErrorSeries`: a list of fluxes errors;
    - `doPeriodicityRemoving`: whether to remove periodicity or not;
    - `minSep`: minimum distance between two candidate start times
    in datapoints;
    - `sigma`: local scatter of the flux. This array should be the same length
    as the detrended flux array.
    """

    seriesLength = len(timeSeries)
    if seriesLength != len(fluxSeries) or seriesLength != len(fluxErrorSeries):
        raise ValueError("The length of all the series must be the same")

    # ---

    lightCurveTableSchema = pandera.DataFrameSchema(
        index=pandera.Index(int),  # also check that it is increasing
        columns={
            "time": pandera.Column(float),
            "flux": pandera.Column(float),
            "fluxError": pandera.Column(float)
        }
    )
    lightCurve = pandas.DataFrame(
        {
            "time": timeSeries,
            "flux": fluxSeries,
            "fluxError": fluxErrorSeries
        }
    )
    #lightCurveTableSchema(lightCurve)

    # ---

    flaresTableSchema = pandera.DataFrameSchema(
        {
            "istart": pandera.Column(int, nullable=True),
            "istop": pandera.Column(int, nullable=True)
        }
    )

    flares = pandas.DataFrame()

    # ---

    gaps = findGaps(lightCurve, 30, 10)

    # ---

    detrendedLightCurve = detrendSavGolUltraViolet(lightCurve, gaps, 5)
    print(detrendedLightCurve)

    if detrendedLightCurve["fluxDetrended"].isna().all():
        raise ValueError("Finding flares only works on detrended light curves")

    # ---

    detrendedLightCurve = findIterativeMedian(detrendedLightCurve, gaps, 30)

    istart = numpy.array([], dtype="int")
    istop = numpy.array([], dtype="int")

    isFlare = None
    # work on periods of continuous observation with no gaps
    for (le, ri) in gaps:
        error = detrendedLightCurve.iloc[le:ri]["fluxError"]
        flux = detrendedLightCurve.iloc[le:ri]["fluxDetrended"]
        median = detrendedLightCurve.iloc[le:ri]["iterativeMedian"]
        time = detrendedLightCurve.iloc[le:ri]["time"]

        # run final flare-find on DATA - MODEL

        isFlare = findFlaresInContObsPeriod(
            flux,
            median,
            error,
            sigma[le:ri] if sigma is not None else None,
            n1=n1,
            n2=n2,
            n3=n3
        )

        # pick out final flare candidate indices
        istart_gap = numpy.array([])
        istop_gap = numpy.array([])
        candidates = numpy.where(isFlare > 0)[0]
        if (len(candidates) > 0):
            # find start and stop index, combine neighboring candidates
            # in to same events
            separated_candidates = numpy.where(
                (numpy.diff(candidates)) > minSep
            )[0]
            istart_gap = candidates[
                numpy.append([0], separated_candidates + 1)
            ]
            istop_gap = candidates[
                numpy.append(
                    separated_candidates,
                    [len(candidates) - 1]
                )
            ]
        istart = numpy.array(
            numpy.append(istart, istart_gap + le),
            dtype="int"
        )
        istop = numpy.array(
            numpy.append(istop, istop_gap + le),
            dtype="int"
        )

    if len(istart) > 0:
        lst = [
            equivalentDuration(detrendedLightCurve, i, j, error=True)
            for (i, j) in zip(istart, istop)
        ]
        ed_rec, ed_rec_err = zip(*lst)

        fl = detrendedLightCurve["fluxDetrended"].values
        ampl_rec = [
            (
                numpy.max(fl[i:j])
                /
                detrendedLightCurve["iterativeMedian"].values[i] - 1.
            )
            for (i, j) in zip(istart, istop)
        ]
        tstart = detrendedLightCurve.iloc[istart]["time"].values
        tstop = detrendedLightCurve.iloc[istop]["time"].values

        newFlare = pandas.DataFrame(
            {
                "ed_rec": ed_rec,
                "ed_rec_err": ed_rec_err,
                "ampl_rec": ampl_rec,
                "istart": istart,
                "istop": istop,
                "tstart": tstart,
                "tstop": tstop,
                "total_n_valid_data_points": detrendedLightCurve["flux"].values.shape[0],
                "dur": tstop - tstart
              }
          )

        flares = pandas.concat([flares, newFlare], ignore_index=True)

    #flaresTableSchema(flares)

    return (detrendedLightCurve, flares)


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


def flareModelMendoza2022(t, tpeak, fwhm, ampl, upsample=False, uptime=10):
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

    t_new = t-tpeak / fwhm

    if upsample:
        dt = numpy.nanmedian(numpy.diff(numpy.abs(t_new)))
        timeup = numpy.linspace(
            min(t_new) - dt,
            max(t_new) + dt,
            t_new.size * uptime
        )

        flareup = flareEqn(timeup, tpeak, fwhm, ampl)

        # and now downsample back to the original time
        downbins = numpy.concatenate(
            (t_new - dt / 2.0, [max(t_new) + dt / 2.0])
        )
        flare = binned_statistic(
            timeup,
            flareup,
            statistic="mean",
            bins=numpy.sort(downbins)
        )[0]
    else:
        flare = flareEqn(t_new, tpeak, fwhm, ampl)

    return flare


def flareModelDavenport2014(t, tpeak, dur, ampl, upsample=False, uptime=10):
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
    - `upsample`: if `True`, then up-sample the model flare to ensure
    more precise energies;
    - `uptime`: how many times to up-sample the data.

    Returns:

    - `flare`: the flux of the flare model evaluated at each time.
    """

    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    fwhm = dur / 2.0  # crude approximation for a triangle shape

    if upsample:
        dt = numpy.nanmedian(numpy.diff(t))
        timeup = numpy.linspace(min(t) - dt, max(t) + dt, t.size * uptime)

        flareup = numpy.piecewise(
            timeup,
            [
                (timeup <= tpeak) * (timeup-tpeak) / fwhm > -1.0,
                (timeup > tpeak)
            ],
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
                    _fd[0]
                    *
                    numpy.exp(((x - tpeak) / fwhm) * _fd[1])
                    +
                    _fd[2]
                    *
                    numpy.exp(((x - tpeak) / fwhm) * _fd[3])
                )
            ]
        ) * numpy.abs(ampl)  # amplitude

        # this uses binned statistic
        downbins = numpy.concatenate((t-dt/2.0, [max(t) + dt/2.0]))
        flare = binned_statistic(
            timeup,
            flareup,
            statistic="mean",
            bins=downbins
        )[0]
    else:
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
        ) * numpy.abs(ampl)  # amplitude

    return flare


def injectFakeFlares(
    lc,
    flares,
    model="mendoza2022",
    gapwindow=0.1,
    fakefreq=0.0005/24/60/60,
    inject_before_detrending=False,
    d=False,
    seed=None
):
    """
    Create a number of events, inject them in to data. Use grid of amplitudes
    and durations, keep ampl in relative flux units. Keep track of energy
    in Equiv Dur. Duration defined in minutes. Amplitude defined multiples
    of the median error.

    Parameters:

    - `model`: the flare model to use (*`mendoza2022` or `davenport2014`*);
    - `gapwindow`: ?;
    - `fakefreq`: flares per second, but at least one per continuous
    observation period will be injected;
    - `inject_before_detrending`: by default, flares are injected before
    the light curve is detrended;
    - `d`: if `True`, a seed for random numbers will be set;
    - `seed`: if `d == True`, seed will be set to this number.

    Returns:

    - `FlareLightCurve` with fake flare signatures.
    """

    fake_flares = pandas.DataFrame(
        columns=[
            "duration_d",
            "amplitude",
            "ed_inj",
            "peak_time"
        ]
    )

    # either inject flares into the un-detrended light curve
    if inject_before_detrending is True:
        typ = "flux"
        typerr = "flux_err"
    # or into the detrended one
    elif inject_before_detrending is False:
        typ = "detrended_flux"
        typerr = "detrended_flux_err"

    # how many flares do you want to inject,
    # at least one per gap or as defined by the frequency
    nfakesum = max(
        len(gaps),
        int(
            numpy.rint(
                fakefreq
                *
                (lc["time"].values.max() - lc["time"].values.min())
            )
        )
    )

    # init arrays for the synthetic flare parameters
    t0_fake = []  # peak times
    ed_fake = []  # ED
    dur_fake = []  # duration
    ampl_fake = []  # amplitude

    # init the synthetic flare counter to allow to point to the right places
    # in the arrays above (XXX_fake, etc)
    # ckm = 0

    gaps = findGaps(lc)
    # are there real flares to deal with in the gap?
    real_flares_in_gap = pandas.DataFrame()
    for (le, ri) in gaps:
        real_gap = flares.query("`istart` >= @le & `istop` <= @ri")
        real_flares_in_gap = pandas.concat([real_flares_in_gap, real_gap])
    # pick flux, time, and flux error arrays
    for k in range(len(gaps)):
        le, ri = gaps[k]
        error = lc.iloc[le:ri][typerr].values
        flux = lc.iloc[le:ri][typ].values
        time = lc.iloc[le:ri]["time"].values
        nfake = 1

        # generate the time constraints for the flares you want to inject
        mintime, maxtime = numpy.min(time), numpy.max(time)
        dtime = maxtime - mintime

        # generate a distribution of durations and amplitudes
        distribution = generateFakeFlareDistribution(
            nfake,
            d,
            seed
        )

        # add the distribution for this observing period
        # to the full list of injected flares
        print(distribution[0], distribution[1])
        dur_fake = distribution[0]
        ampl_fake = distribution[1]

        # generate random peak time, avoid known flares
        isok = False
        # keep picking new random peak times for your synthetic flares
        # until it does not overlap with a real one
        while isok is False:
            # choose a random peak time
            # if you set a seed you will get the same synthetic flares
            # all the time
            if isinstance(seed, int):
                t0 = (
                    modRandom(1, d, seed * k)
                    *
                    dtime
                    +
                    mintime
                )[0]
            # if you do note set a seed, the synthetic flares will be
            # randomly distributed
            elif seed is None:
                t0 = (
                    modRandom(1, d)
                    *
                    dtime
                    +
                    mintime
                )[0]

            # check if there are there any real flares to deal with
            # at that peak time. Only relevant if there were any flares
            # detected at all
            if len(real_flares_in_gap) > 0:
                b = 0
                # are there any real flares happening at peak time?
                # fake flares should not overlap with real ones
                for index, row in real_flares_in_gap.iterrows():
                    if (
                        t0 >= real_flares_in_gap.at[index, "tstart"]
                        and
                        t0 <= real_flares_in_gap.at[index, "tstop"]
                    ):
                        b += 1

                # number of flares that overlap should be 0
                if b == 0:
                    isok = True
            # no real flares, no trouble
            else:
                isok = True

            # add the peak time to the list
            # generate the flare flux from the Davenport 2014 model
            if model == "davenport2014":
                fl_flux = flareModelDavenport2014(
                    time,
                    t0,
                    dur_fake[0],
                    ampl_fake[0]
                )
            elif model == "mendoza2022":
                fl_flux = flareModelMendoza2022(
                    time,
                    t0,
                    dur_fake[0],
                    ampl_fake[0]
                )
            else:
                raise ValueError(f"Unknown flare model: {model}")

            lc.iloc[le:ri][typ] = (
                lc.iloc[le:ri][typ].values
                +
                fl_flux * lc.iloc[le:ri]["iterativeMedian"].values
            )

            # calculate the injected ED
            ed_fake = numpy.sum(numpy.diff(time) * fl_flux[:-1])
            # inject flare in to light curve by adding the flare flux
            fake_flares.at[k, "duration_d"] = dur_fake[0]
            fake_flares.at[k, "amplitude"] = ampl_fake[0]
            fake_flares.at[k, "ed_inj"] = ed_fake
            fake_flares.at[k, "peak_time"] = t0

        # Increment the counter
        # ckm += nfake

    # return the FLC with the injected flares
    return lc, fake_flares


def sampleFlareRecovery(
    lc,
    flares,
    iterations=2000,
    mode=None,
    func=None,
    save_lc_to_file=False,
    folder="",
    fakefreq=0.05/24/60/60,
    path=None
):
    """
    Runs a number of injection recovery cycles and characterizes the light
    curve by recovery probability and equivalent duration underestimation.
    Inject one flare per light curve.

    Parameters:

    - `iterations`: number of injection/recovery cycles;
    - `fakefreq`: number of flares per sec, but at least one per continuous
    observation period will be injected.

    Returns:

    - `lc`: detrended light curve with all fake_flares listed in the attribute;
    - `fake_lc`: light curve with the last iteration of synthetic
    flares injected.
    """
    widgets = [progressbar.Percentage(), progressbar.Bar()]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=iterations).start()
    for i in range(iterations):
        fake_lc, fake_flares = injectFakeFlares(
            lc,
            flares,
            fakefreq=fakefreq
        )

        injs = fake_flares

        fake_flcd = findFlares(fake_lc, fake=True)
        recs = fake_flcd
        print(recs)
        if save_lc_to_file is True:
            fake_lc.to_fits(f"{folder}after.fits")

        injrec_results = pandas.DataFrame(
            columns=[
                "istart",
                "istop",
                "cstart",
                "cstop",
                "tstart",
                "tstop",
                "ed_rec",
                "ed_rec_err",
                "duration_d",
                "amplitude",
                "ed_inj",
                "peak_time",
                "ampl_rec",
                "dur"
            ]
        )

        # Merge injected and recovered flares
        recs['temp'] = 1
        injs['temp'] = 1
        merged = injs.merge(recs,how='outer')
        merged_recovered = merged[(merged["tstart"] < merged["peak_time"]) & (merged["tstop"] > merged["peak_time"])]
        rest = injs[~injs["amplitude"].isin(merged_recovered["amplitude"].values)]
        merged_all = pandas.concat([merged_recovered, rest]).drop('temp',axis=1)
        injrec_results = pandas.concat([injrec_results, merged_all], ignore_index=True)

        bar.update(i + 1)

        # Add to previous runs of sampleFlareRecovery on the same LC or create new table
        if len(fake_flares) > 0:
            fake_flares = pandas.concat([fake_flares,injrec_results], ignore_index=True)
        else:
            fake_flares = injrec_results

    if save is True:
        # finally read in the result
        lc.fake_flares = pandas.read_csv(path)

    # end monitoring
    bar.finish()
    # fake_flares = fake_flares.drop_duplicates()
    return flares, fake_flares


def setupBins(
    injrec,
    flares,
    ampl_bins=None,
    dur_bins=None,
    flares_per_bin=None
):
    """
    Get amplitude and duration bins.
    """
    # did the user give appropriate bins?
    bins = numpy.array(
        [
            bool(ampl_bins is not None),
            bool(dur_bins is not None)
        ]
    )

    # if only one or no bin is given explicitly,
    # make sure flares_per_bin is set
    if ~bins.all() and flares_per_bin is None:
        raise ValueError(
            " ".join((
                "Give either ampl_bins and dur_bins, or either of",
                "the two together with flares_per_bin, or neither of",
                "the two but flares_per_bin"
            ))
        )

    # if only one out of [ampl_bins, dur_bins] is specified,
    # specify the other by fixing the `flares_per_bin`
    if bins.any() and ~bins.all():
        # which one is not defined?
        if ampl_bins is None:
            b = copy.copy(dur_bins)
        elif dur_bins is None:
            b = copy.copy(ampl_bins)

        # if defined bins are given as array, find length
        lngh = (
            b if isinstance(b, float) or isinstance(b, int)
            else len(b)
        )

        # define the other bins accordingly
        if ampl_bins is None:
            ampl_bins = int(numpy.rint(injrec.shape[0] / lngh / flares_per_bin))
        elif dur_bins is None:
            dur_bins = int(numpy.rint(injrec.shape[0] / lngh / flares_per_bin))
    # if no bins are specified, choose bins of equal size
    # with approximately `flares_per_bin` in each bin:
    elif ~bins.any():
        bins = int(numpy.rint(numpy.sqrt(injrec.shape[0] / flares_per_bin)))
        ampl_bins = bins
        dur_bins = bins

    # if no flares are given, substitute with fake flares
    if len(flares) == 0:
        flares = injrec

    # set bins according to data
    if isinstance(ampl_bins, int):
        ampl_bins = numpy.linspace(
            min(
                injrec["ampl_rec"].min(),
                flares["ampl_rec"].min(),
                injrec["amplitude"].min()
            ),
            max(
                injrec["ampl_rec"].max(),
                flares["ampl_rec"].max(),
                injrec["amplitude"].max()
            ),
            ampl_bins
        )

    if isinstance(dur_bins, int):
        dur_bins = numpy.linspace(
            min(
                injrec["dur"].min(),
                flares["dur"].min(),
                injrec["duration_d"].min()
            ),
            max(
                injrec["dur"].max(),
                flares["dur"].max(),
                injrec["duration_d"].max()
            ),
            dur_bins
        )

    return ampl_bins, dur_bins


def tileUpInjectionRecovery(
    df,
    typ,
    ampl="amplitude",
    dur="duration_d",
    otherfunc="count",
    ampl_bins=numpy.arange(0, 50, 0.025),
    dur_bins=numpy.arange(0, .2*24*60*60, 5e-3*24*60*60)
):
    """
    Tile up the injection recovery data into amplitude and duration bins.
    Return a multi-indexed matrix that can be accessed to assign recovered
    ED/amplitude/duration ratio or recovery probability to a given
    observation (AMPL, DUR) or its recovery corrected form.

    Parameters:

    - `df`: injection recovery table;
    - `typ`: type of inj-rec parameter to obtain. Can be
    `recovery_probability`, `ed_ratio`, `amplitude_ratio`,
    `duration_ratio`;
    - `ampl`: column name used to bin on one parameter axis'
    - `dur`: column name used to bin on the other axis;
    - `otherfunc`: pandas groupby applicable function string
    (`std`, `count`, `mean`, etc). Use this to get another statistic
    on the desired inj-rec parameter that is not median;
    - `ampl_bins`: bins for one axis, should cover both injected
    and recovered range;
    - `dur_bins`: bins for the other axis, should cover both injected
    and recovered range.

    Returns:

    - multi-indexed tiled injection-recovery dataset'
    - column name for relevant parameter.
    """

    # calculate helpful columns
    if "rec" not in df.columns:
        df["rec"] = df["ed_rec"].fillna(0).astype(bool).astype(int)
    if "dur" not in df.columns:
        df["dur"] = df.tstop - df.tstart

    d1 = df.assign(
        Amplitude=pandas.cut(df[ampl], ampl_bins),
        Duration=pandas.cut(df[dur],  dur_bins)
    )

    types = {
        "ed_ratio": ("ed_rec", "ed_inj", "edrat"),
        "amplitude_ratio": ("ampl_rec", "amplitude", "amplrat"),
        "duration_ratio": ("dur", "duration_d", "durrat")
    }

    if typ == "recovery_probability":
        grouped = d1.groupby(["Amplitude", "Duration"])
        d2 = grouped.rec.sum() / grouped.rec.count()
        d3 = getattr(grouped.rec, otherfunc)()
        val = "rec"
    else:
        d1["rel"] = d1[types[typ][0]] / d1[types[typ][1]]
        grouped = d1.groupby(["Amplitude", "Duration"])
        d2 = grouped.rel.median()
        d3 = getattr(grouped.rel, otherfunc)()
        val = types[typ][2]

    return pandas.DataFrame({val: d2, otherfunc: d3}), val


def multiIndexIntoDfWithNans(
    x,
    df,
    i1="ampl_rec",
    i2="dur",
    i3="edrat"
):
    """
    Helps with indexing in multi-indexed tables that also have NaNs.

    Parameters:

    - `x`: row from the flare detection table;
    - `df`: multi-indexed table with NaNs;
    - `i1`, `i2`, `i3`: names of 1st index, 2nd index and value column
    in the table.

    Returns:

    - `float`: value at index given by x.
    """
    try:
        return df.loc[(x[i1], x[i2]), i3]
    except KeyError:
        return numpy.nan


def characterizeFlares(
    flares,
    df,
    otherfunc="count",
    amplrec="ampl_rec",
    durrec="dur",
    amplinj="amplitude",
    durinj="duration_d"
):
    """
    Assign ED recovery ratios, and recovery probability to all flares
    whose recovered parameters are covered by the synthetic data.

    Parameters:

    - `flares`: flare table;
    - `df`: injection-recovery table;
    - `otherfunc`: additional column for statistical analysis. Can accept
    `count`, `std`, and other simple Pandas methods that work
    on groupby objects.
    - `amplrec`: column name for recovered amplitude;
    - `durrec`: column name for recovered duration;
    - `amplinj`: column name for injected amplitude;
    - `durrec`: column name for injected duration.

    Returns:

    - flares with additional columns.
    """

    # define observed flare duration
    if "dur" not in flares.columns:
        flares["dur"] = flares["tstop"] - flares["tstart"]
    if "dur" not in df.columns:
        df["dur"] = df["tstop"] - df["tstart"]

    ds = dict()

    # calculate inj-rec ratio for ED, amplitude, and duration
    for typ in ["ed_ratio", "amplitude_ratio", "duration_ratio"]:
        d, val = tileUpInjectionRecovery(
            df,
            typ,
            otherfunc=otherfunc,
            ampl=amplrec,
            dur=durrec
        )
        d = d.dropna(how="all", axis=0)
        ds[typ] = d

        def helper(x): return multiIndexIntoDfWithNans(
            x,
            d,
            i1="ampl_rec",
            i2="dur",
            i3=val
        )
        flares[typ] = flares.apply(helper, axis=1)

        def helper(x): return multiIndexIntoDfWithNans(
            x,
            d,
            i1="ampl_rec",
            i2="dur",
            i3=otherfunc
        )
        flares["{}_{}".format(typ, otherfunc)] = flares.apply(helper, axis=1)

    # calculate recovery probability from corrected values
    flares["amplitude_corr"] = flares[amplrec] / flares.amplitude_ratio
    flares["duration_corr"] = flares[durrec] / flares.duration_ratio
    flares["ed_corr"] = flares["ed_rec"] / flares.ed_ratio
    d, val = tileUpInjectionRecovery(
        df,
        "recovery_probability",
        otherfunc=otherfunc,
        ampl=amplinj,
        dur=durinj
    )
    d = d.dropna(how="all", axis=0)
    ds["recovery_probability"] = d

    def helper(x): return multiIndexIntoDfWithNans(
        x,
        d,
        i1="amplitude_corr",
        i2="duration_corr",
        i3=val
    )
    flares["recovery_probability"] = flares.apply(
        helper,
        axis=1
    )

    def helper(x): return multiIndexIntoDfWithNans(
        x,
        d,
        i1="amplitude_corr",
        i2="duration_corr",
        i3=otherfunc
    )
    flares["{}_{}".format("recovery_probability", otherfunc)] = flares.apply(
        helper,
        axis=1
    )

    return flares, ds


def wrapCharacterizationOfFlares(
    injrec,
    flares,
    ampl_bins=None,
    dur_bins=None,
    flares_per_bin=None
):
    """
    Take injection-recovery results for a data set and the corresponding
    flare table. Determine recovery probability, ED ratio, amplitude ratio,
    duration ratio, and the respective standard deviation. Count on how many
    synthetic flares the results are based.

    Parameters:

    - `injrec`: table with injection-recovery results from AltaiPony;
    - `flares`: table with flare candidates detected by AltaiPony;
    - `ampl_bins`: number of bins in amplitude;
    - `dur_bins`: number of bins in duration.

    Returns:

    - flares and injrec merged with the characteristics listed above.
    """

    # define observed flare duration
    flares["dur"] = flares["tstop"] - flares["tstart"]

    ampl_bins, dur_bins = setupBins(
        injrec,
        flares,
        ampl_bins=ampl_bins,
        dur_bins=dur_bins,
        flares_per_bin=flares_per_bin
    )

    flares = flares.dropna(subset=["ed_rec"])
    injrec.ed_rec = injrec.ed_rec.fillna(0)
    injrec['rec'] = injrec.ed_rec.astype(bool).astype(float)

    flcc, dscc = characterizeFlares(
        flares,
        injrec,
        otherfunc="count",
        amplrec="ampl_rec",
        durrec="dur",
        amplinj="amplitude",
        durinj="duration_d",
        ampl_bins=ampl_bins,
        dur_bins=dur_bins
    )

    fl, ds = characterizeFlares(
        flares,
        injrec,
        otherfunc="std",
        amplrec="ampl_rec",
        durrec="dur",
        amplinj="amplitude",
        durinj="duration_d",
        ampl_bins=ampl_bins,
        dur_bins=dur_bins
    )

    fl = fl.merge(flcc)
    fl = fl.drop_duplicates()

    fl["ed_corr_err"] = numpy.sqrt(
        fl.ed_rec_err**2
        +
        fl.ed_corr**2
        *
        fl.ed_ratio_std**2
    )

    fl["amplitude_corr_err"] = (
        fl.amplitude_corr
        *
        fl.amplitude_ratio_std
        /
        fl.amplitude_ratio
    )

    fl["duration_corr_err"] = (
        fl.duration_corr
        *
        fl.duration_ratio_std
        /
        fl.duration_ratio
    )

    return fl


def modRandom(
    x,
    d: bool = False,
    seed: Optional[int] = 667
):
    """
    Helper function that generates deterministic random numbers
    if needed for testing.

    Parameters:

    - `d`: flag to set if random numbers shall be deterministic;
    - `seed`: sets the seed value for random number generator.
    """

    if d is True:
        numpy.random.seed(seed)
        return numpy.random.rand(x)
    else:
        numpy.random.seed()  # do not remove, otherwise the seed is fixed
        return numpy.random.rand(x)


def generateFakeFlareDistribution(
    nfake,
    d: bool,
    seed: int,
    ampl=[1e-4, 50],
    dur=[10, 1e6],
    mode="uniform"
):
    """
    Creates different distributions of fake flares to be injected
    into light curves.

    Parameters:

    - `nfake`: number of fake flares to be created;
    - `ampl`: amplitude range in relative flux units;
    - `dur`: duration range in sec;
    - `mode`: distribution of fake flares in (duration, amplitude) space.
    With `uniform` flares are distributed evenly in duration
    and amplitude space.

    Returns

    - `dur_fake`: durations of generated fake flares in days;
    - `ampl_fake`: amplitudes of generated fake flares in relative flux units.
    """
    if mode == "uniform":
        dur_fake = (
            modRandom(nfake, d, seed)
            *
            (dur[1] - dur[0])
            +
            dur[0]
        )
        ampl_fake = (
            modRandom(nfake, d, seed)
            *
            (ampl[1] - ampl[0])
            +
            ampl[0]
        )

    return dur_fake.tolist(), ampl_fake.tolist()


# timeSeries = [1.0, 1.1, 1.2]
# fluxSeries = [2.0, 2.1, 2.2]
# fluxErrorSeries = [3.0, 3.1, 3.2]
# findFlares(timeSeries, fluxSeries, fluxErrorSeries)
