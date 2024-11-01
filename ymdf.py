import numpy
import pandas
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter

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

    dt = numpy.diff(lightCurve.index)

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


def detrendSavGolUltraViolet(
    lightCurve: pandas.DataFrame,
    gaps: List[Tuple[int, int]],
    windowLength: int
) -> pandas.DataFrame:
    """
    Construct a light curve model. Based on original Appaloosa (Davenport 2016)
    with Savitzky-Golay filtering from `scipy` and iterative `sigma_clipping`
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
            sigma_clip(lightCurve.iloc[le:ri]["flux"])
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
        good = sigma_clip(flux)
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
) -> int | Tuple[int, int]:
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
    timeSeries: list[float],
    fluxSeries: list[float],
    fluxErrorSeries: list[float],
    doPeriodicityRemoving: bool = False,
    minSep: int = 3,
    sigma: Optional[list[float]] = None
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Obtaining and processing a light curve.

    Parameters:

    - `timeSeries`: a list of time values;
    - `fluxSeries`: a list of flux values;
    - `fluxErrorSeries`: a list of fluxes errors;
    - `doPeriodicityRemoving`: whether to remove periodicity or not;
    - `minsep`: minimum distance between two candidate start times
    in datapoints;
    - `sigma`: local scatter of the flux. This array should be the same length
    as the detrended flux array.
    """

    seriesLength = len(timeSeries)
    if seriesLength != len(fluxSeries) or seriesLength != len(fluxErrorSeries):
        raise ValueError("The length of all the series must be the same")

    # ---

    lightCurveTableSchema = pandera.DataFrameSchema(
        index=pandera.Index(float),  # also check that it is increasing
        columns={
            "flux": pandera.Column(float),
            "fluxError": pandera.Column(float)
        }
    )
    lightCurve = pandas.DataFrame(
        {
            "flux": fluxSeries,
            "fluxError": fluxErrorSeries
        },
        index=timeSeries
    )
    lightCurveTableSchema(lightCurve)

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

    # work on periods of continuous observation with no gaps
    for (le, ri) in gaps:
        error = detrendedLightCurve.iloc[le:ri]["error"].values
        flux = detrendedLightCurve.iloc[le:ri]["fluxDetrended"].values
        median = detrendedLightCurve.iloc[le:ri]["iterativeMedian"].values
        time = detrendedLightCurve.iloc[le:ri]["time"]

        # run final flare-find on DATA - MODEL

        isFlare = None
        if sigma is None:
            isFlare = findFlaresInContObsPeriod(flux, median, error)
        else:
            isFlare = findFlaresInContObsPeriod(
                flux,
                median,
                error,
                sigma=sigma[le:ri]
            )

    # pick out final flare candidate indices
    candidates = numpy.where(isFlare > 0)[0]
    istart = None
    istop = None
    istartGap = numpy.array([])
    istopGap = numpy.array([])
    if len(candidates) > 0:
        # find start and stop index, combine neighboring candidates
        # in to same events
        separatedCandidates = numpy.where(
            (numpy.diff(candidates)) > minsep
        )[0]
        istartGap = candidates[
            numpy.append([0], separatedCandidates + 1)
        ]
        istopGap = candidates[
            numpy.append(separatedCandidates, [len(candidates) - 1])
        ]

        # stitch indices back into the original light curve
        #
        istart = numpy.array(
            numpy.append(istart, istartGap + le), dtype="int"
        )
        flares["istart"] = istart
        #
        istop = numpy.array(
            numpy.append(istop, istopGap + le), dtype="int"
        )
        flares["istop"] = istop
        # print(f"Found {len(istartGap)} candidates in the ({le},{ri}) gap")
    else:
        print(f"No candidates were found in the ({le},{ri}) gap")

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
        tstart = detrendedLightCurve.index[istart]
        tstop = detrendedLightCurve.index[istop]

        newFlare = pd.DataFrame(
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

        flares = pd.concat([flares, newFlare], ignore_index=True)

    flaresTableSchema(flares)

    return (detrendedLightCurve, flares)


timeSeries = [1.0, 1.1, 1.2]
fluxSeries = [2.0, 2.1, 2.2]
fluxErrorSeries = [3.0, 3.1, 3.2]
findFlares(timeSeries, fluxSeries, fluxErrorSeries)
