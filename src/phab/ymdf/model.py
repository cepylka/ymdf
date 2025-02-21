import numpy
import pandas
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic
from scipy import special
import progressbar
import astropy.units as units
from astropy import constants
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
import pathlib
import lightkurve

import pandera
from typing import Tuple, List, Literal, Optional, Any


def findGaps(
    lightCurve: pandas.DataFrame,
    maximumGap: float,
    minimumObservationPeriod: int
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
    right = gapOut[1:] - 1

    # drop too short observation periods
    tooShort = numpy.where(numpy.diff(gapOut) < minimumObservationPeriod)
    left = numpy.delete(left, tooShort)
    right = numpy.delete(right, tooShort)

    return list(zip(left, right))


def addPaddingToList(
    lst: list[int],
    padding: int,
    maxAllowedIndex: int
) -> list[int]:
    """
    Padding every element in the list with a certain amount
    of increments/decrements to the right/left of every element.

    Parameters:

    - `lst` - original list of indexes;
    - `padding` - how many new indexes to add to the left and right
    of every original list index;
    - `maxAllowedIndex` - there is actually a yet another list, for which
    this list is a subset of indexes, and we need to care not to pad more
    indexes to the right of the last element here.
    """
    paddedList: list[int] = []
    for i in lst:
        # pad to the left
        paddingIndex = 0
        leftPad: list[int] = []
        while paddingIndex > -padding and i + paddingIndex > 0:
            paddingIndex -= 1
            x = i + paddingIndex
            if x not in lst and x not in paddedList:
                leftPad.insert(0, x)
            else:
                break
        paddedList += leftPad
        paddedList.append(i)
        # pad to the right
        paddingIndex = 0
        rightPad: list[int] = []
        while paddingIndex < padding and i + paddingIndex < maxAllowedIndex:
            paddingIndex += 1
            x = i + paddingIndex
            if x not in lst and x not in paddedList:
                rightPad.append(x)
            else:
                break
        paddedList += rightPad
    return paddedList


def medSig(originalArray: numpy.ndarray):
    """
    Returns median and outlier-robust estimate of standard deviation
    (1.48 x median of absolute deviations).

    Adapted from K2SC (Aigrain et al. 2016).
    """
    # might be redundant, as `a` already comes masked (so without NaNs/INFs)
    lst = numpy.isfinite(originalArray)

    nfinite = lst.sum()
    # unlikely, as for that the `a` should be all NaNs/INFs
    if nfinite == 0:
        return numpy.nan, numpy.nan
    # not very likely either, as for that only one element from `a`
    # should be not NaN/INF
    elif nfinite == 1:
        return originalArray[lst], numpy.nan
    else:
        med = numpy.median(originalArray[lst])
        sig = 1.48 * numpy.median(numpy.abs(originalArray[lst] - med))
        return med, sig


def expandMask(
    msk: numpy.ndarray,
    longdecay: int = 1
):
    """
    Expand the mask if multiple outliers occur in a row. Add
    `sqrt(outliers in a row)` masked points before and after
    the outlier sequence.

    Parameters:

    - `msk`: mask;
    - `longdecay`: optional parameter to expand the mask more by
        this factor after the series of outliers.

    Returns:

    - `array`: expanded mask
    """
    i = j = k = 0

    while i < len(msk):
        v = bool(msk[i])

        if v is False:
            if j == 0:
                k += 1
                j = 1
                i += 1
            else:
                k += 1
                i += 1
        else:
            if j == 0:
                i += 1
            else:
                if k >= 2:
                    addto = int(numpy.rint(numpy.sqrt(k)))
                    msk[i - k - addto:i - k] = False
                    msk[i:i + longdecay * addto] = False
                    i += longdecay * addto
                else:
                    i += 1
                j = 0
                k = 0

    return msk


def sigmaClip(
    originalArray: numpy.ndarray,
    max_iter: int = 10,
    max_sigma: float = 3.0,
    separate_masks: bool = False,
    mexc=None
):
    """
    Iterative sigma-clipping routine that separates not finite points
    and down and upwards outliers.

    Adapted from (Aigrain et al. 2016).

    1: good data point
    0: masked outlier

    Parameters:

    - `originalArray`: flux array;
    - `max_iter`: how often do we want to recalculate sigma
    to get ever smaller outliers;
    - `max_sigma` where do we clip the outliers;
    - `separate_masks`: if true, will give two boolean arrays for positive
    and negative outliers;
    - `mexc`: custom mask to additionally account for.

    Returns:

    - boolean array (all) or two boolean arrays (positive/negative)
    with the final outliers as zeros.

    Example:

    ``` py
    search_result = lightkurve.search_lightcurve(
        "AU Mic",
        author="SPOC"
    )
    lk = search_result[1].download()
    lkTable = lk.to_pandas()
    #print(lkTable.columns)

    x = sigmaClip(lkTable.iloc[:-49587]["flux"].to_numpy())
    xTrue = numpy.where(x == True)
    #print(f"nice values: {xTrue[0].size}") # 40743
    xFalse = numpy.where(x == False)
    #print(f"outliers: {xFalse[0].size}") # 10406
    ```
    """

    # perform sigma-clipping on finite points only,
    # or custom indices given by mexc
    mexc = (
        numpy.isfinite(originalArray)
        if mexc is None
        else numpy.isfinite(originalArray) & mexc
    )

    # initialize different masks for up and downward outliers
    mhigh = numpy.ones_like(mexc, dtype=bool)
    mlow = numpy.ones_like(mexc, dtype=bool)
    mask = numpy.ones_like(mexc, dtype=bool)

    # iteratively (with i) clip outliers above/below (-)max_sigma * sig
    i = 0
    nm = None
    while (
        nm != mask.sum()
        and
        i < max_iter
    ):
        # okay values are finite and not outliers
        mask = mexc & mhigh & mlow

        # safety check if the mask looks fine
        nm = mask.sum()
        if nm > 1:
            # calculate median and MAD adjusted standard deviation
            med, sig = medSig(originalArray[mask])
            # print(med, sig)
            # indices of okay values above median
            mhigh[mexc] = originalArray[mexc] - med < max_sigma * sig
            # indices of okay values below median
            mlow[mexc] = originalArray[mexc] - med > -max_sigma * sig

            # okay values are finite and not outliers
            mask = mexc & mhigh & mlow

            # expand the mask left and right
            mhigh = expandMask(mhigh)
            # print(f"- number of True in mhigh after expandMask: {numpy.where(mhigh == True)[0].size}")

            i += 1

    if separate_masks:
        return mlow, mhigh
    else:
        return mlow & mhigh


def establishWindowLength(
    timeSeries: pandas.Series,
    windowLengthCandidate: int,
    waveRange: Literal["uv", "photo"]
) -> int:
    if waveRange not in ["uv", "photo"]:
        raise ValueError(f"Unknown wave range: {waveRange}")

    dayInSeconds = (
        1
        if waveRange == "uv"
        else 24 * 60 * 60
    )

    # print("windowLengthCandidate",windowLengthCandidate)
    if windowLengthCandidate == 0:
        dt = numpy.nanmedian(
            timeSeries[1:].values / dayInSeconds
            -
            timeSeries[0:-1].values / dayInSeconds
        )
        # print("dt", dt)
        windowLengthCandidate = numpy.floor(0.25 / dt)
        if windowLengthCandidate % 2 == 0:
            windowLengthCandidate += 1
        # print("windowLengthCandidate",windowLengthCandidate)
    # window length must be larger than polyorder
    # print("window", max(windowLengthCandidate, 5))
    return max(windowLengthCandidate, 5)


def detrendSavGol(
    lightCurve: pandas.DataFrame,
    gaps: List[Tuple[int, int]],
    padding: int,
    waveRange: Literal["uv", "photo"],
    windowLength: int = 0
) -> pandas.DataFrame:
    """
    Construct a light curve model. Based on original Appaloosa (Davenport 2016)
    with Savitzky-Golay filtering from `scipy` and iterative `sigmaClip`ping
    adopted from K2SC (Aigrain et al. 2016).

    Parameters:

    - `lightCurve`: light curve;
    - `gaps`: found gaps in series;
    - `padding`: how many new indexes to add to the left and right
    of every original list index;
    - `waveRange`: which wave range the flux belongs to;
    - `windowLength`: number of data points for Savitzky-Golay filter. If you
    don't know your window length, use the default `0` value (*for calculation
    of window length for each gap*).
    """
    if waveRange not in ["uv", "photo"]:
        raise ValueError(f"Unknown wave range: {waveRange}")

    if (
        "fluxDetrended" in lightCurve.columns
        and
        not lightCurve["fluxDetrended"].isna().all()
    ):
        # print("Working with existed detrended flux")
        lightCurve["fluxModel"] = (
            lightCurve["flux"]
            -
            lightCurve["fluxDetrended"]
            +
            numpy.nanmean(lightCurve["fluxDetrended"])
        )
        return lightCurve

    if not isinstance(windowLength, int):
        raise ValueError("Window length must be a number")

    lightCurve["fluxDetrended"] = numpy.array(
        [numpy.nan] * len(lightCurve.index),
        dtype=float
    )

    lightCurve["fluxModel"] = numpy.array(
        [numpy.nan] * len(lightCurve.index),
        dtype=float
    )

    lightCurve["flareTrue"] = False

    gaps = [
        (windowLength, gaps[i][0], gaps[i][1])
        for i in range(len(gaps))
    ]

    if len(gaps) == 0:
        gaps = [
            (windowLength, lightCurve.index[0], lightCurve.index[-1])
        ]

    for (wl, le, ri) in gaps:
        wl = establishWindowLength(
            lightCurve.iloc[le:ri]["time"],
            wl,
            waveRange
        )

        # iterative sigma clipping
        correctValues = numpy.where(
            sigmaClip(lightCurve.iloc[le:ri]["flux"].values)
        )[0] + le

        outliersIndexes = []
        for index in list(range(le, ri)):
            if index not in correctValues:
                outliersIndexes.append(index)
        # print(f"correct values count: {len(correctValues)}")
        # print(f"outliers count: {len(outliersIndexes)}")
        outliers = addPaddingToList(
            outliersIndexes,
            padding,
            max(correctValues)
        )

        betweenGaps = lightCurve[lightCurve.index.isin(correctValues)]

        if betweenGaps.empty:
            continue
        else:
            # print("time", betweenGaps["time"])
            betweenGaps["fluxModel"] = savgol_filter(
                betweenGaps["flux"],
                wl,
                3,
                mode="nearest"
            )
            # print("Detrend", betweenGaps["fluxModel"])

            betweenGaps["fluxDetrended"] = (
                betweenGaps["flux"]
                -
                betweenGaps["fluxModel"]
                +
                numpy.nanmean(betweenGaps["fluxModel"])
            )

        for index, row in lightCurve[le:ri].iterrows():
            if index in outliers:
                lightCurve.at[index, "flareTrue"] = True
            else:
                if index in betweenGaps.index:
                    lightCurve.at[index, "fluxDetrended"] = betweenGaps.at[
                        index,
                        "fluxDetrended"
                    ]
                    lightCurve.at[index, "fluxModel"] = betweenGaps.at[
                        index,
                        "fluxModel"
                    ]

        sta = list(
            numpy.where(
                numpy.diff(lightCurve[le:ri]["flareTrue"]) == 1
            )[0]
        )
        fin = list(
            numpy.where(
                numpy.diff(lightCurve[le:ri]["flareTrue"]) == -1
            )[0]
        )

        # treat outliers at end and start of time series:
        if len(sta) > len(fin):
            fin.append(ri - le - 1)
        elif len(sta) < len(fin):
            sta = [0] + sta
        elif (
            len(sta) == len(fin)
            and
            len(sta) != 0
        ):
            # outliers on both ends
            if (
                sta[0] > fin[0]
                or
                sta[-1] > fin[-1]
            ):
                sta = [0] + sta
                fin.append(ri - le - 1)

        # compute flux model as the mean value between
        # start and end of flare, that is, we interpolate
        # linearly

        medianModel = numpy.nanmean(lightCurve["fluxModel"])
        flareTrueCounts: pandas.Series = lightCurve["flareTrue"].value_counts()
        flareTrueCount: int = (
            flareTrueCounts[True]
            if True in flareTrueCounts.index
            else 0
        )
        # print(f"flareTrueCount: {flareTrueCount}")

        off = 0
        for i, j in list(zip(sta, fin)):
            d = 0
            if j + 2 > ri - le - 1:  # treat end of time series
                k = i
            elif i == 0:
                i = 0
                d = 1
                k = j + 2
            else:
                k = j + 2
            k = min(len(lightCurve[le:ri]), k)

            upper = min(j + d - i + off, flareTrueCount)

            # workaround for a bug that sometimes occurs, not sure why
            # if k == len(lightCurve[le:ri]):
            #     k -= 1

            for index, row in lightCurve[le:ri].iterrows():
                if index >= off and index <= upper:
                    lightCurve.at[index, "fluxModel"] = numpy.nanmean(
                        [
                            lightCurve.at[i, "fluxModel"],
                            lightCurve.at[k, "fluxModel"]
                        ]
                    )
            off += j + d - i

        for index, row in lightCurve[le:ri].iterrows():
            if lightCurve.at[index, "flareTrue"] is True:
                lightCurve.at[index, "fluxDetrended"] = (
                    lightCurve.at[index, "flux"]
                    -
                    lightCurve.at[index, "fluxModel"]
                    +
                    medianModel
                )
        #     else:
        #         lightCurve.at[index, "fluxModel"] = numpy.nanmean(
        #             lightCurve["flux"]
        #         )

    # with pandas.option_context("mode.use_inf_as_null", True):
    lightCurve = lightCurve.replace([numpy.inf, -numpy.inf], numpy.nan)
    # print("fluxDetrended", lightCurve["fluxDetrended"])
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
    n2=3,
    n3=2,
    addtail=False,
    tailthreshdiff=1
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

    # if no local scatter characteristics are given, use formal error as sigma
    if sigma is None:
        sigma = error
    A0 = flux - median  # excursion should be positive # "n0"
    A1 = numpy.abs(flux - median) / sigma  # n1
    A2 = numpy.abs(flux - median - error) / sigma  # n2

    # apply thresholds n0-n2:
    pass_thresholds = numpy.where((A0 > 0) & (A1 > n1) & (A2 > n2))
    # print(pass_thresholds)
    # array of indices where thresholds are exceeded:
    is_pass_thresholds = numpy.zeros_like(flux)
    is_pass_thresholds[pass_thresholds] = 1

    reverse_counts = numpy.zeros_like(flux, dtype="int")
    for k in range(2, len(flux)):
        reverse_counts[-k] = (
            is_pass_thresholds[-k]
            *
            (
                reverse_counts[-(k - 1)]
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
    if addtail == True:
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
            while ((A0[s] > 0) & (A1[s] > n1tail) & (A2[s] > n2tail)):
                s += 1
            newstops.append(s)

        # overwrite old flare stop indices
        istop_i = newstops

    # create boolean flare mask
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
    stop: int
    # error=False #always return errors
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
    stop = int(stop) + 1

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

    # if error == True:
    flare_chisq = chiSquare(
        residual[:-1],
        (
            lct["fluxError"].values[:-1]
            /
            numpy.nanmedian(lct["iterativeMedian"].values)
        )
    )
    edError = numpy.sqrt(ed**2 / (stop-1-start) / flare_chisq)
    return ed, edError
    # else:
    #     return ed


def equivalentDurationModel(
    timeSeries,
    fluxSeries,
    fluxErrorSeries,
    fluxQuiet
    # error=False #always return errors
) -> float | Tuple[float, float]:
    """
    Returns the equivalent duration of a flare event found within
    indices [start, stop], calculated as the area under
    the residual (flux-flux_median).

    Use only on detrended light curves.

    Returns also the uncertainty on ED following Davenport (2016)

    Parameters:

    - `lightCurve`: flare light curve;
    - `error`: if true, then will compute uncertainty on ED.

    Returns:

    - `ed`: equivalent duration in seconds;
    - `edError`: uncertainty in seconds.
    """

    residual = ((
        fluxSeries
        /
        fluxQuiet)
        -
        1.0
    )
    # print(residual)
    x = timeSeries  # in seconds, add `* 60.0 * 60.0 * 24.0` for days
    print(numpy.trapz(residual, x=x))
    ed = numpy.sum(numpy.diff(x) * residual[:-1])

    # # if error == True:
    # flare_chisq = chiSquare(
    #     residual[:-1],
    #     (
    #         fluxErrorSeries[:-1]
    #         /
    #         fluxQuiet
    #     )
    # )
    # edError = numpy.sqrt(ed**2 / (len(timeSeries)-1) / flare_chisq)
    return ed
    # else:
    #     return ed
# def equivalentDurationModelT(
#     timeSeries,
#     fluxSeries,
#     fluxErrorSeries,
#     start,
#     stop
#     # error=False #always return errors
# ) -> float | Tuple[float, float]:
#     """
#     Returns the equivalent duration of a flare event found within
#     indices [start, stop], calculated as the area under
#     the residual (flux-flux_median).

#     Use only on detrended light curves.

#     Returns also the uncertainty on ED following Davenport (2016)

#     Parameters:

#     - `lightCurve`: flare light curve;
#     - `error`: if true, then will compute uncertainty on ED.

#     Returns:

#     - `ed`: equivalent duration in seconds;
#     - `edError`: uncertainty in seconds.
#     """
#     lightCurve = pandas.DataFrame(
#             {
#                 "time": timeSeries,
#                 "flux": fluxSeries,
#                 "fluxError": fluxErrorSeries
#             }
#         )
#         if max(lightCurve["fluxError"]) > min(lightCurve["flux"]):
#             lightCurve["fluxError"] = max(
#                 1e-16,
#                 numpy.nanmedian(
#                     pandas.Series(
#                         lightCurve["flux"]
#                     ).rolling(3, center=True).std()
#                 )
#             ) * numpy.ones_like(lightCurve["flux"])
#         gaps = findGaps(
#             lightCurve,
#             maximumGap=maximumGap,
#             # parameter of min time between gaps is needed
#             # to be elevated to the main function
#             minimumObservationPeriod=minimumObservationPeriod
#         )

#         detrendedLightCurve = detrendSavGolUltraViolet(
#             lightCurve,
#             gaps,
#             padding,
#             5
#         )
#         # print(gaps)

#         if detrendedLightCurve["fluxDetrended"].isna().all():
#             raise ValueError("Finding flares only works on detrended light curves")

#         detrendedLightCurve = findIterativeMedian(
#             detrendedLightCurve,
#             gaps,
#             30
#             )
#     start = int(start)
#     stop = int(stop)+1

#     lct = detrendedLightCurve.loc[start:stop]
#     residual = (
#         lct["fluxDetrended"].values
#         /
#         numpy.nanmedian(lct["iterativeMedian"].values)
#         -
#         1.0
#     )
#     x = lct["time"].values  # in seconds, add `* 60.0 * 60.0 * 24.0` for days
#     ed = numpy.sum(numpy.diff(x) * residual[:-1])

#     flare_chisq = chiSquare(
#         residual[:-1],
#         (
#             lct["fluxError"].values[:-1]
#             /
#             numpy.nanmedian(lct["iterativeMedian"].values)
#         )
#     )
#     edError = numpy.sqrt(ed**2 / (stop-1-start) / flare_chisq)
#     return ed, edError


def findFlareEnergy(
    timeSeries,
    fluxSeries,
    fluxErrorSeries,
    foundFlares: pandas.DataFrame,
    starRadius,
    starDistance,
    maximumGap,
    minimumObservationPeriod,
    padding,
    minimumerror,
):
    # lightCurve = pandas.DataFrame(
    #     {
    #         "time": timeSeries,
    #         "flux": fluxSeries,
    #         "fluxError": fluxErrorSeries
    #     }
    # )
    # if max(lightCurve["fluxError"]) > min(lightCurve["flux"]):
    #     lightCurve["fluxError"] = max(
    #         1e-16,
    #         numpy.nanmedian(
    #             pandas.Series(
    #                 lightCurve["flux"]
    #             ).rolling(3, center=True).std()
    #         )
    #     ) * numpy.ones_like(lightCurve["flux"])
    # gaps = findGaps(
    #     lightCurve,
    #     maximumGap=maximumGap,
    #     # parameter of min time between gaps is needed
    #     # to be elevated to the main function
    #     minimumObservationPeriod=minimumObservationPeriod
    # )

    # detrendedLightCurve = detrendSavGolUltraViolet(
    #     lightCurve,
    #     gaps,
    #     padding,
    #     5
    # )
    # # print(gaps)

    # if detrendedLightCurve["fluxDetrended"].isna().all():
    #     raise ValueError("Finding flares only works on detrended light curves")

    # detrendedLightCurve = findIterativeMedian(
    #     detrendedLightCurve,
    #     gaps,
    #     30
    # )
    luminosity = luminosityQuiescent(
        timeSeries,
        fluxSeries,
        fluxErrorSeries,
        starDistance,
        starRadius,
        maximumGap,
        minimumObservationPeriod,
        padding,
        minimumerror
    )
    foundFlares["luminosity quiscent"] = numpy.array(numpy.NaN, dtype=float)
    foundFlares["flare_erg"] = numpy.array(numpy.NaN, dtype=float)
    foundFlares["flare_erg_rec"] = numpy.array(numpy.NaN, dtype=float)
    for index, row in foundFlares.iterrows():
        foundFlares.at[index, "luminosity quiscent"] = luminosity
        foundFlares.at[index, "flare_erg"] = foundFlares.at[
            index,
            "ed_rec"
        ] * luminosity
        foundFlares.at[index, "flare_erg_rec"] = foundFlares.at[
            index,
            "ed_corr"
        ] * luminosity
    return foundFlares


def luminosityQuiescent(
    timeSeries,
    fluxSeries,
    fluxErrorSeries,
    starDistance,
    starRadius,
    maximumGap,
    minimumObservationPeriod,
    padding,
    minimumerror
):
    integrated_flux = (
        fluxQuiescent(
            timeSeries,
            fluxSeries,
            fluxErrorSeries,
            maximumGap,
            minimumObservationPeriod,
            padding,
            minimumerror
        )
        *
        (
            (starDistance*units.pc).to("cm")
            /
            (starRadius*constants.R_sun.to("cm"))
        )**2
    )

    lum = (
        integrated_flux
        *
        numpy.pi
        *
        (starRadius*constants.R_sun.to("cm")) ** 2
    )

    return lum.value


def fluxQuiescent(
    timeSeries,
    fluxSeries,
    fluxErrorSeries,
    maximumGap,
    minimumObservationPeriod,
    padding,
    minimumerror
):
    lightCurve = pandas.DataFrame(
        {
            "time": timeSeries,
            "flux": fluxSeries,
            "fluxError": fluxErrorSeries
        }
    )
    if max(lightCurve["fluxError"]) > min(lightCurve["flux"]):
        lightCurve["fluxError"] = max(
            minimumerror,
            numpy.nanmedian(
                pandas.Series(
                    lightCurve["flux"]
                ).rolling(3, center=True).std()
            )
        ) * numpy.ones_like(lightCurve["flux"])

    gaps = findGaps(
        lightCurve,
        maximumGap=maximumGap,
        # parameter of min time between gaps is needed
        # to be elevated to the main function
        minimumObservationPeriod=minimumObservationPeriod
    )

    detrendedLightCurve = detrendSavGolUltraViolet(
        lightCurve,
        gaps,
        padding,
        5
    )
    # print(gaps)

    if detrendedLightCurve["fluxDetrended"].isna().all():
        raise ValueError("Finding flares only works on detrended light curves")

    detrendedLightCurve = findIterativeMedian(detrendedLightCurve, gaps, 30)
    return numpy.nanmedian(detrendedLightCurve["iterativeMedian"])


def findFlares(
    lightCurve: pandas.DataFrame,
    n1: int,
    n2: int,
    n3: int,
    minSep: int,
    padding: int,
    maximumGap: float,
    minimumObservationPeriod: float,
    waveRange: Literal["uv", "photo"],
    fluxDetrended: Optional[List[float]] = None,
    sigma: Optional[List[float]] = None
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Obtaining and processing a light curve.

    Parameters:

    - `lightCurve`: pandas table with required columns `time`, `flux`,
    `fluxError` and optional `fluxDetrended` column. If `fluxDetrended`
    is provided, then there will be no calculation of detrended flux;
    - `n1`: how many sigma the flux value is above;
    - `n2`: how many sigma and flux error value the flux value is above;
    - `n3`: number of consecutive points required to flag as a flare;
    - `minSep`: minimum distance between two candidate start times
    in datapoints;
    - `padding` - how many new indexes to add to the left and right
    of every original list index;
    - `maximumGap`: should be larger than step in time series;
    - `minimumObservationPeriod`: minimum number of datapoints in continuous
    observation, without gaps as defined by `maximumGap`;
    - `waveRange`: which wave range the flux belongs to;
    - `sigma`: local scatter of the flux. This array should be the same length
    as the detrended flux array.
    """

    if waveRange not in ["uv", "photo"]:
        raise ValueError(f"Unknown wave range: {waveRange}")

    lightCurveTableSchema = pandera.DataFrameSchema(
        index=pandera.Index(int),  # also check that it is increasing
        columns={
            "time": pandera.Column(float),
            "flux": pandera.Column(float),
            "fluxError": pandera.Column(float),
            "fluxDetrended": pandera.Column(float, required=False)
        }
    )
    lightCurveTableSchema(lightCurve)

    if max(lightCurve["fluxError"]) > min(lightCurve["flux"]):
        lightCurve["fluxError"] = max(
            1e-16,
            numpy.nanmedian(
                pandas.Series(
                    lightCurve["flux"]
                ).rolling(3, center=True).std()
            )
        ) * numpy.ones_like(lightCurve["flux"])

    flares = pandas.DataFrame()

    gaps = findGaps(
        lightCurve,
        maximumGap=maximumGap,
        # parameter of min time between gaps is needed
        # to be elevated to the main function
        minimumObservationPeriod=minimumObservationPeriod
    )

    detrendedLightCurve = detrendSavGol(
        lightCurve,
        gaps,
        padding,
        waveRange,
        windowLength=0
    )

    if detrendedLightCurve["fluxDetrended"].isna().all():
        raise ValueError("Finding flares only works on detrended light curves")

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
            equivalentDuration(detrendedLightCurve, i, j)
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
                "istart": istart,
                "istop": istop,
                "tstart": tstart,
                "tstop": tstop,
                "ed_rec": ed_rec,
                "ed_rec_err": ed_rec_err,
                "ampl_rec": ampl_rec,
                "total_n_valid_data_points": detrendedLightCurve[
                    "flux"
                ].values.shape[0],
                "dur": tstop - tstart
            }
        )

        flares = pandas.concat([flares, newFlare], ignore_index=True)

    flaresTableSchema = pandera.DataFrameSchema(
        {
            "istart": pandera.Column(int, nullable=True),
            "istop": pandera.Column(int, nullable=True)
        }
    )
    flaresTableSchema(flares)

    # print("initial_detrend with Itmed",detrendedLightCurve)
    return (detrendedLightCurve, flares)


def findFakeFlares(
    lightCurve,
    gaps,
    n1: int,
    n2: int,
    n3: int,
    minSep: int,
    maximumGap: float,
    minimumObservationPeriod: float,
    sigma: Optional[list[float]] = None,
    doPeriodicityRemoving: bool = False
):
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

    flares = pandas.DataFrame()
    # print(lightCurve.head(4))
    # gaps = findGaps(lightCurve, 30, 10)

    isFlare = None
    # work on periods of continuous observation with no gaps
    for (le, ri) in gaps:
        temp_lightCurve = lightCurve.iloc[le:ri]
        # print(temp_lightCurve)
        error = temp_lightCurve["fluxDetrendedError"]
        flux = temp_lightCurve["fluxDetrended"]
        median = temp_lightCurve["iterativeMedian"]
        time = temp_lightCurve["time"]
        # A0 = flux - median  # excursion should be positive # "n0"
        # A1 = numpy.abs(flux - median)/error  # n1
        # A2 = numpy.abs(flux - median - error) /error # n2
        # print("outside of isflare",max(A0),max(A1),max(A2))
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
        # print(isFlare)
        temp_lightCurve["isFlare"] = isFlare
        # print(temp_lightCurve)
        # pick out final flare candidate indices
        istart = numpy.array([], dtype="int")
        istop = numpy.array([], dtype="int")
        istart_gap = numpy.array([])
        istop_gap = numpy.array([])
        candidates = numpy.where(temp_lightCurve["isFlare"] == True)[0]
        # print("candidates", candidates)
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
        # print("sep candidates", separated_candidates)

        istart = numpy.array(
            numpy.append(istart + le, istart_gap + le),
            dtype="int"
        )
        istop = numpy.array(
            numpy.append(istop + le, istop_gap + le),
            dtype="int"
        )

        ed_fake = []
        ampl_rec = []

        if len(istart) > 0:
            for (i, j) in zip(istart, istop):
                lct = temp_lightCurve.loc[i:j]
                residual = (
                        lct["fluxDetrended"].values
                        /
                        numpy.nanmedian(lct["iterativeMedian"].values)
                        -
                        1.0
                    )
                x = lct["time"].values  # in seconds, so for days you'll need to add `* 60.0 * 60.0 * 24.0`
                ed_fake_val = numpy.sum(numpy.diff(x) * residual[:-1])
                ed_fake.append(ed_fake_val)

                fl = lightCurve["fluxDetrended"].values
                ampl_rec_val = (
                    numpy.max(fl[i:j])
                    /
                    lightCurve["iterativeMedian"].values[i]
                    -
                    1.0
                )
                ampl_rec.append(ampl_rec_val)

            tstart = lightCurve.iloc[istart]["time"].values
            tstop = lightCurve.iloc[istop]["time"].values

            newFlare = pandas.DataFrame(
                {
                    "istart": istart,
                    "istop": istop,
                    "tstart": tstart,
                    "tstop": tstop,
                    "ed_rec": ed_fake,
                    "ampl_rec": ampl_rec,
                    "total_n_valid_data_points": len(lightCurve["flux"]),
                    "dur": tstop - tstart
                  }
              )

            flares = pandas.concat([flares, newFlare], ignore_index=True)
            # print(flares)

    # flaresTableSchema(flares)

    return flares


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


def flareModelMendoza2022(t, tpeak, fwhm, ampl, upsample=False, uptime=0.5):
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
                (timeup <= tpeak) * (timeup - tpeak) / fwhm > -1.0,
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
        downbins = numpy.concatenate(
            (
                t - dt / 2.0,
                [max(t) + dt / 2.0]
            )
        )
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
    # print("model",flare)
    return flare


def generateRandomPeakTime(tstart, tstop, dtime, mintime, seed=None):
    """
    Parameters:

    - `tstart`: array of tstart of real flares;
    - `tstop`: array of tstop of real flares, same length as tstart;
    - `dtime`: float, time difference inside the gap;
    - `mintime`: float, start time of the gap;
    - `seed`: one seed provided would generate an array of seeds
    like `[seed * 0, ..., seed * nfake]`, where `nfake` is the number
    of injected flares.
    """

    isOK = False
    # keep picking new random peak times for your synthetic flares
    # until it does not overlap with a real one
    while isOK is False:
        # choose a random peak time
        # if you set a seed you will get the same synthetic flares
        # all the time
        if isinstance(seed, int):
            numpy.random.seed(seed * s)
            t0 = (
                numpy.random.rand(1)
                *
                dtime
                +
                mintime
            )[0]
            b = 0
            # are there any real flares happening at peak time?
            # fake flares should not overlap with real ones

            for j in range(len(tstart)):
                if (
                    t0 >= tstart[j]
                    and
                    t0 <= tstop[j]
                ):
                    b += 1
            # print("b",b)
            # number of flares that overlap should be 0
            if b == 0:
                isOK = True

        # if you do note set a seed, the synthetic flares will be
        # randomly distributed
        elif seed is None:
            numpy.random.seed()  # do not remove, otherwise the seed is fixed
            t0 = (
                numpy.random.rand(1)
                *
                dtime
                +
                mintime
            )[0]

            # check if there are there any real flares to deal with
            # at that peak time. Only relevant if there were any flares
            # detected at all

            b = 0
            # are there any real flares happening at peak time?
            # fake flares should not overlap with real ones

            for j in range(len(tstart)):
                if (
                    t0 >= tstart[j]
                    and
                    t0 <= tstop[j]
                ):
                    b += 1
            # print("b",b)
            # number of flares that overlap should be 0
            if b == 0:
                isOK = True
        # no real flares, no trouble
        else:
            isOK = True
        return t0


def split_listic(a, n):
    k, m = divmod(len(a), n)
    return (
        a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)
    )


def injectFakeFlares(
    lc,
    flares,
    gaps,
    ampl,
    dur,
    model="mendoza2022",
    fakefreq=0.0005,#/24/60/60, # we need to allow user to choose flare freq, so elevate to characterise flares
    maxFlaresPerGap=2,
    d=False,
    seed=None
):
    """
    Create a number of events, inject them in to data. Use grid of amplitudes
    and durations, keep ampl in relative flux units. Keep track of energy
    in Equiv Dur. Duration defined in minutes. Amplitude defined multiples
    of the median error.

    Parameters:

    - `lc`: detrended light curve with time, flux and error;
    - `flares`: retrieved flares from initial light curve;
    - `gaps`: gaps in the data;
    - `ampl`: amplitude range for generating fake flares;
    - `dur`: duration for generating fake flares;
    - `model`: the flare model to use (*`mendoza2022` or `davenport2014`*);
    - `fakefreq`: flares per second, but at least one per continuous
    observation period will be injected;
    - `d`: if `True`, a seed for random numbers will be set;
    - `seed`: if `d == True`, seed will be set to this number.

    Returns:

    - `FlareLightCurve` with fake flare signatures.
    """
    # print("before", lc["fluxDetrended"])
    lc["fluxDetrendedError"] = numpy.array(numpy.NaN, dtype=float)

    lc_fake = lc.copy()
    # lc_fake.set_index("time", inplace=True, drop=False)

    # print("fake lccc", lc_fake)

    fake_flares = pandas.DataFrame(
        columns=[
            "istart",
            "istop",
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

    nfakesum = max(
        len(gaps) * maxFlaresPerGap,
        int(
            numpy.rint(
                fakefreq
                *
                (
                    lc_fake["time"].values.max()
                    -
                    lc_fake["time"].values.min()
                )
            )
        )
    )
    # print("nfakesum", len(gaps), maxFlaresPerGap, nfakesum)

    # init arrays for the synthetic flare parameters
    t0_fake = []  # peak times
    ed_fake = []  # ED
    dur_fake = []  # duration
    ampl_fake = []  # amplitude

    # init the synthetic flare counter to allow to point to the right places
    # in the arrays above (XXX_fake, etc)
    # ckm = 0

    # are there real flares to deal with in the gap?
    real_flares_in_gap = pandas.DataFrame()
    for (le, ri) in gaps:
        real_gap = flares.query("`istart` >= @le & `istop` <= @ri")
        real_flares_in_gap = pandas.concat([real_flares_in_gap, real_gap])
    flare_len = len(real_flares_in_gap)

    for s in range(len(gaps)):
        le, ri = gaps[s]
        # error = lc_fake.iloc[le:ri]["fluxError"].values
        # flux = lc_fake.iloc[le:ri]["fluxDetrended"].values
        time = lc_fake.iloc[le:ri]["time"].values
        # print(time)
        nfake = int(nfakesum/len(gaps))

        # generate the time constraints for the flares you want to inject
        mintime, maxtime = numpy.min(time), numpy.max(time)
        dtime = maxtime - mintime

        # generate a distribution of durations and amplitudes
        distribution = generateFakeFlareDistribution(
            nfake,
            d,
            seed,
            ampl,
            dur
        )

        # add the distribution for this observing period
        # to the full list of injected flares
        dur_fake = distribution[0]
        ampl_fake = distribution[1]
        t0 = []
        tstart = real_flares_in_gap["tstart"]
        tstop = real_flares_in_gap["tstop"]

        time_chunks = list(split_listic(time, nfake))
        # print("time_chunks",time_chunks)
        # time_chunks = [time[x:x+int(len(time)/nfake)] for x in range(0, nfake)]
        # generate random peak time, avoid known flares
        # print("time_chunks", max(time_chunks[0]), min(time_chunks[1]))

        for k in range(nfake):
            t0 = generateRandomPeakTime(
                tstart,
                tstop,
                dtime,
                mintime,
                seed=None
            )

            # add the peak time to the list
            # generate the flare flux from the Davenport 2014 model

            if model == "davenport2014":
                fl_flux = flareModelDavenport2014(
                    time_chunks[k].tolist(),
                    t0,
                    dur_fake[k],
                    ampl_fake[k]
                )
            elif model == "mendoza2022":
                fl_flux = flareModelMendoza2022(
                    time_chunks[k].tolist(),
                    t0,
                    dur_fake[k],
                    ampl_fake[k]
                )
            else:
                raise ValueError(f"Unknown flare model: {model}")

            aa = min(time_chunks[k].tolist())
            bb = max(time_chunks[k].tolist())
            temp_lc = lc_fake.query("`time`>= @aa & `time`<= @bb")

            # inject flare in to light curve by adding the flare flux
            for i, (index, row) in enumerate(temp_lc.iterrows()):
                adf = (
                    lc_fake.at[index, "fluxDetrended"]
                    +
                    fl_flux[i] * lc_fake.at[index, "iterativeMedian"]
                )
                lc_fake.at[index, "fluxDetrended"] = adf
                temp_lc.at[index, "fluxDetrended"] = adf

            ed_fake = 0.0
            residual = (
                temp_lc["fluxDetrended"].values
                /
                numpy.nanmedian(temp_lc["iterativeMedian"].values)
                -
                1.0
            )
            ed_fake = numpy.sum(numpy.diff(time_chunks[k]) * residual[:-1])

            fake_flares.at[s+k, "duration_d"] = dur_fake[k]
            fake_flares.at[s+k, "amplitude"] = ampl_fake[k]
            fake_flares.at[s+k, "ed_inj"] = ed_fake
            fake_flares.at[s+k, "peak_time"] = t0

    # error minimum is a safety net for the spline function if `mode == 3`
    lc_fake["fluxDetrendedError"] = max(
        1e-16,
        numpy.nanmedian(
            pandas.Series(
                lc_fake["fluxDetrended"]
            ).rolling(3, center=True).std()
        )
    ) * numpy.ones_like(lc_fake["fluxDetrended"])

    return lc_fake, fake_flares


def sampleFlareRecovery(
    lc,
    flares,
    ampl,
    dur,
    iterations,
    n1,
    n2,
    n3,
    minSep,
    maximumGap,
    minimumObservationPeriod,
    mode=None,
    func=None,
    fakefreq=0.0005,
    path=None,
    maxFlaresPerGap=2
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

    flc = lc.copy()
    # print("sample recovery copy of lc", flc.tail(20))
    fake_flares = pandas.DataFrame()

    gaps = findGaps(
        flc,
        maximumGap=maximumGap,
        # the other place where we need to allow
        # choosing the time between gaps 7200
        minimumObservationPeriod=minimumObservationPeriod
    )

    widgets = [progressbar.Percentage(), progressbar.Bar()]
    bar = progressbar.ProgressBar(
        widgets=widgets,
        max_value=iterations
    ).start()

    for itera in range(iterations):
        lc_fake = flc.copy()

        fake_lc, injs = injectFakeFlares(
            lc_fake,
            flares,
            gaps,
            ampl,
            dur,
            fakefreq=fakefreq,
            maxFlaresPerGap=maxFlaresPerGap
        )
        # fig,ax=plt.subplots(figsize=(15,4))
        # ax.plot(lc["time"],lc["fluxDetrended"])
        # ax.plot(fake_lc["time"],fake_lc["fluxDetrended"])
        # ax.axhline(0., color="k")
        # ax.set_ylim(-1e-13,1e-13)
        # with PdfPages(f'./UV_flux{itera}.pdf') as pdf:
        #     pdf.savefig(fig, bbox_inches='tight')

        recs = findFakeFlares(
            fake_lc,
            gaps,
            n1=n1,
            n2=n2,
            n3=n3,
            minSep=minSep,
            maximumGap=maximumGap,
            minimumObservationPeriod=minimumObservationPeriod
        )

        # merge injected and recovered flares
        len_of_table = len(recs)
        injs["rec"] = numpy.array(numpy.NaN, dtype=float)
        m = 0
        for index, row in recs.iterrows():
            for r in injs.index:
                value_to_put_inbetween = injs.at[r, "peak_time"]
                if (
                    value_to_put_inbetween > recs.at[index, "tstart"]
                    and
                    value_to_put_inbetween < recs.at[index, "tstop"]
                ):
                    recs.at[index, "duration_d"] = injs.at[r, "duration_d"]
                    recs.at[index, "amplitude"] = injs.at[r, "amplitude"]
                    recs.at[index, "ed_inj"] = injs.at[r, "ed_inj"]
                    recs.at[index, "peak_time"] = injs.at[r, "peak_time"]
                    recs.at[index, "rec"] = 1
                    injs.at[r, "rec"] = 0
        for r, row in injs.iterrows():
            if row["rec"] == 0:
                recs.at[len_of_table+m, "duration_d"] = injs.at[r, "duration_d"]
                recs.at[len_of_table+m, "amplitude"] = injs.at[r, "amplitude"]
                recs.at[len_of_table+m, "ed_inj"] = injs.at[r, "ed_inj"]
                recs.at[len_of_table+m, "peak_time"] = injs.at[r, "peak_time"]
                recs.at[len_of_table+m, "rec"] = 0
                m += 1


        bar.update(itera + 1)

        # add to previous runs of sampleFlareRecovery on the same LC
        # or create new table
        if len(fake_flares) > 0:
            fake_flares = pandas.concat([fake_flares, recs], ignore_index=True)
        else:
            fake_flares = recs

    bar.finish()

    return flares, fake_flares


# def setupBins(
#     injrec,
#     flares,
#     ampl_bins=None,
#     dur_bins=None,
#     flares_per_bin=None
# ):
#     """
#     Get amplitude and duration bins.
#     """
#     # did the user give appropriate bins?
#     bins = numpy.array(
#         [
#             bool(ampl_bins is not None),
#             bool(dur_bins is not None)
#         ]
#     )

#     # if only one or no bin is given explicitly,
#     # make sure flares_per_bin is set
#     if ~bins.all() and flares_per_bin is None:
#         raise ValueError(
#             " ".join((
#                 "Give either ampl_bins and dur_bins, or either of",
#                 "the two together with flares_per_bin, or neither of",
#                 "the two but flares_per_bin"
#             ))
#         )

#     # if only one out of [ampl_bins, dur_bins] is specified,
#     # specify the other by fixing the `flares_per_bin`
#     if bins.any() and ~bins.all():
#         # which one is not defined?
#         if ampl_bins is None:
#             b = copy.copy(dur_bins)
#         elif dur_bins is None:
#             b = copy.copy(ampl_bins)

#         # if defined bins are given as array, find length
#         lngh = (
#             b if isinstance(b, float) or isinstance(b, int)
#             else len(b)
#         )

#         # define the other bins accordingly
#         if ampl_bins is None:
#             ampl_bins = int(numpy.rint(injrec.shape[0] / lngh / flares_per_bin))
#         elif dur_bins is None:
#             dur_bins = int(numpy.rint(injrec.shape[0] / lngh / flares_per_bin))
#     # if no bins are specified, choose bins of equal size
#     # with approximately `flares_per_bin` in each bin:
#     elif ~bins.any():
#         bins = int(numpy.rint(numpy.sqrt(injrec.shape[0] / flares_per_bin)))
#         ampl_bins = bins
#         dur_bins = bins

#     # if no flares are given, substitute with fake flares
#     if len(flares) == 0:
#         flares = injrec

#     # set bins according to data
#     if isinstance(ampl_bins, int):
#         ampl_bins = numpy.linspace(
#             min(
#                 injrec["ampl_rec"].min(),
#                 flares["ampl_rec"].min(),
#                 injrec["amplitude"].min()
#             ),
#             max(
#                 injrec["ampl_rec"].max(),
#                 flares["ampl_rec"].max(),
#                 injrec["amplitude"].max()
#             ),
#             ampl_bins
#         )

#     if isinstance(dur_bins, int):
#         dur_bins = numpy.linspace(
#             min(
#                 injrec["dur"].min(),
#                 flares["dur"].min(),
#                 injrec["duration_d"].min()
#             ),
#             max(
#                 injrec["dur"].max(),
#                 flares["dur"].max(),
#                 injrec["duration_d"].max()
#             ),
#             dur_bins
#         )

#     return ampl_bins, dur_bins


# def tileUpInjectionRecovery(
#     fake_flares,
#     typ,
#     otherfunc="count",
#     ampl_bins=10,
#     dur_bins=10,
#     ampl_bins=numpy.arange(0.1, 200., 20.0),
#     dur_bins=numpy.arange(10., 1000., 40.)
# ):
#     """
#     Tile up the injection recovery data into amplitude and duration bins.
#     Return a multi-indexed matrix that can be accessed to assign recovered
#     ED/amplitude/duration ratio or recovery probability to a given
#     observation (AMPL, DUR) or its recovery corrected form.

#     Parameters:

#     - `df`: injection recovery table;
#     - `typ`: type of inj-rec parameter to obtain. Can be
#     `recovery_probability`, `ed_ratio`, `amplitude_ratio`,
#     `duration_ratio`;
#     - `ampl`: column name used to bin on one parameter axis'
#     - `dur`: column name used to bin on the other axis;
#     - `otherfunc`: pandas groupby applicable function string
#     (`std`, `count`, `mean`, etc). Use this to get another statistic
#     on the desired inj-rec parameter that is not median;
#     - `ampl_bins`: bins for one axis, should cover both injected
#     and recovered range;
#     - `dur_bins`: bins for the other axis, should cover both injected
#     and recovered range.

#     Returns:

#     - multi-indexed tiled injection-recovery dataset'
#     - column name for relevant parameter.
#     """

#     # calculate helpful columns
    # if "rec" not in fake_flares.columns:
    #     for index, row in fake_flares.iterrows():
    #         if pandas.notna(row["ed_rec"]):
    #             fake_flares.at[index,"rec"] = 1.0
    #         else:
    #             fake_flares.at[index,"rec"] = 0.0
        # fake_flares["rec"] = fake_flares["ed_rec"].fillna(0).astype(bool).astype(int)
#     if "dur" not in fake_flares.columns:
#         fake_flares["dur"] = fake_flares["tstop"] - fake_flares["tstart"]

#     d1 = fake_flares.assign(
#         Amplitude=pandas.cut(fake_flares["amplitude"], ampl_bins),
#         Duration=pandas.cut(fake_flares["duration_d"], dur_bins)
#     )

#     types = {
#         "ed_ratio": ("ed_rec", "ed_inj", "edrat"),
#         "amplitude_ratio": ("ampl_rec", "amplitude", "amplrat"),
#         "duration_ratio": ("dur", "duration_d", "durrat")
#     }

#     if typ == "recovery_probability":
#         grouped = d1.groupby(["Amplitude", "Duration"])
#         d2 = grouped["rec"].sum() / grouped["rec"].count()
#         # d3 = getattr(grouped["rec"], otherfunc)()
#         d3 = d1.apply(lambda d: (d['Amplitude'], d['Duration']), axis=1).value_counts()
#         val = "rec"
#     else:
#         d1["rel"] = d1[types[typ][0]] / d1[types[typ][1]]
#         print(d1["rel"])
#         grouped = d1.groupby(["Amplitude", "Duration"])
#         d2 = grouped["rel"].median()
#         d3 = getattr(grouped["rel"], otherfunc)()
#         val = types[typ][2]
#     print("tiledUp", pandas.DataFrame({val: d2, otherfunc: d3}), val)
#     # print("amplrat",types["amplitude_ratio"][2])
#     return pandas.DataFrame({val: d2, otherfunc: d3}), val


# def multiIndexIntoDfWithNans(
#     x,
#     df,
#     i1="ampl_rec",
#     i2="dur",
#     i3="edrat"
# ):
#     """
#     Helps with indexing in multi-indexed tables that also have NaNs.

#     Parameters:

#     - `x`: row from the flare detection table;
#     - `df`: multi-indexed table with NaNs;
#     - `i1`, `i2`, `i3`: names of 1st index, 2nd index and value column
#     in the table.

#     Returns:

#     - `float`: value at index given by x.
#     """
#     try:
#         return df.loc[(x[i1], x[i2]), i3]
#     except KeyError:
#         return numpy.nan


# def findIntervalValue(value1, value2, df, typ):
#     # Find the interval for value1 in the first level of the index
#     interval1 = None
#     for idx in df.index.get_level_values(0).unique():
#         if isinstance(idx, pd.Interval) and idx.left < value1 <= idx.right:
#             interval1 = idx
#             break

#     # Find the interval for value2 in the second level of the index
#     interval2 = None
#     for idx in df.index.get_level_values(1).unique():
#         if isinstance(idx, pd.Interval) and idx.left < value2 <= idx.right:
#             interval2 = idx
#             break

#     # If both intervals are found, return the corresponding values
#     if interval1 is not None and interval2 is not None:
#         try:
#             return df.loc[(interval1, interval2), 'recovery_probability'], df.loc[(interval1, interval2), 'counts']
#         except KeyError:
#             return None, None

#         if interval1_match and interval2_match:
#             return df.at[index, typ], df.at[index, 'counts']

#     return None, None # Return None if no matching interval is found


def findIntervalValue(value1, value2, df, typ):
    # find the interval for value1 in the first level of the index
    interval1 = next(
        (
            idx for idx in df.index.get_level_values(0).unique()
            if (
                isinstance(idx, pandas.Interval)
                and
                idx.left < value1 <= idx.right
            )
        ),
        None
    )

    # find the interval for value2 in the second level of the index
    interval2 = next(
        (
            idx for idx in df.index.get_level_values(1).unique()
            if (
                isinstance(idx, pandas.Interval)
                and
                idx.left < value2 <= idx.right
            )
        ),
        None
    )

    # if both intervals are found, return the corresponding values
    if interval1 is not None and interval2 is not None:
        try:
            return (
                df.loc[(interval1, interval2), typ],
                df.loc[(interval1, interval2), "counts"]
            )
        except KeyError:
            return None, None
    else:
        return None, None


def get_interval_mid(x):
    """
    Find the midpoint of the interval to use as ticks.
    """

    if isinstance(x, pandas.Interval):
        return x.mid
    elif pandas.isna(x):
        return numpy.nan
    else:
        return x


def plotHeatmap(
    data,
    val,
    label=None,
    ID=None,
    valcbr=(0.0, 1.0),
    # ovalcbr=(0, 50),
    xlabel="duration [s]",
    ylabel="amplitude",
    cmap="viridis",
    font_scale=1.5,
    interpolate=False
):
    """
    Plot a heatmap from the "fake_flares" table.

    Parameters:

    - `data`: `fake_flares` attribute or equivalent table;
    - `val`: column name in `data` to map;
    - `label`: human-readable version of `val`;
    - `ID`: target ID;
    - `valcbr`: value range for `val`;
    - `xlabel`: x label for plot;
    - `ylabel`: y label for plot;
    - `cmap`: colormap, default is `viridis`;
    - `font_scale`: set the size of tick labels and bar label;
    - `interpolate`: ?.

    Returns:

    - `matplotlib.figure.Figure`.
    """

    seaborn.set(font_scale=font_scale)
    df = data.copy()

    # calculate midpoints for both index levels
    level_0_mids = df.index.get_level_values(0).map(get_interval_mid)
    level_1_mids = df.index.get_level_values(1).map(get_interval_mid)

    # add the midpoints as new columns
    df["Amplitude"] = level_0_mids
    df["Duration"] = level_1_mids

    # if you want to add the midpoints to the index instead
    df = df.set_index(["Amplitude", "Duration"], append=False)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,7))

    if label is None:
        label = val

    heatmap1_data = pandas.pivot_table(
        df,
        values=val,
        index=["Amplitude"],
        columns=["Duration"]
    )

    if interpolate == True:
        heatmap1_data.iloc[:, :] = (
            heatmap1_data.bfill(axis=0).values
            +
            heatmap1_data.ffill(axis=0).values
            +
            heatmap1_data.bfill(axis=1).values
            +
            heatmap1_data.ffill(axis=1).values
        ) / 4
        heatmap1_data = heatmap1_data.bfill(
            axis=0
        ).ffill(axis=0).bfill(axis=1).ffill(axis=1)

    try:
        heatmap = seaborn.heatmap(
            heatmap1_data.values,
            cbar_kws={"label": label},
            vmin=valcbr[0],
            vmax=valcbr[1],
            annot=False,
            ax=ax,
            yticklabels=[
                "{:.2e}".format(x) for x in heatmap1_data.index.values
            ],
            xticklabels=[
                "{:.2e}".format(x) for x in heatmap1_data.columns.values
            ]
        )
    except AttributeError:
        heatmap = seaborn.heatmap(
            heatmap1_data.values,
            cbar_kws={"label": label},
            vmin=valcbr[0],
            vmax=valcbr[1],
            annot=False,
            ax=ax,
            yticklabels=[
                "{:.2e}".format(x) for x in heatmap1_data.index.values.categories.values.mid.values
            ],
            xticklabels=[
                "{:.2e}".format(x) for x in heatmap1_data.columns.values.categories.values.mid.values
            ]
        )

    fig = heatmap.get_figure()

    # Do some layout stuff

    fig.tight_layout()
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(ID, fontsize=16)

    with PdfPages(f"./heatmap_{val}.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight")

    return fig


def characterizeFlares(
    flares,
    fake_flares,
    ampl,
    dur,
    otherfunc="count",
    plot_heatmap=True
):
    """
    Assign ED recovery ratios, and recovery probability to all flares
    whose recovered parameters are covered by the synthetic data.

    Parameters:

    - `flares`: flare table;
    - `fake_flares`: injection-recovery table;
    - `ampl`: ?;
    - `dur`: ?;
    - `otherfunc`: additional column for statistical analysis. Can accept
    `count`, `std`, and other simple Pandas methods that work
    on groupby objects.
    - `plot_heatmap`: produce heatmaps for `recovery_probability` and `ed_ratio`.

    Returns:

    - flares with additional columns.
    """

    # not used anywhere yet
    amplrec = "ampl_rec",
    durrec = "dur",
    amplinj = "amplitude",
    durinj = "duration_d"

    # define observed flare duration
    if "dur" not in flares.columns:
        flares["dur"] = flares["tstop"] - flares["tstart"]
    if "dur" not in fake_flares.columns:
        fake_flares["dur"] = fake_flares["tstop"] - fake_flares["tstart"]
    if "rec" not in fake_flares.columns:
        for index, row in fake_flares.iterrows():
            if pandas.notna(row["ed_rec"]):
                fake_flares.at[index, "rec"] = 1.0
            else:
                fake_flares.at[index, "rec"] = 0.0

    ampl_bins = numpy.arange(ampl[0], ampl[1], (ampl[1] - ampl[0])/15.0)
    dur_bins = numpy.arange(dur[0], dur[1], (dur[1] - dur[0])/15.0)
    # print("fake_flares", fake_flares)
    fake_flares_cop = fake_flares.query(
        "`amplitude`.notna() & `duration_d`.notna()"
    )

    fake_flares_cop = fake_flares_cop.assign(
        Amplitude=pandas.cut(fake_flares_cop["amplitude"], ampl_bins),
        Duration=pandas.cut(fake_flares_cop["duration_d"], dur_bins)
    )
    # fake_flares.dropna(how='any', subset=['Amplitude','Duration'], axis=1)
    types = {
        "ed_ratio": ("ed_rec", "ed_inj", "ed_ratio"),
        "amplitude_ratio": ("ampl_rec", "amplitude", "amplitude_ratio"),
        "duration_ratio": ("dur", "duration_d", "duration_ratio")
    }

    # print("all flares injected and recovered", fake_flares_cop)
    fake_flares_cop.to_pickle("./fake_flares.pkl")
    fake_flares_cop.to_csv("./fake_flares.csv")

    for typ in ["recovery_probability"]:
        grouped = fake_flares_cop.groupby(["Amplitude", "Duration"])
        # print(
        #     "grouped", grouped["rec"].mean(),
        #     "count", grouped["rec"].count()
        # )
        d2 = grouped["rec"].mean()  # .sum() #/ grouped["rec"].count()

        print(d2)

        d3 = fake_flares_cop.apply(
            lambda d: (d["Amplitude"], d["Duration"]), axis=1
        ).value_counts()

        print(d3)
        probability = pandas.DataFrame(
            {"recovery_probability": d2, "counts": d3}
        )
        # probability = probability.loc[probability.index.notna().all(level=None)]
        # probability = probability.reindex(probability.index.dropna())

        # reset the index if needed
        # probability = probability.reset_index()
        print(probability, type(probability))
        print("typ", typ, "rec", probability[typ])

        for index, row in flares.iterrows():
            dura = flares.at[index, "dur"]
            amp = flares.at[index, "ampl_rec"]
            print(dura, amp)
            pavule, counts = findIntervalValue(amp, dura, probability, typ)
            print(pavule)
            flares.at[index, "recovery_probability"] = pavule#/counts

        # probability["recovery_probability"] = (
        #     probability["recovery_probability"] / probability["counts"]
        # )
        if plot_heatmap == True:
            plotHeatmap(
                probability,
                "recovery_probability",
                label="Recovery Probability",
                ID=None,
                valcbr=(0.0, 1.0),
                # ovalcbr=(0,50),
                xlabel="duration [s]",
                ylabel="amplitude",
                cmap="viridis",
                font_scale=1.5,
                interpolate=False
            )

    for typ in ["ed_ratio", "amplitude_ratio", "duration_ratio"]:
        fake_flares_cop["rel"] = (
            fake_flares_cop[types[typ][0]]
            /
            fake_flares_cop[types[typ][1]]
        )

        grouped = fake_flares_cop.groupby(["Amplitude", "Duration"])
        d2 = grouped["rel"].median()
        # d2 = d2.dropna(how="all", axis=0)
        d3 = fake_flares_cop.apply(
            lambda d: (d["Amplitude"], d["Duration"]), axis=1
        ).value_counts()

        ratio = pandas.DataFrame({typ: d2, "counts": d3})
        ratio = ratio.reindex(ratio.index.dropna())

        print(ratio)
        print("typ", typ, "rec", ratio[typ])
        for index, row in flares.iterrows():
            dura = flares.at[index, "dur"]
            amp = flares.at[index, "ampl_rec"]
            savule, counts = findIntervalValue(amp, dura, ratio, typ)
            print(savule)
            flares.at[index,typ] = savule

        if plot_heatmap == True and typ == "ed_ratio":
            plotHeatmap(
                ratio,
                "ed_ratio",
                label="ED Ratio",
                ID=None,
                valcbr=(0.0, 1.0),
                # ovalcbr=(0,50),
                xlabel="duration [s]",
                ylabel="amplitude",
                cmap="viridis",
                font_scale=1.5,
                interpolate=False
            )

    # calculate recovery probability from corrected values

    flares["amplitude_corr"] = flares["ampl_rec"] / flares["amplitude_ratio"]
    flares["duration_corr"] = flares["dur"] / flares["duration_ratio"]
    flares["ed_corr"] = flares["ed_rec"] / flares["ed_ratio"]

    return flares


# def wrapCharacterizationOfFlares(
#     fake_flares,
#     flares,
#     ampl_bins=None,
#     dur_bins=None,
#     flares_per_bin=None
# ):
#     """
#     Take injection-recovery results for a data set and the corresponding
#     flare table. Determine recovery probability, ED ratio, amplitude ratio,
#     duration ratio, and the respective standard deviation. Count on how many
#     synthetic flares the results are based.

#     Parameters:

#     - `injrec`: table with injection-recovery results from AltaiPony;
#     - `flares`: table with flare candidates detected by AltaiPony;
#     - `ampl_bins`: number of bins in amplitude;
#     - `dur_bins`: number of bins in duration.

#     Returns:

#     - flares and injrec merged with the characteristics listed above.
#     """

#     # define observed flare duration
#     flares["dur"] = flares["tstop"] - flares["tstart"]

#     ampl_bins, dur_bins = setupBins(
#         fake_flares,
#         flares,
#         ampl_bins=ampl_bins,
#         dur_bins=dur_bins,
#         flares_per_bin=flares_per_bin
#     )

#     flares = flares.dropna(subset=["ed_rec"])
#     injrec['ed_rec'] = injrec['ed_rec'].fillna(0)
#     injrec['rec'] = injrec['ed_rec'].astype(bool).astype(float)

#     flcc, dscc = characterizeFlares(
#         flares,
#         injrec,
#         otherfunc="count",
#         amplrec="ampl_rec",
#         durrec="dur",
#         amplinj="amplitude",
#         durinj="duration_d",
#         ampl_bins=ampl_bins,
#         dur_bins=dur_bins
#     )

#     fl, ds = characterizeFlares(
#         flares,
#         injrec,
#         otherfunc="std",
#         amplrec="ampl_rec",
#         durrec="dur",
#         amplinj="amplitude",
#         durinj="duration_d",
#         ampl_bins=ampl_bins,
#         dur_bins=dur_bins
#     )

#     fl = fl.merge(flcc)
#     fl = fl.drop_duplicates()

#     fl["ed_corr_err"] = numpy.sqrt(
#         fl["ed_rec_err"]**2
#         +
#         fl["ed_corr"]**2
#         *
#         fl["ed_ratio_std"]**2
#     )

#     fl["amplitude_corr_err"] = (
#         fl["amplitude_corr"]
#         *
#         fl["amplitude_ratio_std"]
#         /
#         fl["amplitude_ratio"]
#     )

#     fl["duration_corr_err"] = (
#         fl["duration_corr"]
#         *
#         fl["duration_ratio_std"]
#         /
#         fl["duration_ratio"]
#     )

#     return fl


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

    if d == True:
        numpy.random.seed(seed)
        return numpy.random.rand(x)
    else:
        numpy.random.seed()  # do not remove, otherwise the seed is fixed
        return numpy.random.rand(x)


def generateFakeFlareDistribution(
    nfake,
    d: bool,
    seed: int,
    ampl,
    dur,
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
    # print(dur_fake.tolist(), ampl_fake.tolist())
    return dur_fake.tolist(), ampl_fake.tolist()


def fitsToPandas(
    fitsFile: str
) -> pandas.DataFrame:
    """
    Openning a FITS file and creating a Pandas table from it.
    """
    fitsFilePath: pathlib.Path = pathlib.Path(fitsFile)
    if not fitsFilePath.exists():
        raise ValueError(f"The path [{fitsFilePath}] does not exist")
    if not fitsFilePath.is_file():
        raise ValueError(f"The path [{fitsFilePath}] is not a file")

    lc = Table.read(fitsFilePath)

    # FITS stores data in big-endian, but pandas works with little-endian,
    # so the byte order needs to be swapped
    # https://stackoverflow.com/a/30284033/1688203
    narr = numpy.array(lc).byteswap().newbyteorder()
    pndraw = pandas.DataFrame(narr)
    # print(pndraw.columns)

    flux = pandas.DataFrame(
        columns=[
            "time",
            "flux",
            "fluxError"
        ]
    )

    flux["time"] = pndraw["TIME"]
    flux["flux"] = pndraw["PDCSAP_FLUX"]
    flux["fluxError"] = pndraw["PDCSAP_FLUX_ERR"]

    return flux


def tessLightkurveToPandas(
    lightKurve: lightkurve.lightcurve.TessLightCurve
) -> pandas.DataFrame:
    """
    Converting lightkurve object to Pandas table.

    There are 2 ways of obtaining the lightkurve object.

    First one is to download it from MAST:

    ``` py
    name = "Karmn J07446+035"
    search_result = lightkurve.search_lightcurve(
        name,
        author="SPOC",
        cadence="short",
    )
    lk = search_result[0].download()
    ```

    And second one is to open an already downloaded FITS file:

    ``` py
    lk = lightkurve.TessLightCurve.read("./some.fits")
    ```
    """
    pndraw = lightKurve.to_pandas()
    # print(pndraw.columns)

    flux = pandas.DataFrame(
        columns=[
            "time",
            "flux",
            "fluxError"
        ]
    )

    flux["time"] = pndraw.index
    flux["flux"] = pndraw["pdcsap_flux"]
    flux["fluxError"] = pndraw["pdcsap_flux_err"]

    return flux
